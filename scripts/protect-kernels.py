#!/usr/bin/env python3
"""Seal and verify release kernel payloads; key material never enters packages.

XChaCha20-Poly1305, compatible with the vendored Monocypher runtime. Nonces
are keyed content digests, so repeated reproducible builds encrypt identical
messages identically; different payloads use different nonces.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parents[1]
MAGIC = b"PX_KERNEL_MANIFEST_V1_20261115".ljust(32, b"\0")
TABLE_SIZE = 40 + 16 * 64

def key_bytes():
    value = os.environ.get("PX_KERNEL_BUILD_KEY", "")
    if not value:
        raise ValueError("PX_KERNEL_BUILD_KEY is required (32 random bytes as hex)")
    key = bytes.fromhex(value)
    assert len(key) == 32, "invalid build key length"
    return key

def key_header(path):
    key = key_bytes()
    mask = hashlib.blake2b(b"ProxiChorizo key share v1", key=key, digest_size=32).digest()
    def array(name, data):
        return "static volatile const unsigned char %s[32] = {%s};\n" % (name, ",".join(map(str,data)))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("// Generated build-only key shares. Do not distribute.\n" + array("px_key_a", mask) + array("px_key_b", bytes(a^b for a,b in zip(key,mask))))
    path.chmod(0o600)

def sections(data):
    if data[:4] == b"\x7fELF":
        assert data[4:6] == b"\x02\x01", "expected ELF64 little-endian"
        h = struct.unpack_from('<16sHHIQQQIHHHHHH', data)
        sh = [struct.unpack_from('<IIQQQQIIQQ', data,h[6]+i*h[11]) for i in range(h[12])]
        st = sh[h[13]]; strings = data[st[4]:st[4]+st[5]]
        return [(bytes(strings[s[0]:]).split(b'\0',1)[0].decode(), s[4], s[5]) for s in sh if s[1] != 8]
    assert data[:2] == b'MZ', 'expected ELF or PE executable'
    pe = struct.unpack_from('<I',data,0x3c)[0]
    assert data[pe:pe+4] == b'PE\0\0'
    n, optional = struct.unpack_from('<H',data,pe+6)[0], struct.unpack_from('<H',data,pe+20)[0]
    out=[]
    for i in range(n):
        off=pe+24+optional+i*40; name=data[off:off+8].split(b'\0',1)[0].decode()
        size,pos=struct.unpack_from('<II',data,off+16);out.append((name,pos,size))
    return out

def cuda_payloads(data):
    out=[]
    for name,start,size in sections(data):
        if not name.startswith('.nv_fat'): continue
        pos=start;end=start+size
        while pos < end:
            pos=data.find(b'\x50\xed\x55\xba',pos,end)
            if pos < 0: break
            magic,version,header,n=struct.unpack_from('<IHHQ',data,pos)
            assert version==1 and header==16 and 0<n<=1<<30 and pos+header+n<=end, 'invalid fatbin header'
            out.append((pos+header,n,1));pos+=header+n
    return out

def manifest(data):
    off=data.find(MAGIC)
    assert off>=0 and data.find(MAGIC,off+1)<0, 'missing or duplicate protection manifest'
    assert off+TABLE_SIZE<=len(data)
    return off

def inspect_bytes(data,kind):
    off=manifest(data);count,version=struct.unpack_from('<II',data,off+32)
    assert version==1 and 0<count<=16, 'unsealed kernel payloads'
    entries=[]
    for i in range(count):
        size,nonce,mac,k=struct.unpack_from('<Q24s16sI',data,off+40+64*i)
        assert size and any(nonce) and any(mac) and k in (1,2,3)
        entries.append((size,k,nonce,mac))
    assert len({(n,k) for n,k,*_ in entries})==count, 'ambiguous payload identity'
    cuda=cuda_payloads(data)
    if kind=='cuda':
        assert len(cuda)==count>=2 and all(k==1 for _,k,*_ in entries), 'missing CUDA payload'
        assert sorted((n,k)for _,n,k in cuda)==sorted((n,k)for n,k,*_ in entries)
        for pos,n,_ in cuda:
            assert data[pos:pos+4]!=b'\x7fELF' and b'.version ' not in data[pos:pos+n], 'plaintext CUDA code'
    else:
        assert count==2 and {k for _,k,*_ in entries}=={2,3} and not cuda
        for f in ['proxichorizo.cl','txid_blake2b.cl']:
            assert (ROOT/'opencl'/f).read_bytes() not in data, 'plaintext OpenCL source'
    return entries

def seal(path,kind,receipt):
    from nacl.bindings import crypto_aead_xchacha20poly1305_ietf_encrypt as enc
    from nacl.bindings import crypto_aead_xchacha20poly1305_ietf_decrypt as dec
    key=key_bytes(); data=bytearray(path.read_bytes());off=manifest(data)
    assert struct.unpack_from('<II',data,off+32)==(0,1), 'already sealed'
    blobs=cuda_payloads(data) if kind=='cuda' else []
    if kind=='opencl':
        for k,f in [(2,'proxichorizo.cl'),(3,'txid_blake2b.cl')]:
            plain=(ROOT/'opencl'/f).read_bytes();pos=data.find(plain)
            assert pos>=0 and data.find(plain,pos+1)<0, 'missing or duplicate OpenCL source'
            blobs.append((pos,len(plain),k))
    assert 0<len(blobs)<=16 and len({(n,k) for _,n,k in blobs})==len(blobs)
    rows=[]
    for i,(pos,n,k) in enumerate(blobs):
        plain=bytes(data[pos:pos+n]);ad=struct.pack('<QI',n,k)
        nonce=hashlib.blake2b(ad+plain,key=key,digest_size=24).digest()
        sealed=enc(plain,ad,nonce,key); cipher,mac=sealed[:-16],sealed[-16:]
        assert dec(sealed,ad,nonce,key)==plain
        data[pos:pos+n]=cipher
        struct.pack_into('<Q24s16sI12x',data,off+40+i*64,n,nonce,mac,k)
        rows.append({'offset':pos,'size':n,'kind':k,'plaintext_sha256':hashlib.sha256(plain).hexdigest(),'ciphertext_sha256':hashlib.sha256(cipher).hexdigest()})
    struct.pack_into('<I',data,off+32,len(blobs));inspect_bytes(data,kind)
    path.write_bytes(data)
    result={'binary':path.name,'kind':kind,'sha256':hashlib.sha256(data).hexdigest(),'payloads':rows,'authenticated_roundtrip':'PASS'}
    if receipt:receipt.write_text(json.dumps(result,indent=2)+'\n')
    print('Protected',path.name,kind,len(blobs),'payloads; authenticated roundtrip PASS')

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['key-header','seal','verify']);p.add_argument('path',type=Path);p.add_argument('--kind',choices=['cuda','opencl']);p.add_argument('--receipt',type=Path)
    a=p.parse_args()
    if a.action=='key-header':key_header(a.path)
    elif a.action=='seal':assert a.kind;seal(a.path,a.kind,a.receipt)
    else:assert a.kind;inspect_bytes(a.path.read_bytes(),a.kind);print('Protection envelope verified:',a.path.name,a.kind)
if __name__=='__main__':main()
