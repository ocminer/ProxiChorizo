"""Independent Python big-integer point decoding + libsodium scalar multiplication."""
import hashlib, json
from pathlib import Path
from nacl.signing import SigningKey
from nacl.bindings import crypto_scalarmult_ed25519_noclamp, crypto_core_ed25519_add
ROOT=Path(__file__).resolve().parents[1]
P=2**255-19
D=(-121665*pow(121666,P-2,P))%P
I=pow(2,(P-1)//4,P)
def decode(s):
    z=int.from_bytes(s,'little');y=(z&((1<<255)-1))%P;sign=z>>255
    u=(y*y-1)%P;v=(D*y*y+1)%P
    x2=u*pow(v,P-2,P)%P
    x=pow(x2,(P+3)//8,P)
    if x*x%P!=x2:x=x*I%P
    if x*x%P!=x2:return None
    if x%2!=sign:x=(-x)%P
    return x,y
def add(p,q):
    x,y=p;a,b=q;t=D*x*a*y*b%P
    return ((x*b+y*a)*pow(1+t,P-2,P)%P,(y*b+x*a)*pow(1-t,P-2,P)%P)
def encode(p):
    x,y=p
    return (y | ((x&1)<<255)).to_bytes(32,'little')
def vector(seed,alpha):
    pk=bytes(SigningKey(seed).verify_key)
    sc=bytearray(hashlib.sha512(seed).digest()[:32]);sc[0]&=248;sc[31]&=127;sc[31]|=64
    for ctr in range(256):
        dg=hashlib.sha512(b'\x03\x01'+pk+alpha+bytes([ctr,0])).digest()
        h=decode(dg[:32])
        if h is None:continue
        for _ in range(3):h=add(h,h)
        if h==(0,1):continue
        break
    else:raise AssertionError('no point')
    hbytes=encode(h)
    gamma=crypto_scalarmult_ed25519_noclamp(bytes(sc),hbytes)
    cg=gamma
    for _ in range(3):cg=crypto_core_ed25519_add(cg,cg)
    beta=hashlib.sha512(b'\x03\x03'+cg+b'\x00').digest()
    return dict(seed=seed.hex(),pk=pk.hex(),x=bytes(sc).hex(),alpha=alpha.hex(),ctr=ctr,h=hbytes.hex(),cg=cg.hex(),beta=beta.hex())
for line in (ROOT/'vectors.json').read_text().splitlines():
    ref=json.loads(line);got=vector(bytes.fromhex(ref['seed']),bytes.fromhex(ref['alpha']))
    assert all(got[k]==ref[k] for k in ['pk','x','ctr','h','cg','beta']),ref
vectors=[]
for n in range(256):
    seed=hashlib.sha256(('independent 5090 key '+str(n)).encode()).digest()
    for alpha in [bytes(45) if n==0 else hashlib.sha512(('nonce fixture '+str(n)).encode()).digest()[:45],bytes([255])*45 if n==0 else hashlib.sha512(('nonce edge '+str(n)).encode()).digest()[:45]]:
        vectors.append(vector(seed,alpha))
(ROOT/'independent-vectors.jsonl').write_text(''.join(json.dumps(v)+'\n' for v in vectors))
print('PASS independent oracle agrees with16 upstream vectors; generated',len(vectors),'vectors across256 different keys; maximum counter',max(v['ctr'] for v in vectors))
