#!/usr/bin/env python3
"""Exercise startup authentication without a GPU on Linux and Windows."""
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

binary=Path(sys.argv[1]).resolve();receipt=json.loads(Path(sys.argv[2]).read_text())
data=bytearray(binary.read_bytes())
assert hashlib.sha256(data).hexdigest()==receipt['sha256']
magic=b'PX_KERNEL_MANIFEST_V1_20261115'.ljust(32,b'\0');off=data.index(magic)
def run(path,expected):
    p=subprocess.run([str(path),'--help'],capture_output=True,timeout=45)
    assert p.returncode==expected,(path.name,p.returncode,p.stderr[-200:])
    if expected:assert b'Kernel protection check failed' in p.stderr
run(binary,0)
with tempfile.TemporaryDirectory(prefix='protection-test-',dir=binary.parent)as td:
    tampered=Path(td)/binary.name
    for payload in receipt['payloads']:
        at=payload['offset']+min(31,payload['size']-1)
        data[at]^=0x80;tampered.write_bytes(data);tampered.chmod(0o700);data[at]^=0x80
        run(tampered,3)
    original=bytes(data[off+32:off+36]);struct.pack_into('<I',data,off+32,0)
    tampered.write_bytes(data);run(tampered,3);data[off+32:off+36]=original
print('Protection startup:',len(receipt['payloads']),'corrupted payloads and unsealed manifest rejected; original starts')
