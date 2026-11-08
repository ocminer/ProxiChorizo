#!/usr/bin/env python3
"""Random txid-blake2b cases for the CPU test: every prefix length 0..760 (all block-boundary shapes),
a few nonces each, hashed with Python's hashlib (RFC 7693). Output: one 'prefixhex noncehex hashhex' per line."""
import hashlib, random, sys
rnd = random.Random(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
out = sys.stdout
for plen in range(0, 761):
    for rep in range(3):
        prefix = bytes(rnd.getrandbits(8) for _ in range(plen))
        nonce = rnd.getrandbits(64) if rep else 0x0102000000000000 | rnd.getrandbits(16)
        msg = prefix + nonce.to_bytes(8, 'big')
        out.write('%s %016x %s\n' % (prefix.hex() or '-', nonce, hashlib.blake2b(msg, digest_size=32).hexdigest()))  # '-' = empty prefix
