#!/usr/bin/env python3
"""Verify GPU diagnostic output against Python's independent BLAKE2b."""
import argparse
import hashlib
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('kind', choices=['blackwell', 'pascal'])
parser.add_argument('output', type=Path)
args = parser.parse_args()
checked = 0
for line in args.output.read_text().splitlines():
    if args.kind == 'blackwell':
        plen, mutated, nonce, word = line.split()
        plen, mutated = int(plen), int(mutated)
        assert plen in (120, 248, 504, 632) and mutated in (0, 1)
    else:
        nonce, word = line.split()
        plen, mutated = 504, 0
    assert len(nonce) == 16 and len(word) == 16
    prefix = bytearray((i * 37 + 11) & 255 for i in range(plen - 120))
    prefix += bytes(120)
    if mutated:
        prefix[plen - 17] = 1
    digest = hashlib.blake2b(prefix + bytes.fromhex(nonce), digest_size=32).digest()
    assert int.from_bytes(digest[24:], 'little') == int(word, 16), (plen, mutated, nonce)
    checked += 1
assert checked == (67696 if args.kind == 'blackwell' else 8192), checked
print(f'PASS: {checked} independent hashlib hit checks ({args.kind})')
