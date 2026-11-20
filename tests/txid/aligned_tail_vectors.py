#!/usr/bin/env python3
"""Verify public protocol examples using only Python's independent hashlib."""
from pathlib import Path
import hashlib
import json

root = Path(__file__).resolve().parents[2]
vectors = json.loads((root / "vectors-aligned-tail.json").read_text())["vectors"]
assert len(vectors) == 11
for v in vectors:
    prefix = bytes.fromhex(v["prefix_hex"])
    one, two = bytes.fromhex(v["extranonce1"]), bytes.fromhex(v["extranonce2"])
    assert len(one) == 2 and len(two) == 6
    assert len(prefix) == v["prefix_length"] and len(prefix) % 128 == 120
    assert prefix[-120:] == bytes(120)
    digest = hashlib.blake2b(prefix + one + two, digest_size=32).digest()
    work = int.from_bytes(digest[24:], "big")
    bits = (work & -work).bit_length() - 1 if work else 64
    assert digest.hex() == v["hash_hex"] and bits == v["work_bits"]
    assert hashlib.blake2b(prefix + one + two).digest()[:32] != digest
print("PASS: 11 public aligned-tail vectors, full independent BLAKE2b-256 digests and share bit order")
