# ProxiChorizo 1.0.1

Bug-fix release. Recommended for every rig that mines with more than one NVIDIA GPU in one miner process.

## Fixed

- **Duplicate shares with several NVIDIA GPUs and `--autotune`.** When one miner process ran several NVIDIA GPUs with `--autotune`, every GPU searched the same nonce range. Each share was found by every card, and all copies after the first were rejected (`22 duplicate share`), about half of all shares on a two-GPU rig. All hashrate and share counters appeared on GPU #0, while the other GPUs showed `0.00 MH/s` and no sensor readings. Each GPU now gets its own nonce range again, and the miner checks at startup that every GPU has its own range, whichever tuning path set it up.
- Without `--autotune` (platform defaults or the cached first-start tune), the nonce ranges were already separate. AMD/OpenCL (`proxichorizo-cl`) was not affected and has the same startup check now.

Verified on two RTX 3070 in one process with `--autotune`: 1.0.0 showed 33 accepted, 33 duplicate rejected, one GPU at 25.9 MH/s and the other at 0. 1.0.1 shows both GPUs mining (12.9 + 13.0 MH/s), 38 accepted, 0 rejected.

Kernels, hashrates and the stratum protocol are unchanged from 1.0.0.
