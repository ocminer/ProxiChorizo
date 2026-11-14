# ProxiChorizo 2.0.3

Faster Blake2b mining on the NVIDIA RTX PRO 6000 Blackwell Server Edition.

- **RTX PRO 6000:** the aligned nonce search now covers every message layout
  where it is faster. Current 369-byte mainnet layout: **16.20 → 16.59 GH/s
  (+2.4%)** at 600 W in a matched A/B/B/A development benchmark. Across 46
  layouts from 256 to 376 bytes the gain averaged **+1.5%** (best +5.4%).
  Layouts whose prefix length is a multiple of 8 keep the ordinary kernel.
- **Validation:** every layout of the new search was compared with the complete
  CPU hash (2,176 cases, 1.25 million nonces); a short check runs at every start.
  Pool mining on 24 RTX PRO 6000 cards ran with zero rejected shares.
- **Other GPUs:** RTX 5090 keeps the 2.0.2 layouts; other NVIDIA models and AMD
  keep their existing kernels.

The README and the packaged benchmark CSV give message lengths, power, clocks
and measurement limitations. These are development measurements, not guaranteed
pool rates.

Downloads cover Linux CUDA and OpenCL, HiveOS, mmpOS/SMOS, native Windows NVIDIA
and native Windows AMD. Docker images are `ocminersupr/proxichorizo:2.0.3` /
`:latest` for NVIDIA and `:2.0.3-amd` / `:amd` for AMD, mirrored under
`ghcr.io/ocminer/proxichorizo`.

ProxiChorizo License 1.0 and third-party notices are unchanged.
