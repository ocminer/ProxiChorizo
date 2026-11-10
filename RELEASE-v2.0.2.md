# ProxiChorizo 2.0.2

Faster measured RTX 5090 Blake2b workloads and fixes for solo mining with the
current Proxima wallet controller. The miner remains compatible with Stratum
pools and legacy VRF jobs.

- **RTX 5090:** matched development benchmarks against v2.0.1 gained
  **1.49–2.53%** across the selected message layouts. For the current 369-byte
  mainnet layout: **16.57 → 16.91 GH/s at 600 W (+2.06%)**,
  **15.23 → 15.59 GH/s at 450 W (+2.38%)**, and a second card measured
  **14.26 → 14.54 GH/s at 420 W (+1.94%)**. Other GPU models and unmeasured
  layouts retain their existing kernels. The historical 600-byte synthetic
  benchmark is unchanged.
- **Solo:** better solutions are no longer discarded after 256 candidates.
  Candidate selection, expired jobs and delayed responses are handled correctly.
  Blake2b solo mining needs the wallet's seeker endpoint and token; the GPU miner
  does not need the wallet key. See the updated solo guide in `USAGE.md`.
- **Validation:** more than 50 million nonce comparisons matched the complete
  CPU hash. The unmodified upstream v1.6.3 wallet controller produced confirmed
  testnet transits, including after a forced controller restart. Steady miner
  CPU use measured below 1% of one core.

The README and packaged benchmark CSV give message lengths, power settings,
actual power, clocks and measurement limitations. These are matched development
measurements, not guaranteed pool rates or new measurements of every GPU model.
AMD shares the solo host fixes; AMD device kernels are unchanged. Physical AMD
and Windows GPU execution were not requalified for this change.

Downloads cover Linux CUDA and OpenCL, HiveOS, mmpOS/SMOS, and native Windows
NVIDIA. The regular Linux archive includes the mmpOS/SMOS launchers; HiveOS uses
`proxichorizo-hiveos.tar.gz`. Docker images are
`ocminersupr/proxichorizo:2.0.2` / `:latest` for NVIDIA and
`:2.0.2-amd` / `:amd` for AMD, mirrored under `ghcr.io/ocminer/proxichorizo`.

ProxiChorizo License 1.0 and third-party notices are unchanged.
