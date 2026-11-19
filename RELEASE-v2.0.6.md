# ProxiChorizo 2.0.6

Faster Blake2b hashing on selected NVIDIA and AMD cards, new GTX 1080 Ti support,
and an optional OpenCL mode for much lower CPU use.

New exact-integer hash profiles move invariant work out of the nonce loop and tune
launch sizes for the tested cards. All twelve Blake2b rounds are retained.
Selection is automatic; other cards and layouts retain their existing kernels,
including the fast Blackwell path. The miner does not change clocks or power settings.

Matched Linux A/B/B/A comparisons against the published **2.0.5** miner, using
369-byte jobs and each board's full firmware power allowance:

| GPU | Allowance | 2.0.5 GH/s | 2.0.6 candidate GH/s | Gain | Sampled draw, before → after |
|---|---:|---:|---:|---:|---:|
| RTX 3070, sample A | 250 W | 3.229 | 3.241 | +0.38% | 234 → 238 W |
| RTX 3070, sample B | 250 W | 3.250 | 3.266 | +0.48% | 222 → 226 W |
| RTX 4070 Ti SUPER | 285 W | 6.699 | 6.773 | +1.11% | 285 → 284 W |
| AMD Instinct MI50* | 225 W | 2.660 | 2.797 | +5.15% | 221 → 221 W |
| AMD Radeon RX 7600 XT* | 165 W | 2.051 | 2.069 | +0.85% | 163 → 163 W |

NVIDIA used 45-second runs with the first 15 seconds excluded and five complete
5-second reporting windows retained per run. *AMD measurements used cooled,
short runs: 20 seconds on MI50 (10 measured seconds) and 11 seconds on RX 7600 XT
(5 measured seconds), excluding the first 5 seconds. Longer full-power runs
reached the test rigs' junction-temperature guards, including with 2.0.5;
these AMD figures are **not sustained thermal qualification**. Small gains can
vary with clocks, cooling and job layout. The [per-run CSV](benchmarks/hash-2.0.6-platform-2026-11-19.csv)
includes binary hashes, sampled power/clocks and measurement details. These initial
comparisons used platform-specific CUDA validation builds and the integrated
OpenCL candidate; they are not guaranteed pool rates or Windows measurements.

The integrated CUDA build also retained Blackwell performance in the same
45-second ABBA check; all previously shipped native GPU functions and the
Blackwell PTX kernels retain identical instructions. Differences below are
within the observed run variation, not additional speed improvements:

| GPU | Full allowance | 2.0.5 GH/s | 2.0.6 check GH/s | Difference |
|---|---:|---:|---:|---:|
| RTX 5090 | 600 W | 19.949 | 19.867 | -0.41% |
| RTX 5080 | 400 W | 10.665 | 10.665 | 0.00% |
| RTX 5070 Ti | 350 W | 8.901 | 8.888 | -0.14% |

**GTX 1080 Ti support:** the CUDA packages now include SM61. A tested GTX 1080 Ti
reached **3.256 GH/s** at its 330 W allowance (about 304 W drawn). Its new profile
was 2.85% faster than the generic path in the same candidate. Version 2.0.5 did
not contain an SM61 image, so this is new support, not a 2.0.5 speed comparison.
Pascal automatically uses the verified fe51 compatibility path for legacy
initialization; unsupported legacy fe32 overrides are refused.

**Optional low-CPU OpenCL mode:** on the RX 7900 XTX, setting
`PX_OPENCL_LOW_CPU=1` reduced CPU use from 100.2% to 6.4% of one core in a matched
full-worker test. Throughput was 6.026 → 6.000 GH/s (-0.44%). It is opt-in so the
faster default remains available. Both paths check completion errors and verify
shares on the CPU. OpenCL now also accepts `--token-file` and `PROXI_TOKEN`, like
CUDA, and creates GPU contexts only for the cards selected with `-d`.

Packages cover Linux CUDA/OpenCL, native Windows NVIDIA/AMD, HiveOS, mmpOS, SMOS
and NVIDIA/AMD Docker images. Physical Windows GPU execution remains unqualified.
The published CUDA and OpenCL kernel sources compile directly, with no kernel
encryption or obfuscation. ProxiChorizo License 1.0 is unchanged.


Get started: https://proxi.suprnova.cc/StartMining.html

Pool: `stratum+tcp://proxi.suprnova.cc:3355` or `stratum+ssl://proxi.suprnova.cc:3356`.
Use your registered pool token and worker name. Solo mining uses your own
`proxi node mine` controller.

Docker: `ocminersupr/proxichorizo:2.0.6` / `:latest`, and
`:2.0.6-amd` / `:amd`; mirrored at `ghcr.io/ocminer/proxichorizo`.
