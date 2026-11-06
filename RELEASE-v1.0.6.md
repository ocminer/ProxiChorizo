# ProxiChorizo 1.0.6

Faster Linux OpenCL mining on the measured AMD cards, plus quicker NVIDIA job
changes. This release includes the Linux/HiveOS/MMPOS/SMOS archives, native Windows
NVIDIA package and NVIDIA Docker images. The miner's license is unchanged.

## AMD / OpenCL

Faster measured defaults for RX 7900 XTX, RX 7600 XT and MI50, exact GPU share verification and sleeping completion waits.

| GPU | Power cap | Published 1.0.5 | Portable 1.0.6 build | Gain |
|---|---:|---:|---:|---:|
| Radeon RX 7900 XTX | 303 W | 12.281 MH/s | **13.628 MH/s** | **+11.0%** |
| Radeon RX 7600 XT | 165 W | 4.052 MH/s | **4.519 MH/s** | **+11.5%** |
| Instinct MI50 | 150 W | 5.753 MH/s | **6.880 MH/s** | **+19.6%** |

Matched published 1.0.5 versus portable 1.0.6 builds, four 60-second A/B/B/A arms per card. All three passed independent math and full-output checks. XTX ROCm 7.14 runtime CPU spin remains; MI50 measured at 150 W because the earlier 225 W trial reached the thermal guard. Windows remains NVIDIA CUDA.

## NVIDIA

The x-only mining pipeline can stop an obsolete launch when the pool sends a new
job. Workers retain sleeping waits. The Blackwell ladder's instruction order is
also adjusted. Fixed-job rates on eight tested GPUs remain within 0.23% of 1.0.5;
see the README and packaged CSV for the measured rates. Live job-change checks
returned 113 accepted shares, zero rejected and one stale across four GPUs.

New-job abort: when the pool sends a new job, the running GPU batch for the old job
is stopped (`PX_NO_ABORT=1` restores the 1.0.5 behaviour). In a live pool A/B on our
own rig the GPU time lost per job change fell from about 200 ms to about 100 ms on the
RTX 3070 and from 134 ms to 77 ms on the RTX 5080 (RTX PRO 6000: 153 ms to 73 ms).
That is roughly +1.0 % useful work on the RTX 3070, +0.6 % on the RTX 5080 and +0.75 %
on the RTX PRO 6000 at the pool's job cadence; `--bench` uses one fixed job and does not
show it. The job check costs about 1 to 1.5 % more of one CPU core per GPU.
The reordered sm_120 ladder adds +0.38 % on the RTX PRO 6000 and about +0.2 % on the
RTX 5090 in `--bench`.

## Validation and packages

The portable AMD build passed independent VRF, field arithmetic, complete-output
and torsion-point checks on all three measured models. Final AMD live checks:
78 accepted, zero rejected, zero stale. CPU reference/profile checks and Linux
startup/TCP/TLS checks passed. NVIDIA correctness checks passed on Blackwell,
Ada, Ampere and CMP 170HX. Windows GPU hardware was not tested.

Every archive includes build metadata, license/notice files, documentation and
internal SHA-256 checksums. The release also provides outer archive checksums.
AMD/OpenCL is included in the Linux and HiveOS archives; native Windows and Docker
remain NVIDIA CUDA packages. An installed OpenCL runtime is required for AMD.

The XTX host's ROCm 7.14 event-thread CPU spin remains a runtime limitation.
MI50 results use a 150 W cap; the earlier 225 W development trial reached the
junction-temperature guard and was excluded. This qualification is not a long
soak or a claim of support for every untested AMD model.

## Benchmark documentation correction

All eight NVIDIA benchmark rows used the full portable binaries of both versions.
The README and CSV frozen into the v1.0.6 build incorrectly label the RTX 4070 Ti
SUPER comparison as using architecture-specific builds. The current repository
README and CSV correct that provenance label; the rates, gains and miner binaries
are unchanged.
