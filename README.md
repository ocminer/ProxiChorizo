# ProxiChorizo — GPU miner for Proxima (PROX)

A GPU miner for [Proxima](https://lunfardo314.github.io/), by suprnova.cc, for
NVIDIA (CUDA) and AMD (OpenCL). Current mining uses the transaction's BLAKE2b-256
hash. **No miner fee.** Pool mining needs your PROX address and worker name (existing tokens also work);
no wallet key or legacy seed is needed on the GPU machine.

**Download:** [latest release](https://github.com/ocminer/ProxiChorizo/releases/latest) ·
**Pool:** https://proxi.suprnova.cc · **Community:** https://discord.gg/29RHZwvVPN

Linux, Windows NVIDIA/AMD, HiveOS, mmpOS, SMOS and Docker are supported.
TCP (`stratum+tcp://`) and TLS (`stratum+ssl://`) connections, automatic long-job
negotiation and ordinary-job fallback are available on both GPU backends.
[Solo mining](USAGE.md#2-solo-mining--your-own-node-and-proxi-node-mine) against your
own `proxi node mine` controller also supports keyless Blake2b work. Earlier VRF
work remains available for legacy jobs. All kernels remain readable; the miner
does not change clocks, power allowances or fans.

Setup: [current quick start](QUICKSTART-v2.1.2.md) · [usage](USAGE.md) · [HiveOS](hiveos/README.md) ·
[Docker](docker/README.md) · [mmpOS / SMOS](mmpos/README.md) ·
[ready-to-edit examples](examples/README.md).

### 2.1.2 — padded and split-nonce BLAKE2b jobs

On three captured 504-byte padded pool templates, the normal miner gained 1.03% on RTX 5090, 2.68% on RTX 5080 and 3.03% on RTX 5070 Ti versus v2.1.1. On the RTX 5090, the first verified share arrived in 1.21 seconds, down from 2.19 seconds.

| GPU | 504-byte padded job, GH/s | 381-byte split job, GH/s |
|---|---:|---:|
| RTX 5090 | 21.938 → 22.235 | 8.426 → 17.665 |
| RTX 5080 | 11.632 → 11.892 | 4.175 → 8.831 |
| RTX 5070 Ti | 9.702 → 9.913 | 3.461 → 7.306 |
| RTX 3070 | 3.547 → 3.564 | 1.614 → 3.302 |
| GTX 1080 Ti | 3.368 → 3.338 | 1.489 → 3.013 |
| RX 7900 XTX | 7.286 → 7.386 | 3.102 → 6.658 |
| RX 7600 XT | 2.479 → 2.516 | 1.027 → 2.233 |

The table gives fixed-layout Linux benchmark means, from v2.1.1 to the 2.1.2 development kernels. Each timed arm was 15 seconds on NVIDIA, 10 seconds on RX 7900 XTX and five seconds on the cooled RX 7600 XT. The RTX 3070 and GTX 1080 Ti kept their prior padded kernels. In a longer warmed GTX 1080 Ti comparison the difference was −0.045%. The larger gains apply to ordinary prefixes that split the nonce across two compression blocks. All three captured production prefixes were padded. [Individual benchmark arms](benchmarks/proxima-blake2b-2.1.2.csv) · [Release notes](RELEASE-v2.1.2.md)

### 2.1.1 — faster startup and `--no-autotune` (2026-11-20)

Use `--no-autotune` to skip legacy VRF tuning trials. A valid saved result is
still used; otherwise the miner uses its platform default. Mandatory correctness
checks and the short Blake2b kernel comparison remain enabled. It cannot be
combined with `--autotune`.

On HiveOS, add either of these lines to **Extra config**:

```sh
AUTOTUNE=0
```

or, when using the existing extra-argument field:

```sh
EXTRA_ARGS="--no-autotune"
```

For current Blake2b pool mining, leave the legacy **Pass / SEED** field empty.
OpenCL now skips legacy VRF compilation and tuning when mining without a key.
Legacy VRF users still need their key and an initial driver compilation; cached
profiles now compile only the selected variant, rather than compiling a default
variant first. These startup changes do not remove share verification.

### 2.1.1 — long-job benchmarks

The new Blackwell path removes additions of known zero message words and moves
fixed work outside the nonce loop. It still evaluates all twelve Blake2b rounds.
Pascal additionally compares a 512-thread launch with both previous choices.
Each card keeps its old kernel unless the new choice wins its calibration by at
least 2%; ordinary jobs retain their previous paths.

Linux A/B/B/A measurements against the published **2.1.0** executable, using the
same synthetic **504-byte prefix ending in 120 zero bytes**:

| GPU | Full firmware allowance | 2.1.0 | 2.1.1 kernels | Change |
|---|---:|---:|---:|---:|
| RTX 5090 | 600 W | 19.802 GH/s | 21.611 GH/s | +9.13% |
| GTX 1080 Ti, sample 1 | 330 W | 3.146 GH/s | 3.298 GH/s | +4.83% |
| GTX 1080 Ti, sample 2 | 300 W | 3.199 GH/s | 3.339 GH/s | +4.39% |
| GTX 1080 Ti, sample 3 | 300 W | 3.209 GH/s | 3.369 GH/s | +4.97% |
| GTX 1080 Ti, sample 4 | 300 W | 3.008 GH/s | 3.155 GH/s | +4.89% |
| RTX 4070 Ti SUPER | 285 W | 7.351 GH/s | 7.333 GH/s | -0.24%; unchanged within variation |
| RX 7900 XTX | 350 W | 7.205 GH/s | 7.198 GH/s | -0.10%; unchanged within variation |
| Instinct MI50* | 225 W | 3.249 GH/s | 3.245 GH/s | -0.14%; unchanged within variation |

NVIDIA arms lasted 30 seconds, XTX arms 15 seconds. *MI50 used cooled four-second
arms because this test rig cannot sustain the full-power load within its thermal
guard; this is a short diagnostic, not sustained thermal qualification. Clocks
and fan settings were unchanged. The 5090's other miner was compute-paused but
retained its VRAM allocation; Ada retained an idle application context. No other
application was hashing on the measured GPU. Cards reserved for NEB tests were
excluded. The [per-run CSV](benchmarks/long-job-development-2026-11-20.csv) records
executable hashes, individual arms, actual power draw and clocks. These measurements used development executables with the 2.1.1 kernels; their
hashes are retained in the CSV. They are not pool hashrates, earnings estimates
or Windows measurements.

AMD retains its existing fast kernels. A separate same-binary comparison of the
generic and zero-tail paths measured **6.030 → 7.210 GH/s (+19.58%)** on RX 7900 XTX
and **2.670 → 3.251 GH/s (+21.76%)** on MI50, with the same short-run conditions.
Those long-job benefits are already present in 2.1.0; they are not new gains in
2.1.1. RX 7600 XT passed a bounded correctness/startup check,
but its test rig did not qualify for a new sustained performance figure.

The final CUDA startup-fix build has identical GPU instructions to the benchmarked
build. It passed 1,614 independently verified loopback shares across Blackwell,
Pascal and Ada, plus 27 accepted live TLS pool shares with no rejects or invalid
hashes. Its 5090 confirmation run measured 21.640 GH/s. OpenCL startup tests also
passed on XTX, MI50 and RX 7600 XT. The reported ten-card RX 5700 XT configuration
was not available for a physical startup-time measurement. The benchmark and validation records retain their original executable hashes;
release archives identify their own source commit in `BUILD-INFO.json`.

The final published Linux CUDA/OpenCL payloads additionally passed 1,569
independently verified loopback shares across Ada, Pascal and RX 7900 XTX, and
32 accepted live TLS shares with no rejects or invalid hashes. All native CUDA
kernel instructions and Blackwell PTX match the qualified development build;
Linux and Windows kernel instructions also match. Windows startup, CPU and
TCP/TLS checks passed in CI; physical Windows GPU execution was not tested.


### 2.1.0 — aligned-tail jobs on NVIDIA and AMD (2026-11-20)

The miner automatically requests `proxima.aligned-tail.v1` on every platform,
including Pascal and mixed-GPU rigs. Supporting pools can send padded transaction
preimages that simplify the final Blake2b block. Each GPU measures the available
kernels and retains its faster valid path. Ordinary jobs and older pools keep
working; no new command-line option is required.

Pool developers: [PROTOCOL-UPGRADE](PROTOCOL-UPGRADE.md) explains the public wire
format, canonical transaction padding, connection-local job validation, fallback
and interoperability tests. The format does not require an upgrade-7 wait.

Initial Linux development measurements for the same **504-byte padded layout**:

| GPU | Firmware allowance | Existing kernel | Automatic selection | Difference | Sampled draw, before → after |
|---|---:|---:|---:|---:|---:|
| RX 7900 XTX | 350 W | 6.115 GH/s | 7.237 GH/s | +18.35% | 282 → 276 W |
| RTX 4070 Ti SUPER | 285 W | 6.759 GH/s | 7.346 GH/s | +8.69% | 284 → 284 W |
| GTX 1080 Ti | 330 W | 3.177 GH/s | Existing kernel retained | No gain claimed | 145 → 158 W |

The [per-run CSV](benchmarks/aligned-tail-development-2026-11-19.csv) records
clocks, sampled power and executable hashes. These are short eight-second diagnostic runs, not sustained release benchmarks
or a promise of pool earnings. The new zero-tail kernel was slower on the tested
GTX 1080 Ti, so automatic selection kept its existing kernel. Negotiation is still
enabled by default there. Ordinary-layout measurements from 2.0.6 remain below;
different job lengths and clock/thermal conditions must not be conflated.

Development integration checks recorded 3,606 accepted staging shares, zero
CPU-invalid or non-stale rejected shares, and five job-change stales. Sixteen
padded transactions were independently confirmed in the test node's reliable
chain, with their canonical hashes and signatures checked. Pool block rows were
also checked; staging did not execute PPLNS accounting or payouts.

CUDA and OpenCL still verify every submitted hit on the CPU. Multi-GPU padded
jobs keep disjoint nonce ranges even when cards use different kernel enumerations.
Packages include Linux, native Windows NVIDIA/AMD, HiveOS, mmpOS, SMOS and both
Docker variants. All kernels remain readable, without encryption or obfuscation.
The miner never changes power limits, clocks or fan settings.

### 2.0.6 — per-card Blake2b improvements (2026-11-19)

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

### 2.0.5 setup examples — 2026-11-15

Version 2.0.5 ships ready-to-edit setups for mining at suprnova.cc in [`examples/`](examples/README.md)
(Linux, Windows NVIDIA and AMD, HiveOS, mmpOS, SMOS, Docker, systemd), included in every
package. The launchers no longer print the outdated "no SEED/KEY_FILE" notice: current jobs
need only pool login details.

The CUDA and OpenCL builds now compile the published kernel sources directly.
All per-card optimizations from 2.0.4 are retained. Matched Linux A/B/B/A checks
against the published 2.0.4 binaries used 40-second runs, a 369-byte prefix and
unchanged clocks/power settings:

| GPU | Power limit | v2.0.4 GH/s | v2.0.5 check GH/s | Difference |
|---|---:|---:|---:|---:|
| RTX 5090 | 600 W | 19.728 | 19.707 | -0.11% |
| RTX 5080 | 360 W | 10.663 | 10.656 | -0.07% |
| RTX 5070 Ti | 300 W | 8.882 | 8.881 | -0.01% |
| RTX 3070 | 240 W | 3.204 | 3.204 | +0.02% |
| RX 7900 XTX | 303 W | 6.046 | 6.026 | -0.34% |

These are regression checks with platform-specific Linux validation builds,
not a new performance-gain claim or guaranteed pool rates. The small differences
are within the observed run variation. The [measurement CSV](benchmarks/hash-2.0.5-regression-2026-11-15.csv)
records binary hashes, power draw and test details. Full-reference hash checks
passed on all five cards. The 5090 also passed the legacy VRF vectors; NVIDIA
native fallback paths passed. Physical Windows GPU performance remains unmeasured.

### 2.0.4 RTX 50 hash improvements — 2026-11-15

Version 2.0.4 adds a separately compiled Blackwell hash path for the
RTX 5090, RTX 5080 and RTX 5070 Ti. Each card checks it against the CPU reference
at startup. A warmed comparison chooses the kernel for each message layout;
layouts that do not improve keep their native path. Other GPU models keep their
existing selection, and AMD hash calculations are unchanged.

| GPU | Prefix bytes | Power limit | v2.0.3 GH/s | 2.0.4 kernel GH/s | Change | Measured watts, old → new |
|---|---:|---:|---:|---:|---:|---:|
| RTX 5090 | 369 | 600 W | 16.902 | 19.857 | +17.48% | 600 → 600 W |
| RTX 5090 | 369 | 450 W | 15.393 | 16.785 | +9.04% | 450 → 450 W |
| RTX 5080 | 369 | 360 W | 8.351 | 10.671 | +27.78% | 281 → 353 W |
| RTX 5070 Ti | 369 | 300 W | 6.912 | 8.883 | +28.51% | 242 → 297 W |

These are matched A/B/B/A measurements at unchanged clocks and power limits,
using Linux development builds of the kernels selected for 2.0.4. Measured binary
hashes are retained. The 5080 and
5070 Ti use more of their available power with the faster kernel. The
[measurement CSV](benchmarks/hash-blackwell-development-2026-11-15.csv) records
run lengths, clocks, drivers and the measured binary hash. Hashrate depends on
the job's message layout; these short measurements are not guaranteed pool rates.

The optional module is compiled by the installed driver and cached by CUDA.
Its first startup can take longer. If driver compilation is unavailable, the
miner retains its native kernel; a hash-check failure stops startup. Set
`PX_TXID_BW=0` to disable the new path for diagnosis. Windows GPU performance
has not been measured in this development session.

On the tested RX 7900 XTX Linux host, ROCm consumed approximately one CPU core
with both v2.0.3 and v2.0.4. Profiling located this in its async-event runtime;
this release does not resolve that host-specific driver behavior.

### 2.0.3 RTX PRO 6000 improvements — 2026-11-14

Version 2.0.3 uses the faster aligned nonce search on the **RTX PRO 6000 Blackwell
Server Edition** for every message layout where it is faster. RTX 5090 keeps its
2.0.2 layouts; other GPU models keep their existing kernels.

| GPU | Prefix bytes | Power limit | v2.0.2 GH/s | 2.0.3 development GH/s | Change |
|---|---:|---:|---:|---:|---:|
| RTX PRO 6000 Blackwell Server Edition | 369 (current mainnet layout) | 600 W | 16.20 | 16.59 | +2.4% |
| RTX PRO 6000 Blackwell Server Edition | 316 (best measured layout) | 600 W | 16.38 | 17.26 | +5.4% |
| RTX PRO 6000 Blackwell Server Edition | 46 layouts, 256–376 bytes | 600 W | | | +1.5% mean |

The 369-byte row is a matched A/B/B/A comparison (30-second runs); the sweep rows
compare the two kernels in one development build with one run each. Cards drew
about 445 W at automatic clocks (2430 MHz), CUDA 12.9.1, driver 580.178.04. Layouts
whose prefix length is a multiple of 8 keep the ordinary kernel, which measured
0.07–0.33% faster there. Every layout of the new search was checked against the
complete CPU hash (2,176 cases, 1.25 million nonces); a shorter check runs at
every start. Details: [measurement CSV](benchmarks/hash-pro6000-2026-11-14.csv).
These are short development measurements, not guaranteed pool rates.

### 2.0.2 Blake2b improvements — 2026-11-10

Version 2.0.2 improves the measured RTX 5090 job layouts and fixes solo candidate
selection. Other GPU models and unmeasured layouts keep their existing kernels.
The measurements below come from the qualified development builds used to select
these changes; they are not relabeled as final release measurements.

Matched A/B/B/A measurements against the published v2.0.1 Linux binary, using
30-second runs and wall-clock hash counts:

| GPU | Prefix bytes | Power limit | v2.0.1 GH/s | Development GH/s | Change |
|---|---:|---:|---:|---:|---:|
| RTX 5090 A | 353 | 600 W | 16.786 | 17.036 | +1.49% |
| RTX 5090 A | 364 | 600 W | 16.550 | 16.968 | +2.53% |
| RTX 5090 A | 369 (current mainnet layout) | 600 W | 16.567 | 16.907 | +2.06% |
| RTX 5090 A | 436 (solo opening layout) | 600 W | 16.786 | 17.116 | +1.97% |
| RTX 5090 A | 369 | 450 W | 15.229 | 15.592 | +2.38% |
| RTX 5090 B | 369 | 420 W | 14.264 | 14.540 | +1.94% |
| RTX 5090 A | 600 (historical synthetic test) | 600 W | 17.356 | 17.354 | −0.01% (unchanged kernel) |

Card A used the development build; card B used the full development binary, with
identical hash-kernel instructions. Both used CUDA 12.9.1 and automatic GPU clocks.
Memory reported 13,801 MHz. Actual average power, GPU clocks, temperatures and the
spread between repeat runs are in the [measurement CSV](benchmarks/hash-development-2026-11-10.csv).
Card B's baseline varied by 1.32% as the card warmed; both candidate runs exceeded
both baseline runs. These are short matched measurements, not guaranteed pool rates
or a clock-tuning recommendation. Compare versions at the same message length.

Solo now keeps considering better solutions after 256 candidates, orders them by
the complete consensus work value, and handles expired jobs and delayed responses.
With the unmodified upstream v1.6.3 wallet controller and zero CPU search workers,
the development build produced 16 confirmed testnet transits across a forced
controller restart. The full binary produced another 13 confirmed testnet transits
and 4,270 accepted pool shares with zero rejections. Steady miner CPU use was
0.50–0.86% of one core in these tests. More than 50 million nonce checks matched
the complete CPU hash. See the [solo guide](USAGE.md#2-solo-mining--your-own-node-and-proxi-node-mine).

The solo fixes share the CUDA/OpenCL host implementation. AMD hardware and Windows
GPU execution have not been qualified for this development change.

## How to mine (quick start)

**You need:** an NVIDIA GPU with a bundled architecture (SM61/75/80/86/89/90/120, a compatible CUDA 12.9 driver) or an AMD GPU with OpenCL,
Linux x86-64 (also HiveOS, MMPOS, SMOS) or Windows 10/11 x64, and a PROX address
(the 64-hex holder ID your wallet shows as `a/<hex>`).

1. **Use your PROX address** as your login. There is no registration and no key
   download. Copy the 64-hex holder ID from your wallet (omit the optional `a/`
   prefix in the command below). Existing pool tokens remain supported. See the
   [current pool setup](https://proxi.suprnova.cc/StartMining.html).
2. **Download** the release for your system from
   [GitHub Releases](https://github.com/ocminer/ProxiChorizo/releases) and unpack it.
3. **Start the miner** with the pool URL, your address and a worker name:

   Linux
   ```bash
   ./proxichorizo -o stratum+tcp://proxi.suprnova.cc:3355 -u YOUR_PROX_ADDRESS.rig1 -p x
   ```
   Windows (Command Prompt, in the unpacked folder; or edit and run `start.cmd`)
   ```
   proxichorizo.exe -o stratum+tcp://proxi.suprnova.cc:3355 -u YOUR_PROX_ADDRESS.rig1 -p x
   ```
   AMD GPUs: the same with `./proxichorizo-cl` on Linux, or `proxichorizo-cl.exe` from the Windows AMD archive.

### Pool addresses

| | URL |
|---|---|
| Stratum | `stratum+tcp://proxi.suprnova.cc:3355` |
| Stratum over TLS (encrypted) | `stratum+ssl://proxi.suprnova.cc:3356` |

TLS needs OpenSSL 1.1 or 3 on Linux (present on HiveOS, MMPOS, SMOS and every current
distribution); the Windows archive brings its own. The pool's certificate is checked.

### Login

* `-u` / `--user`: `YOUR_PROX_ADDRESS.rig1`; `-p x` is the password for address login.
* Existing token login is still valid: `-u rig1 -p YOUR_EXISTING_TOKEN`. With this
  legacy route, `--token-file FILE` or `PROXI_TOKEN` can keep the token off the
  command line; without `-u`, `--name` supplies the worker name.
* No wallet key, seed or downloaded mining key is needed for current Blake2b jobs.
  Key options remain for legacy VRF work. Keep old `proxima.key` files outside the
  miner directory when you only need keyless hash mining.
* Two rigs may use the same worker name; every connection gets its own nonce range anyway.
  Distinct names show up as separate workers in your statistics.

### Mining OSes and Docker

* **HiveOS:** custom miner `proxichorizo`, installation URL = the `proxichorizo-hiveos.tar.gz`
  of the release; pool URL `stratum+tcp://proxi.suprnova.cc:3355`, wallet/template = your
  PROX address, password empty for Blake2b. Worker name = the rig name. Existing tokens also work.
* **MMPOS / SMOS:** the same tarball, launchers `mmpos-run.sh` / `smos-run.sh`; pool URL as
  above, user = PROX address (or existing token), password empty for Blake2b (see [USAGE.md](USAGE.md)).
* **Docker (NVIDIA):** `ocminersupr/proxichorizo:2.1.2` with `TOKEN=YOUR_PROX_ADDRESS`
  and optionally `POOL_URL` (default `stratum+tcp://proxi.suprnova.cc:3355`); see `docker/README.md`.
* **Docker (AMD):** `ocminersupr/proxichorizo:2.1.2-amd` (also `:amd`), the OpenCL miner with AMD's
  ROCm OpenCL runtime inside; the host needs only the amdgpu driver (`--device /dev/kfd --device /dev/dri`).

### Is it working?

Test the connection first, without mining (no GPU needed):
`./proxichorizo --check-pool stratum+ssl://proxi.suprnova.cc:3356` → `pool OK, …`.
After starting, check for successful authorization, GPU hashrate and accepted
shares. Time to the first share depends on the assigned difficulty and your
hashrate. A few `stale` shares at job changes are normal.
Your workers and earnings: https://proxi.suprnova.cc (enter your PROX address).

### Common problems

| Message | Cause |
|---|---|
| `authorization of … refused … check --token and --user` | check your address/worker and `-p x`, or your existing token |
| `the pool sends jobs for key … but this miner holds …` | legacy VRF key mismatch; current Blake2b jobs need no key |
| `stratum: cannot connect` / reconnecting | network or firewall; the miner retries by itself |
| `TLS handshake … failed` | old OpenSSL, or a proxy breaking TLS; use `stratum+tcp://` |
| `share REJECTED (23 low difficulty share)` | should not happen; report it with the log |

Everything else — options, thermal limits, the stats API, solo mining, troubleshooting — is
in **[USAGE.md](USAGE.md)**, included in every release archive.

Writing your own miner or proxy for the pool? The stratum protocol (messages, nonce layout,
share rule, error codes, a worked example) is specified in **[STRATUM-SPEC.md](STRATUM-SPEC.md)**.

**Older versions:** the pool's old HTTP mining endpoint and 1.x VRF miners are no
longer supported. Use [the current release](https://github.com/ocminer/ProxiChorizo/releases/latest)
with Stratum for current Blake2b mining. This does not remove the miner's separate
solo HTTP controller support.

## Which kernel runs on which card

Current Blake2b work uses the hash kernels for the selected GPU backend and
architecture. Eligible long jobs receive a brief comparison of the existing and
zero-tail paths; Pascal also compares its 512-thread launch. The selected path is
checked before use. `--no-autotune` retains this comparison and correctness checks.

Legacy VRF work has a separate profile table in `platform_profiles.h` (NVIDIA)
and `opencl/proxichorizo_cl.cpp` (AMD). For this older work, unmeasured models can
tune at startup and save the result in `proxichorizo-autotune.cache`, keyed by
physical GPU, processor count, driver API/runtime and miner release.
`--autotune` explicitly requests these legacy trials; `--no-autotune` skips them
and uses a valid cached profile or platform default. Keyless Blake2b startup
does not compile or tune the unused OpenCL VRF programs.

The A100/H100 profiles and NVIDIA ladder-kernel measurements in the historical
VRF tables apply to that older work, not current Blake2b rates. See `PLATFORMS.md`
for measured configurations. The distribution includes native SM90 code for H100.
The miner does not change power limits or clocks.

### Memory clock advisory (hash work)

The txid-blake2b hash search uses little global-memory traffic. When a card runs at its power limit, a lower
memory clock leaves more of that budget to the core. This is optional and set by you, not the miner
(Linux, as root; `nvidia-smi -rmc` undoes it):

```
sudo nvidia-smi -i 0 -lmc 810,810    # GPU 0: memory at 810 MHz while mining
sudo nvidia-smi -i 0 -rmc            # back to automatic
```

Measured with 2.0.5, 369-byte jobs, A/B/B/A ([all readings](benchmarks/memclock-hash-2026-11-15.csv)):

| GPU | Power limit | Memory 810 MHz | Memory 7001 MHz |
|---|---:|---|---|
| RTX 5090 | 450 W (power-limited) | **+1.7 %** (core 2152 → 2197 MHz) | +0.8 % |
| RTX 5080 | 360 W (not at its limit) | same hashrate, 10 W less | same hashrate, 8 W less |
| RTX 3070 | 240 W (not at its limit) | same hashrate, 15 W less | same hashrate (5001 MHz), 7 W less |

The gain appears only where the power limit actually holds the core clock down; otherwise the lower
clock only saves power. **On the tested RTX 5090, avoid 405 MHz memory:** the driver also held the
core near 870 MHz and hashrate fell about 60%. Do not generalize this to untested cards. The older clock observations
further below were taken with the VRF work of 1.0.6 and do not apply to hash jobs.

## Download
[GitHub Releases](https://github.com/ocminer/ProxiChorizo/releases):

**Mining at suprnova.cc:** ready-to-edit start scripts and flight-sheet values for Linux, Windows, HiveOS, mmpOS, SMOS and Docker are in [`examples/`](examples/README.md) (also inside every package). You need your PROX address and a worker name; existing tokens also work.

* `proxichorizo-<ver>.tar.gz` — Linux x86-64, built on glibc 2.31. One tarball serves plain Linux, **HiveOS** (`h-run.sh`), **MMPOS** (`mmpos-run.sh`) and **SMOS** (`smos-run.sh`) — pick the launcher for your OS and set the pool URL and your PROX address (or existing token) in the flight sheet (no key seed needed). Ready-made suprnova.cc setups for every platform are in `examples/`.
* `proxichorizo-hiveos.tar.gz` — the same package under the directory name HiveOS's custom-miner installer requires (miner name `proxichorizo`).
* `proxichorizo-<ver>-windows-x86_64.zip` — native **Windows** x64 for **NVIDIA** (CUDA), with the OpenSSL DLLs and `cacert.pem` for `stratum+ssl://`. Set `TOKEN` to your PROX address or existing token in `start.cmd` and run it.
* `proxichorizo-<ver>-windows-amd-x86_64.zip` — native **Windows** x64 for **AMD** (OpenCL, `proxichorizo-cl.exe`), same layout; needs the AMD Adrenalin driver, which provides `OpenCL.dll`. Windows does not expose AMD temperatures to the miner, so watch them in AMD Software.
* Docker: `ocminersupr/proxichorizo:<ver>` for NVIDIA, `ocminersupr/proxichorizo:<ver>-amd` for AMD (`docker/README.md`).
* `SHA256SUMS` and one `.sha256` per asset; every archive carries its own `SHA256SUMS.txt` and `BUILD-INFO.json`.

## Usage
```
./proxichorizo --proxi stratum+tcp://proxi.suprnova.cc:3355 -u YOUR_PROX_ADDRESS.rig1 -p x   # pool; every GPU, one process
./proxichorizo ... -d 0,2                   # only these GPUs (PCI bus order, as nvidia-smi numbers them)
./proxichorizo --proxi http://your-node:8200 --token SEEKER_TOKEN                                               # Blake2b solo, see USAGE.md
./proxichorizo --autotune --bench 10        # fastest kernel variant per GPU, then its hashrate
./proxichorizo --selftest vectors.json      # the math against the Proxima node's VRF vectors
./proxichorizo ... --api-bind 127.0.0.1:4078   # JSON statistics at /, mmpOS line at /mmpos
./proxichorizo ... --max-temp 85 --log-interval 10 --quiet   # thermal limit per GPU (default 88 C, pauses only that card), console cadence
```
Without a GPU argument the miner takes every GPU in the machine in one process; a GPU that overheats pauses alone and resumes when cool, a GPU with a device error is disabled alone. The console shows a start-up banner, per-GPU lines with temperature, power and clock, share results per GPU and a total line (colours on a terminal, plain text in logs).
Historical VRF measurements (1.0.5, `--bench`, platform defaults; not current Blake2b rates): RTX 5090 ≈ 78.6 MH/s at 600 W / ≈ 67 at 420 W, RTX 5080 ≈ 39.4–40.2 MH/s, RTX 3070 ≈ 14.8–15.1 MH/s, RTX 4070 Ti SUPER ≈ 31.3 MH/s, CMP 170HX ≈ 17.0 MH/s; unchanged from earlier releases: RTX PRO 6000 ≈ 43 MH/s (0.5.6), H100 SXM ≈ 21.6 MH/s (0.5.7), A100 ≈ 12.7 MH/s (0.5.7), RTX A6000 ≈ 12 MH/s (0.5.6); one CPU core of the reference seeker ≈ 37 kH/s. See the [benchmarks](#benchmarks) below.
**[USAGE.md](USAGE.md)** has the full guide: address login at the pool, solo mining with `proxi node mine`, HiveOS / MMPOS / SMOS flight sheets, Docker, Windows, every option, troubleshooting.

## Build

**AMD / other OpenCL GPUs:** `cd opencl && make && ./proxichorizo-cl --selftest && ./proxichorizo-cl --proxi ... --token ...` — needs an OpenCL ICD and headers (ROCm or `ocl-icd-opencl-dev opencl-headers`); the kernels are compiled by the driver at start-up (time depends on the driver and card) and checked against the embedded test vectors before mining. Version 1.0.6 adds faster measured profiles for RX 7900 XTX, RX 7600 XT and MI50; see the [AMD benchmarks](#amd-106-benchmarks) below. Remove old `--groups`/`--iters` overrides to use the automatic profiles. Same options as the CUDA miner (`--groups` instead of `--blocks`). See the AMD section of PLATFORMS.md for rates.

```bash
make                 # needs Python3; all generations, slow
make ARCHS="-gencode arch=compute_86,code=sm_86"   # just your card, fast
make test            # self-test against the Go reference vectors, random field test, cross-variant verify
```

Universal builds, including Pascal SM61, require CUDA 12.8 or 12.9; CUDA 13 removed Pascal compilation. For a newer toolkit, explicitly restrict `ARCHS` to the GPUs it supports. Release builds use the qualified
CUDA 12.9.1 compiler; the optional Blackwell module is embedded as PTX. The binary links the
CUDA runtime statically; the release builds also link libstdc++ statically and are made on
glibc 2.31 so they run on HiveOS and mmpOS images.

## What it does, how fast

Per attempt: SHA-512, try-and-increment hash-to-curve (one field exponentiation per try),
a 254-bit scalar multiplication on Ed25519, cofactor ×8, compression (one inversion),
SHA-512. See `PLATFORMS.md` for the measured variants and why each exists.
Reference points (the same binary, platform defaults): RTX 5090 ≈ 78.6 MH/s (600 W), RTX 5080 ≈ 40 MH/s,
RTX 3070 ≈ 15 MH/s, RTX 4070 Ti SUPER ≈ 31.3 MH/s, CMP 170HX ≈ 17.0 MH/s, one CPU core of the reference Rust seeker ≈ 37 kH/s.

## Benchmarks

### 2.0: hash work (txid-blake2b)

The hash search of library upgrade 6: per job the miner hashes every 128-byte block before
the nonce once; a GPU then needs one BLAKE2b compression per attempt (two when the nonce
crosses a block boundary) and tests only the output word that holds the work. Every share is
hashed again on the CPU before it is sent, and every GPU passes the upgrade-6 test vectors
and a CPU comparison of all 32 nonce positions at start (`--hash-selftest`). Measured with
`--hash-bench` with a **600-byte synthetic prefix**. Actual transaction templates have
different lengths, so these historical rates must be compared at the same prefix length:

| GPU | Backend | Power cap | Hash rate | Measured power | Core clock |
|---|---|---|---|---|---|
| RTX 5090 | CUDA | 575 W | **17.3 GH/s** | 575 W | 2730 MHz |
| RTX 5090 | CUDA | 450 W | **16.1 GH/s** | 450 W | 2542 MHz |
| RTX 5070 Ti | CUDA | 300 W | **7.28 GH/s** | 230 W | 2782 MHz |
| RTX 5080 | CUDA | 360 W | **8.96 GH/s** | 309 W | 2857 MHz |
| RTX 3070 | CUDA | 240 W | **3.33 GH/s** (2.0.0: 3.30) | 231 W | 1905 MHz |
| RX 7900 XTX | OpenCL | 303 W | **6.33 GH/s** | 280 W | 2465 MHz |
| Instinct MI50 | OpenCL | 225 W | **2.81 GH/s** | 220 W | 1690 MHz |
| RX 7600 XT | OpenCL | 165 W | **2.16 GH/s** | 163 W | 2531 MHz |

In 2.0.1 the RTX 3070 (sm_86) search uses a 32-bit nonce loop and precomputes the nonce-free steps
of rounds 0 and 1 per job (+0.6 to +1.3 % depending on the prefix length). On the RTX 5090 and the
RX 7900 XTX the search is bound by the integer ALU, and none of the alternatives we measured (other
rotation forms, multiply-unit adds and rotations, wave64, other work-group sizes, two nonces per thread)
was faster in that investigation, so 2.0.1 retained the 2.0.0 kernels on Blackwell (RTX 50),
OpenCL and the other GPU generations.


### 1.0.7: select kernels by GPU model

Measured GPU models retain their fastest qualified defaults. Other Ampere, Ada and
Blackwell models compare candidate kernels on their own hardware at the first
mining start. This includes the RTX 3060 and 3060 Ti: a 3070 measurement is no longer
treated as a measurement of every SM86 card. The comparison includes the older
release kernels and the current ladder kernel. Autotuning now also tests 256 blocks
per SM, the launch size used by the measured ladder profiles.

Automatic tuning results are kept per physical GPU, SM count, CUDA driver API version
and miner version in `proxichorizo-autotune.cache` in the working directory. If you
change clocks or power settings, remove that cache to repeat automatic tuning, or
use `--autotune`. Explicit kernel and launch overrides still take precedence.
Established A100/H100 profiles are retained. AMD's unmeasured models now also test
the ladder and split kernels with their wide one-iteration launch shapes; the old
AMD search omitted those candidates or measured them with unsuitable small launches.
AMD cache entries include the device identity, CU count and OpenCL driver version.
The measured AMD defaults retain their balance between fixed-job speed and job-change
latency. No clock or power settings are applied by the miner.

#### Final 1.0.7 AMD optimization measurements

RX 7600 XT uses the new three-block Karatsuba field multiply. RX 7900 XTX uses
an equivalent shorter scalar and three final cofactor doublings. Each change is
selected only for the model where it won. MI50 retains its faster existing ladder.
Full portable binaries are compared on the same GPU, cap, key and automatic clocks.

| GPU | Cap | v1.0.6 MH/s | v1.0.7 MH/s | Change | v1.0.7 watts |
|---|---:|---:|---:|---:|---:|
| Radeon RX 7900 XTX | 303 W | 13.529 | 13.625 | +0.71% | 302.5 W |
| Radeon RX 7600 XT | 155 W | 4.461 | 4.571 | +2.46% | 154.5 W |
| Instinct MI50 | 150 W | — | 6.901 | Retained kernel | 149.9 W |

XTX uses a 180-second warm-up and RX 7600 XT uses 60 seconds, each followed by
four 60-second A/B/B/A arms. MI50 is a warmed 60-second final-build verification;
its default device source is unchanged, and no new matched gain is claimed.
All three cards passed the independent VRF vectors, randomized field tests,
pipeline checks and 212,888 extended point outputs for each new kernel.
[Exact measurements, clocks, watts and efficiency](benchmarks/amd-1.0.7.csv).

Mining-mode checks against a local test pool returned 1,628 accepted shares on XTX,
511 on RX 7600 XT and 749 on MI50, all independently
verified with zero invalid or duplicate shares. Stale counts were
0, 1 and 4, respectively; jobs changed during these checks.
Measured miner CPU use was 100.10%, 1.60% and 1.16% of one core, respectively.
The installed XTX ROCm runtime still uses about one CPU core; this limitation is unchanged.
An interrupted 303 W XTX all-kernel tuning trial reached the conservative 95°C
junction guard and is excluded; its isolated tuning/cache check passed at 272 W,
peaking at 82°C. These short qualification runs do not establish long-term stability.

#### Final 1.0.7 NVIDIA measurements

The full portable 1.0.7 build uses the same CUDA device bytes as 1.0.6. These warmed
60-second measurements verify the retained defaults; small differences between
runs reflect boost, temperature and sampling, not a new NVIDIA arithmetic gain.
Each card also passed the independent VRF vectors and randomized field tests.

| GPU | Cap | MH/s | Actual watts | Core / memory MHz |
|---|---:|---:|---:|---:|
| RTX 5090 | 600 W | 78.990 | 599.9 | 2685 / 13801 |
| RTX 5090 | 420 W | 69.040 | 419.9 | 2427 / 13801 |
| RTX 5080 | 360 W | 39.430 | 290.9 | 2797 / 14805 |
| RTX 5070 Ti | 300 W | 33.886 | 248.1 | 2782 / 13801 |
| RTX 4070 Ti SUPER | 285 W | 31.301 | 282.1 | 2778 / 10251 |
| RTX 3070 | 240 W | 14.930 | 230.6 | 1905 / 6801 |
| RTX 3070 sample 2 | 220 W | 14.774 | 217.9 | 1879 / 6801 |
| CMP 170HX | 175 W | 16.443 | 174.3 | 1330 / 1728 |

[Full measurements and efficiency](benchmarks/nvidia-1.0.7.csv). The 175 W CMP
setting was used for thermal headroom; it is distinct from the earlier 200 W
regression audit below. A separate 600 W RTX 5090 mining-mode check against a local test pool returned
2,267 independently verified accepted shares, zero invalid/stale/duplicate shares,
and 2.35% miner CPU use measured as a fraction of one core.

#### Regression audit against 1.0.1

These are fresh measurements of the **published 1.0.1 and 1.0.6 Linux binaries**,
not gains introduced by 1.0.7. Each row uses a warm-up followed by A/B/B/A on the
same physical GPU, unchanged power cap and automatic clocks. NVIDIA uses the same
benchmark target (`k=20`) in both binaries. AMD uses the release benchmark defaults.
The reported 3070 regression was not reproduced on either tested card. Physical
3060 and 3060 Ti cards were unavailable, so their new model-selection behavior is
covered by host tests and their actual speed must be measured on the user's card.

| GPU | Cap | v1.0.1 MH/s | v1.0.6 MH/s | Change | v1.0.6 watts |
|---|---:|---:|---:|---:|---:|
| RTX 4070 Ti SUPER | 285 W | 27.367 | 31.280 | +14.3% | 282.7 W |
| RTX 5090 | 600 W | 67.470 | 78.763 | +16.7% | 600.0 W |
| RTX 5080 | 360 W | 35.606 | 39.433 | +10.7% | 292.5 W |
| RTX 3070 | 240 W | 13.021 | 14.940 | +14.7% | 228.1 W |
| RTX 5070 Ti | 300 W | 29.918 | 33.889 | +13.3% | 249.0 W |
| RTX 5090 (420 W) | 420 W | 54.572 | 67.844 | +24.3% | 419.5 W |
| Radeon RX 7600 XT | 165 W | 4.056 | 4.520 | +11.5% | 165.4 W |
| Radeon RX 7900 XTX | 303 W | 12.163 | 13.520 | +11.2% | 302.6 W |
| RTX 3070 (second card) | 220 W | 12.670 | 14.657 | +15.7% | 218.4 W |
| CMP 170HX | 200 W | 14.762 | 16.947 | +14.8% | 199.4 W |
| Instinct MI50 | 150 W | 5.764 | 6.901 | +19.7% | 148.6 W |

At 810 MHz memory, the two 3070s measured 4.99 / 4.89 MH/s on 1.0.1 and
12.44 / 12.47 MH/s on 1.0.6. Both versions were faster at normal memory clocks.
The table's RX 7600 XT and CMP A/B/B/A arms completed successfully; separate later
kernel trials hit conservative thermal guards and are excluded. These short runs
do not establish long-duration stability. Watts are driver readings and can
slightly exceed the configured cap. Full clocks, watts and efficiency are in the
[comparison CSV](benchmarks/release-regression-2026-11-06.csv).

A separate mining-mode A/B/B/A test on the 220 W RTX 3070 used jobs changing every
0.5–2 seconds, 90 seconds after the first rate sample per arm, and a local test pool
with an independent CPU share verifier. Version 1.0.1 returned 10,472 accepted / 22 stale shares; 1.0.6
returned 12,061 accepted / 93 stale. Both had zero invalid or duplicate shares.
These stochastic counts support the throughput improvement, but are not a precise
hashrate estimate. Miner CPU use was 100.19% versus 1.88% of one CPU core. Benchmark
mode intentionally waits differently and its CPU use should not be read as mining
CPU use.


`--bench` on our own cards with platform defaults and driver-reported power. 1.0.5 numbers are matched A/B/B/A runs
(eight 60 s arms) of the 1.0.5 release build against the published 1.0.4 binary on the same card, power cap and clocks
(RTX 4070 Ti SUPER: single-architecture builds of the 1.0.4 and 1.0.5 sources). Pool rates are a little lower (job
changes, idle gaps).

1.0.5 has a faster default kernel (faster hash-to-curve and point multiplication) on Blackwell, Ada, Ampere GA10x and
the CMP 170HX, with launches of 256 blocks per multiprocessor. The cards draw the same or less power for more hashes.

CUDA CPU use: from 1.0.4 on, each GPU's worker thread sleeps while its GPU works, about 3 % of one CPU core per GPU
(RTX 5090: 3.0 %; one process on 2x RTX 3070 + RTX 5080: 8.8 % in total; same hashrate as with the busy wait).
1.0.3 and earlier kept one full CPU core busy per mining GPU (the same 3-GPU rig: 300 %).

| GPU | Power cap | 1.0.4 | **1.0.5** | Gain over 1.0.4 | Power drawn (1.0.5) | MH/s per W (1.0.5) |
|---|---:|---:|---:|---:|---:|---:|
| RTX 5090 (sm_120, 170 SMs) | 600 W | 70.1 MH/s | **78.6 MH/s** | +12.2 % | 600 W | 0.131 |
| RTX 5090 | 420 W | 56.5 MH/s | **66.9 MH/s** | +18.6 % | 420 W | 0.159 |
| RTX 5080 (sm_120) | 360 W | 37.4 MH/s | **40.2 MH/s** | +7.4 % | 326 W | 0.123 |
| RTX 5080 (second card) | 360 W | 36.7 MH/s | **39.4 MH/s** | +7.4 % | 292 W | 0.135 |
| RTX 4070 Ti SUPER (sm_89) | 285 W | 28.5 MH/s | **31.3 MH/s** | +10.1 % | 282 W | 0.111 |
| RTX 3070 (sm_86) | 240 W | 13.8 MH/s | **15.1 MH/s** | +9.7 % | 230 W | 0.066 |
| RTX 3070 (second card) | 240 W | 13.6 MH/s | **14.8 MH/s** | +9.4 % | 231 W | 0.064 |
| CMP 170HX (sm_80, 74 SMs, default clocks) | 250 W | 15.9 MH/s | **17.0 MH/s** | +7.3 % | 180 W | 0.095 |

Earlier release-over-release gains on the same cards: 1.0.2 over 1.0.1 +3.0 to +5.5 % (1.0.3 and 1.0.4 run the 1.0.2
kernels unchanged).

### Clock and power observations

These sweeps use the published 1.0.6 binary and `k=32` on NVIDIA. They are separate
from the `k=20` regression table above. Core and memory columns show the **actual
sampled clocks**, not the requested lock value. Automatic boost, cooling and board
firmware affect the result. Coarse sweep rows are useful operating points, not a
claim that every possible clock combination was tested.

| GPU / setting | Cap | Core / memory MHz | MH/s | Measured W | kH/s/W |
|---|---:|---:|---:|---:|---:|
| RTX 5090, automatic, warmed | 600 W | about 2670 / 13801 | about 79.6 | 600 | 133 |
| RTX 5090, requested core 2600 | 600 W | 2595 / 13801 | 77.76 | 549.5 | 142 |
| RTX 5090, automatic, matched control | 420 W | 2350 / 13801 | 67.94 | 420 | 162 |
| RTX 5090, memory 7001, matched | 420 W | 2412 / 7001 | 69.04 | 420 | 164 |
| RTX 5080, automatic | 360 W | 2797 / 14801 | 39.95 | 295.0 | 135 |
| RTX 5080, requested core 2600 | 360 W | 2595 / 14801 | 37.09 | 233.6 | 159 |
| RTX 5070 Ti, automatic | 300 W | 2782 / 13801 | 34.30 | 251.6 | 136 |
| RTX 5070 Ti, requested core 2600 | 300 W | 2595 / 13801 | 32.05 | 206.0 | 156 |
| RTX 4070 Ti SUPER, automatic | 285 W | about 2776 / 10251 | 31.70 | 283.4 | 112 |
| RTX 4070 Ti SUPER, core 2700 | 285 W | 2700 / 10251 | 30.87 | 248.0 | 124 |
| RTX 3070, automatic | 240 W | 1920 / 6801 | 15.16 | 232.9 | 65 |
| RTX 3070, core 1800 | 240 W | 1800 / 6801 | 14.29 | 182.7 | 78 |

The 420 W 5090 memory change was confirmed by A/B/B/A: **+1.63%** at the same
power. On the 5070 Ti, 7001 MHz memory reduced power but also reduced hashrate
by 0.68%. At 810 MHz memory every tested NVIDIA model lost hashrate. No automatic
memory lock is enabled by this release. These are VRF-work (1.0.6) observations; for the current
hash work see the memory clock advisory above. [All NVIDIA sweep rows and exact readings](benchmarks/clocks-nvidia-2026-11-06.csv).

### NVIDIA 1.0.6 checks

The 1.0.6 CUDA worker can stop the remainder of an obsolete x-only launch when a
new pool job arrives. It retains sleeping waits. The Blackwell ladder also uses
a revised instruction order. In fixed-job benchmarks, throughput was essentially
unchanged on the eight tested cards (within 0.23% of 1.0.5); these small differences
should not be treated as a guaranteed speed increase.

| GPU | Power cap | 1.0.5 MH/s | 1.0.6 MH/s | Change |
|---|---:|---:|---:|---:|
| RTX 5090 | 600 W | 79.117 | 79.270 | +0.19% |
| RTX 5090 | 420 W | 66.468 | 66.617 | +0.23% |
| RTX 5080 (sample 1) | 360 W | 40.770 | 40.792 | +0.06% |
| RTX 5080 (sample 2) | 360 W | 39.912 | 39.947 | +0.09% |
| RTX 4070 Ti SUPER | 285 W | 31.531 | 31.511 | -0.06% |
| RTX 3070 (sample 1) | 240 W | 15.004 | 14.979 | -0.17% |
| RTX 3070 (sample 2) | 240 W | 15.204 | 15.180 | -0.16% |
| CMP 170HX | 250 W | 17.244 | 17.236 | -0.05% |

Four 60-second arms per version, alternating A/B/B/A/A/B/B/A, measured on
2026-11-06. The two RTX 5090 comparisons discard a separate warm-up arm. All rows use the full portable
builds of both versions. Each row compares the same card and power settings. See the
[CSV](benchmarks/nvidia-1.0.6.csv). Live job-change checks on three GPUs in one process
and a separate RTX 5080 returned 113 accepted shares, zero rejected and one stale.
Those live checks qualify job changes; share counts are not a throughput comparison.

**Job changes.** When the pool sends a new job, 1.0.6 stops the GPU batch that is still
running for the old job instead of finishing it (`PX_NO_ABORT=1` restores the 1.0.5
behaviour). `--bench` mines one fixed job, so the table above does not show this gain.
Live pool A/B on our own rig (one process, 2x RTX 3070 + RTX 5080, 5 minutes per mode,
33 job changes per card), GPU time lost per job change:

| GPU | 1.0.5 | 1.0.6 | Useful work at the pool's job cadence |
|---|---:|---:|---:|
| RTX 3070 (sample 1) | 200.7 ms | 106.6 ms | about +1.0 % |
| RTX 3070 (sample 2) | 197.8 ms | 96.6 ms | about +1.0 % |
| RTX 5080 | 134.1 ms | 77.0 ms | about +0.6 % |
| RTX PRO 6000 | 153 ms | 73 ms | about +0.75 % |

Checking for new jobs during a launch costs about 1 to 1.5 % more of one CPU core per GPU
(roughly 3 to 3.6 % instead of about 2 %).

**Blackwell ladder order.** The reordered sm_120 ladder gives +0.38 % on an RTX PRO 6000
(600 W cap, `--bench`, measured separately with a build that has the same ladder code)
and about +0.2 % on the RTX 5090 (table above).

### AMD clock and power sweep (published 1.0.6)

These are coarse 40-second trials on the same boards used above, with actual driver
clocks and power. Core and memory DPM controls can affect both axes; they are not
isolated memory-bandwidth experiments. The installed RDNA3 interfaces did not expose
fine-grained overdrive controls, and their middle/high core states reached similar
actual clocks. No firmware or driver changes were made.

| GPU / setting | Cap | Actual core / memory MHz | MH/s | Watts | kH/s/W |
|---|---:|---:|---:|---:|---:|
| Radeon RX 7900 XTX / automatic | 303 W | 2489 / 1249 | 13.560 | 302.5 | 44.8 |
| Radeon RX 7900 XTX / memory state 456 MHz | 303 W | 2353 / 456 | 12.712 | 222.6 | 57.1 |
| Radeon RX 7600 XT / automatic | 155 W | 2531 / 1124 | 4.459 | 155.2 | 28.7 |
| Radeon RX 7600 XT / memory state 456 MHz | 155 W | 2554 / 456 | 4.482 | 155.2 | 28.9 |
| Instinct MI50 / automatic | 150 W | 1491 / 716 | 6.911 | 148.2 | 46.6 |
| Instinct MI50 / core 1282 MHz | 150 W | 1282 / 370 | 5.924 | 104.0 | 57.0 |

MI50's 1282 MHz setting trades about 14% throughput for about 30% less GPU-reported
power. RX 7600 XT differences are small and need longer matched testing before
calling them gains. The XTX's manual-DPM trial reduced its core boost ceiling as
well as memory speed; an idle device reset restored the original boost afterward.
The miner applies none of these settings. See all trials and restoration caveats
in the [AMD clock CSV](benchmarks/clocks-amd-2026-11-06.csv).

### AMD 1.0.6 benchmarks

The portable 1.0.6 OpenCL build was measured against the published 1.0.5 Linux
binary on 2026-11-06. A/B/B/A, four 60-second arms on each card, same power cap,
clocks, key and default launch geometry. Each rate is the mean of two arms.

| GPU | Power cap | Published 1.0.5 | Portable 1.0.6 build | Gain |
|---|---:|---:|---:|---:|
| Radeon RX 7900 XTX | 303 W | 12.281 MH/s | **13.628 MH/s** | **+11.0%** |
| Radeon RX 7600 XT | 165 W | 4.052 MH/s | **4.519 MH/s** | **+11.5%** |
| Instinct MI50 | 150 W | 5.753 MH/s | **6.880 MH/s** | **+19.6%** |

The measured RX 7900 XTX, RX 7600 XT and MI50 profiles use the new fe29_mlsp_c
kernel. Other model profiles retain their earlier defaults or first-start tuning.
Every potential share is recomputed through the complete VRF on the GPU before
submission. See the [portable release measurements](PLATFORMS.md#amd-106-portable-release-qualification)
and [CSV](benchmarks/amd-1.0.6.csv). The earlier development comparison is retained
separately in PLATFORMS.md.

The worker sleeps on OpenCL completion callbacks. MI50 and RX 7600 XT steady CPU
use is around 1% of one core. The tested XTX host's ROCm 7.14 event thread still
consumes approximately one core; that runtime issue remains. The MI50 comparison
uses 150 W because an earlier 225 W development trial reached the junction guard
and was excluded. These are Linux OpenCL results; the Windows package is NVIDIA
CUDA, and the new AMD profiles have not been qualified on Windows or other models.

### Historical measurements

The following measurements come from earlier releases. They are retained separately
from the development comparison above:

| GPU | Power cap | Rate | Measured with |
|---|---:|---:|---|
| RTX PRO 6000 Blackwell | 600 W | 43.0 MH/s | 0.5.6 (1.0.5 uses the Blackwell default; not re-measured) |
| H100 SXM (datacenter profile) | 700 W | 21.6 MH/s | 0.5.7 |
| A100-SXM4-80GB (datacenter profile) | 400 W | 12.7 MH/s | 0.5.7 |
| RTX A6000 | 300 W | 12 MH/s | 0.5.6 |
| RX 7900 XTX (OpenCL) | 244 W | 11.3 MH/s | 0.5.7 / AMD profiles 2026-10-24 |
| MI50 (OpenCL) | | 4.75–4.84 MH/s | AMD profiles 2026-10-24 |
| RX 7600 XT (OpenCL) | | 4.00 MH/s | AMD profiles 2026-10-24 |

The CMP 170HX and the RTX 4070 Ti SUPER each shared the card with another program that stayed idle during the runs
(not a material effect on the rates). At a reduced power cap an RTX 5090 gains a little from `nvidia-smi -lmc 7001,7001` (+1.4 % at 420 W,
nothing measurable at 600 W); never lock the memory below 7001 MHz. (VRF work; for hash jobs see the memory clock advisory above.)

Earlier releases (1.0.0 table, the same `--bench` method against 0.5.9; history in PLATFORMS.md rounds 10 and 11):

| GPU | Power cap | 0.5.7 | 0.5.8 / 0.5.9 | **1.0.0** | Gain over 0.5.9 | MH/s per W (1.0.0) |
|---|---:|---:|---:|---:|---:|---:|
| RTX 5090 (sm_120, 170 SMs) | 600 W | 42.3 MH/s | 63.9 MH/s | **68.0 MH/s** | +6.4 % | 0.118 |
| RTX 5090 | 420 W | 32.7 MH/s | 48.6 MH/s | **51.0 MH/s** | +5.0 % | 0.126 |
| RTX 5080 (sm_120) | 360 W | 22.1 MH/s | 33.1 MH/s | **35.5 MH/s** | +7.2 % | 0.106 |
| RTX 3070 (sm_86) | 240 W | 7.65 MH/s | 11.5 MH/s | **12.7 MH/s** | +11 % | 0.055 |
| CMP 170HX (sm_80, 74 SMs, core locked at 1300 MHz) | 250 W | 7.6 MH/s | 12.5 MH/s | **14.0 MH/s** | +12.7 % | 0.086 |
| RTX PRO 6000 Blackwell (0.5.6 kernels) | 600 W | 43.0 MH/s | not re-measured | | | |
| H100 SXM (0.5.7 datacenter profile) | 700 W | 21.6 MH/s | unchanged | unchanged | | 0.046 |
| A100-SXM4-80GB (0.5.7 datacenter profile) | 400 W | 12.7 MH/s | unchanged | unchanged | | 0.046 |
| RX 7900 XTX (OpenCL, 0.5.7) | 244 W | 11.3 MH/s | unchanged | unchanged | | 0.046 |

## Layout

- `proxichorizo.cu` — host: stratum client (plain and TLS), HTTP seeker client for solo, variants, autotune, tests.
- `platform_profiles.h` — per-GPU defaults; `tests/platform_profiles.cpp` checks model selection and preservation of existing profiles without requiring a GPU.
- `proxichorizo_device.cuh` — device: field arithmetic (5×51, 8×32 row form and 8×32 product-scanning form), Ed25519 points, SHA-512,
  the VRF attempt, the kernels. Included once per variant from `proxichorizo.cu`.
- `vectors.json`, `tools/vrfvec/` — test vectors generated with the Proxima node's own
  `util/vrf` package (build the tool inside a proxima checkout: `go run ./cmd/vrfvec`).
- `hiveos/`, `mmpos/`, `smos/`, `docker/`, `windows/` — launchers and container files shipped in the archives; `scripts/package.sh`, `scripts/package-windows.ps1` build them; `RELEASE-v<ver>.md` are the release notes.
- `scripts/check-secrets.sh`, `.githooks/pre-commit` — exposure check, see below.

## Contributing / committing

This repository must never carry keys, tokens, seeds, wallet files, internal host names
or private network addresses. `.gitignore` excludes the usual suspects; the pre-commit
hook runs `scripts/check-secrets.sh` on every staged file and refuses the commit on a
hit. Enable it once per clone:

```bash
git config core.hooksPath .githooks
```

and run `make check` before pushing. If the checker flags a false positive, fix the
text rather than the checker.

## License

ProxiChorizo License 1.0 (see `LICENSE`), source-available:

- **Mining is free.** Run it on any hardware and any pool, for yourself or as a paid
  operation, no permission needed.
- **Copy, study, modify and share it**, as long as the source goes with it and your
  changes stay under the same license with the notices intact.
- **Not allowed without written permission from suprnova.cc:** taking the code or
  any substantial part of it (field and point arithmetic, hash-to-curve, VRF attempt,
  kernel designs, autotuning, pool protocol) into another miner that is commercial
  (sold, dev fee, paid service) or closed-source, in source, PTX, SPIR-V or binary form.

Permission requests: open an issue on GitHub or contact suprnova.cc. Versions up to
and including v0.5.4 were released under an MIT license with a commercial-inclusion
clause; what those versions granted stays granted. Author: suprnova.cc.
