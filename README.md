# ProxiChorizo — GPU miner for Proxima (PROX)

A GPU miner for [Proxima](https://lunfardo314.github.io/)'s fair-launch mine chain, for NVIDIA
(CUDA) and AMD (OpenCL) GPUs, by suprnova.cc. Proxima's proof of work is an ECVRF (Ed25519,
RFC 9381) output under the key that signs the transit; the miner searches nonces for the
pool's jobs and reports the ones whose output has enough trailing zero bits.

**From 1.0.0 the miner talks stratum** to the pool — one persistent connection, plain
(`stratum+tcp://`) or encrypted (`stratum+ssl://`) — with its own nonce range and share
difficulty per connection, as on any NOMP-style pool. Solo mining against your own node
(`proxi node mine`, HTTP) works as before.

Built for and used by the Suprnova Proxima pool: https://proxi.suprnova.cc

**2.0 — the new proof of work of library upgrade 6.** On 2026-11-10 Proxima replaces the VRF
work: from slot 196000 (about 12:03 UTC) VRF jobs end and the ledger pauses mining, and from
slot 197000 (about 14:54 UTC) the work is the mining transaction's own ID, a BLAKE2b-256 hash
(RFC 7693) with an 8-byte nonce at its end ("txid-blake2b"). ProxiChorizo 2.0 mines **both**:
VRF jobs until the switch, hash jobs after it, chosen per job by the pool, without a restart.
Update to 2.0 before the switch; from slot 196000 the pool accepts only ProxiChorizo 2.0 or
later. Hash work needs **no key file**: after the switch the miner can run with only the
pool URL, worker and token. Hash rates are in GH/s (RTX 5090 ≈ 16.4 GH/s at 450 W); see
[2.0 hash benchmarks](#20-hash-work-txid-blake2b).

## How to mine (quick start)

**You need:** an NVIDIA GPU (Turing or newer, driver 550+) or an AMD GPU with OpenCL,
Linux x86-64 (also HiveOS, MMPOS, SMOS) or Windows 10/11 x64, and a PROX address
(the 64-hex holder ID your wallet shows as `a/<hex>`).

1. **Register** at https://proxi.suprnova.cc/StartMining.html with your PROX address
   (without the `a/`). You get a **token** and a **key file**. Save the key file as
   `proxima.key` next to the miner and keep the token. The key is a mining key the pool
   made for your account: it can only mine for you and holds no funds. Rewards are paid
   to your PROX address.
2. **Download** the release for your system from
   [GitHub Releases](https://github.com/ocminer/ProxiChorizo/releases) and unpack it.
3. **Start the miner** with the pool URL, your token, the key file and a worker name:

   Linux
   ```bash
   ./proxichorizo -o stratum+tcp://proxi.suprnova.cc:3355 -u rig1 -p YOUR_TOKEN --key-file proxima.key
   ```
   Windows (Command Prompt, in the unpacked folder; or edit and run `start.cmd`)
   ```
   proxichorizo.exe -o stratum+tcp://proxi.suprnova.cc:3355 -u rig1 -p YOUR_TOKEN --key-file proxima.key
   ```
   AMD GPUs on Linux: the same with `./proxichorizo-cl`. The native Windows archive contains the NVIDIA CUDA miner.

### Pool addresses

| | URL |
|---|---|
| Stratum | `stratum+tcp://proxi.suprnova.cc:3355` |
| Stratum over TLS (encrypted) | `stratum+ssl://proxi.suprnova.cc:3356` |

TLS needs OpenSSL 1.1 or 3 on Linux (present on HiveOS, MMPOS, SMOS and every current
distribution); the Windows archive brings its own. The pool's certificate is checked.

### Login

* `-u` / `--user`: the **worker name** (`rig1`), or `YOURADDRESS.rig1`. The address part is
  optional — the token already identifies your account; if you give it, it must match.
  Without `-u` the miner uses `--name`.
* `-p` / `--token`: your **token**. `--token-file FILE` (first line) or the environment
  variable `PROXI_TOKEN` keep it off the command line.
* `--key-file proxima.key`: the key file from the registration (or `--seed-hex HEX`). It is needed for
  VRF jobs only (until slot 196000); hash jobs (from slot 197000) need no key.
* Two rigs may use the same worker name; every connection gets its own nonce range anyway.
  Distinct names show up as separate workers in your statistics.

### Mining OSes and Docker

* **HiveOS:** custom miner `proxichorizo`, installation URL = the `proxichorizo-hiveos.tar.gz`
  of the release; pool URL `stratum+tcp://proxi.suprnova.cc:3355`, wallet/template = your
  token, password = the seed of your key file. Worker name = the rig name.
* **MMPOS / SMOS:** the same tarball, launchers `mmpos-run.sh` / `smos-run.sh`; pool URL as
  above, user = token, password = key seed (see [USAGE.md](USAGE.md)).
* **Docker (NVIDIA):** `ocminersupr/proxichorizo:2.0.0` with `TOKEN`, `KEY_FILE` (or `SEED`)
  and optionally `POOL_URL` (default `stratum+tcp://proxi.suprnova.cc:3355`); see `docker/README.md`.
* **Docker (AMD):** `ocminersupr/proxichorizo:2.0.0-amd` (also `:amd`), the OpenCL miner with AMD's
  ROCm OpenCL runtime inside; the host needs only the amdgpu driver (`--device /dev/kfd --device /dev/dri`).

### Is it working?

Test the connection first, without mining (no GPU needed):
`./proxichorizo --check-pool stratum+ssl://proxi.suprnova.cc:3356` → `pool OK, …`. Then start the miner; it prints:

```
stratum: connected to proxi.suprnova.cc:3355
stratum: authorized as rig1
GPU #0: share accepted (1/1)
Total: 49.75 MH/s on 1/1 GPUs, 418 W (0.119 MH/W) | accepted 42, rejected 0, stale 0 | job 1a19… k=30 | uptime 00:10:00
```

The first shares come within seconds at a low difficulty; within a minute or two the pool
raises it so that each rig sends a share about every 20 seconds (`k` in the total line is the
current difficulty, 2^k attempts per share). A few `stale` shares at job changes are normal.
Your workers and earnings: https://proxi.suprnova.cc (enter your PROX address).

### Common problems

| Message | Cause |
|---|---|
| `authorization of … refused … check --token and --user` | wrong token, or an address in `-u` that is not the token's |
| `the pool sends jobs for key … but this miner holds …` | the key file is not the one from your registration |
| `stratum: cannot connect` / reconnecting | network or firewall; the miner retries by itself |
| `TLS handshake … failed` | old OpenSSL, or a proxy breaking TLS; use `stratum+tcp://` |
| `share REJECTED (23 low difficulty share)` | should not happen; report it with the log |

Everything else — options, thermal limits, the stats API, solo mining, troubleshooting — is
in **[USAGE.md](USAGE.md)**, included in every release archive.

Writing your own miner or proxy for the pool? The stratum protocol (messages, nonce layout,
share rule, error codes, a worked example) is specified in **[STRATUM-SPEC.md](STRATUM-SPEC.md)**.

**Older versions:** ProxiChorizo 0.5.x speaks the pool's old HTTP protocol (port 8100),
which stays open for a while and will then be switched off. Update to 1.0.7.

## Which kernel runs on which card

The platform table in `platform_profiles.h` (NVIDIA) and `opencl/proxichorizo_cl.cpp`
(AMD) selects the measured kernel for each tested model. Models without measurements,
including RTX 3060 / 3060 Ti, tune on the actual card at first mining start. This takes
about a minute of timing plus compilation; OpenCL compilation can take several minutes.
The winner is saved in `proxichorizo-autotune.cache`, separately for each physical GPU,
processor count, driver API/runtime and miner release.

`--autotune` measures every kernel and launch shape; `--variant`, `--blocks` / `--groups`
and `--iters` override the defaults. The startup line reports the selected kernel and
launch shape. After changing power or clocks, run `--autotune` again to compare kernels
under the new settings. The miner does not change power limits or clocks.

A100 and H100 retain their separate measured profiles. The CMP 170HX, tested RTX 3070,
4070 Ti SUPER, 5070 Ti, 5080, 5090 and RTX PRO 6000 models retain the NVIDIA ladder
kernel. Other models and MIG partitions use the applicable fallback and tuning rules;
see `PLATFORMS.md` for the configurations actually measured. The distribution includes
native SM90 code for H100.

## Download
[GitHub Releases](https://github.com/ocminer/ProxiChorizo/releases):

* `proxichorizo-<ver>.tar.gz` — Linux x86-64, built on glibc 2.31. One tarball serves plain Linux, **HiveOS** (`h-run.sh`), **MMPOS** (`mmpos-run.sh`) and **SMOS** (`smos-run.sh`) — pick the launcher for your OS and set the pool URL, your pool token and your key seed in the flight sheet.
* `proxichorizo-hiveos.tar.gz` — the same package under the directory name HiveOS's custom-miner installer requires (miner name `proxichorizo`).
* `proxichorizo-<ver>-windows-x86_64.zip` — native **Windows** x64, with the OpenSSL DLLs and `cacert.pem` for `stratum+ssl://`. Put the pool's `proxima.key` next to it, set your token in `start.cmd` and run it.
* Docker: `ocminersupr/proxichorizo:<ver>` for NVIDIA, `ocminersupr/proxichorizo:<ver>-amd` for AMD (`docker/README.md`).
* `SHA256SUMS` and one `.sha256` per asset; every archive carries its own `SHA256SUMS.txt` and `BUILD-INFO.json`.

## Usage
```
./proxichorizo --proxi stratum+tcp://proxi.suprnova.cc:3355 --token YOUR_TOKEN --key-file proxima.key --name rig1   # pool, stratum (or --token-file FILE); every GPU, one process
./proxichorizo ... -d 0,2                   # only these GPUs (PCI bus order, as nvidia-smi numbers them)
./proxichorizo --proxi http://your-node:8200 --token SEEKER_TOKEN --key-file proxima.key                        # solo, see USAGE.md
./proxichorizo --autotune --bench 10        # fastest kernel variant per GPU, then its hashrate
./proxichorizo --selftest vectors.json      # the math against the Proxima node's VRF vectors
./proxichorizo ... --api-bind 127.0.0.1:4078   # JSON statistics at /, mmpOS line at /mmpos
./proxichorizo ... --max-temp 85 --log-interval 10 --quiet   # thermal limit per GPU (default 88 C, pauses only that card), console cadence
```
Without a GPU argument the miner takes every GPU in the machine in one process; a GPU that overheats pauses alone and resumes when cool, a GPU with a device error is disabled alone. The console shows a start-up banner, per-GPU lines with temperature, power and clock, share results per GPU and a total line (colours on a terminal, plain text in logs).
Measured (1.0.5, `--bench`, platform defaults): RTX 5090 ≈ 78.6 MH/s at 600 W / ≈ 67 at 420 W, RTX 5080 ≈ 39.4–40.2 MH/s, RTX 3070 ≈ 14.8–15.1 MH/s, RTX 4070 Ti SUPER ≈ 31.3 MH/s, CMP 170HX ≈ 17.0 MH/s; unchanged from earlier releases: RTX PRO 6000 ≈ 43 MH/s (0.5.6), H100 SXM ≈ 21.6 MH/s (0.5.7), A100 ≈ 12.7 MH/s (0.5.7), RTX A6000 ≈ 12 MH/s (0.5.6); one CPU core of the reference seeker ≈ 37 kH/s. See the [benchmarks](#benchmarks) below.
**[USAGE.md](USAGE.md)** has the full guide: registering at the pool, solo mining with `proxi node mine`, HiveOS / MMPOS / SMOS flight sheets, Docker, Windows, every option, troubleshooting.

## Build

**AMD / other OpenCL GPUs:** `cd opencl && make && ./proxichorizo-cl --selftest && ./proxichorizo-cl --proxi ... --token ... --key-file ...` — needs an OpenCL ICD and headers (ROCm or `ocl-icd-opencl-dev opencl-headers`); the kernels are compiled by the driver at start-up (about a minute) and checked against the embedded test vectors before mining. Version 1.0.6 adds faster measured profiles for RX 7900 XTX, RX 7600 XT and MI50; see the [AMD benchmarks](#amd-106-benchmarks) below. Remove old `--groups`/`--iters` overrides to use the automatic profiles. Same options as the CUDA miner (`--groups` instead of `--blocks`). See the AMD section of PLATFORMS.md for rates.

```bash
make                 # all supported generations (Turing .. Blackwell), slow
make ARCHS="-gencode arch=compute_86,code=sm_86"   # just your card, fast
make test            # self-test against the Go reference vectors, random field test, cross-variant verify
```

Requires CUDA 12.8+ (13 recommended) and a driver that supports it. The binary links the
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
`--hash-bench` (600-byte prefix, the size of a Proxima mining transaction):

HASHBENCH_TABLE


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
memory lock is enabled by this release. [All NVIDIA sweep rows and exact readings](benchmarks/clocks-nvidia-2026-11-06.csv).

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
nothing measurable at 600 W); never lock the memory below 7001 MHz.

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
