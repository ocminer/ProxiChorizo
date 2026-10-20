# ProxiChorizo — CUDA nonce seeker for Proxima

A GPU miner for [Proxima](https://lunfardo314.github.io/)'s fair-launch mine chain.
Proxima's proof of work is an ECVRF (Ed25519, RFC 9381) output under the key that
signs the transit, so there is no stratum: miners run a *nonce seeker* that fetches
jobs over HTTP from either `proxi node mine` or a pool speaking the same
[nonce-seeker protocol](https://github.com/lunfardo314/proxima/blob/develop/kb/external_nonce_seeker.md),
and post the nonces they find. ProxiChorizo is such a seeker, written for NVIDIA GPUs by suprnova.cc.

Built for and used by the Suprnova Proxima pool: https://proxi.suprnova.cc

**Miners: read [USAGE.md](USAGE.md)** — pool mining with suprnova as the example, solo
mining with `proxi node mine`, HiveOS, mmpOS, Docker, Windows, options, troubleshooting.
It is included in every release archive.

## Network upgrade (hard fork) at slot 29000 — 2026-10-21 17:01 UTC

Proxima's first live library upgrade ([release v1.1.0-take1](https://github.com/lunfardo314/proxima/releases/tag/v1.1.0-take1))
activates at slot 29000. From that slot a mine transit must be stamped at tick 83 of its
slot or later and carry miner version 5, and every transaction carries the version of its
slot's library. **ProxiChorizo needs no update**: a seeker only searches nonces for the job
(predecessor ‖ slot ‖ nonce); the tick, the miner version and the transaction version are
set by whoever builds the transit.

* **Pool miners** (suprnova and any pool on an updated node): nothing to do.
* **Solo miners**: the transit is built by your `proxi node mine`, so run `proxi` and your
  node from **v1.1.0-take1 or newer before slot 29000**. An older `proxi` keeps building
  version-4 transits after the upgrade and every one of them is rejected. A v1.1.0 `proxi`
  mines the old rules until slot 28999 and the new ones from 29000 by itself, so update early.

## Which kernel runs on which card

The platform table in `proxichorizo.cu` (NVIDIA) and `opencl/proxichorizo_cl.cpp` (AMD) holds the
measured default per GPU generation: Blackwell, Ampere GA10x, GA100 and RDNA3 / Vega were
benchmarked (see PLATFORMS.md). A generation without measurements (Ada, Turing, RDNA2/4, anything
unknown) is **tuned on the card itself at the first start**: the two candidate kernels are measured
for about 15 s and the winner is written to `proxichorizo-autotune.cache` next to the miner, keyed by
card and miner version, so later starts (and every new release) use or refresh it. `--autotune`
measures every kernel and shape, `--variant` / `--blocks` / `--iters` override everything; the
start-up line `GPU n ...: <variant>, <blocks>x<threads>x<iters> — <reason>` says what was chosen and why.

## Download
[GitHub Releases](https://github.com/ocminer/ProxiChorizo/releases) (Linux binary mirrored at https://proxi.suprnova.cc/downloads/):

* `proxichorizo-<ver>.tar.gz` — Linux x86-64, built on glibc 2.31. One tarball serves plain Linux, **HiveOS** (`h-run.sh`), **MMPOS** (`mmpos-run.sh`) and **SMOS** (`smos-run.sh`) — pick the launcher for your OS and set the job server, your pool token and your key seed in the flight sheet.
* `proxichorizo-hiveos.tar.gz` — the same package under the directory name HiveOS's custom-miner installer requires (miner name `proxichorizo`).
* `proxichorizo-<ver>-windows-x86_64.zip` — native **Windows** x64. Put the pool's `proxima.key` next to it, set your token in `start.cmd` and run it.
* Docker: `ocminersupr/proxichorizo:<ver>` (`docker/README.md`).
* `SHA256SUMS` and one `.sha256` per asset; every archive carries its own `SHA256SUMS.txt` and `BUILD-INFO.json`.

## Usage
```
./proxichorizo --proxi http://proxi.suprnova.cc:8100 --token YOUR_TOKEN --key-file proxima.key --name rig1     # pool
./proxichorizo --proxi http://your-node:8200 --token SEEKER_TOKEN --key-file proxima.key                        # solo, see USAGE.md
./proxichorizo --autotune --bench 10        # fastest kernel variant per GPU, then its hashrate
./proxichorizo --selftest vectors.json      # the math against the Proxima node's VRF vectors
./proxichorizo ... --api-bind 127.0.0.1:4078   # JSON statistics at /, mmpOS line at /mmpos
```
Measured: RTX 5080 ≈ 17 MH/s, RTX 3070 ≈ 6.9 MH/s, CMP 170HX ≈ 5–6 MH/s; one CPU core of the reference seeker ≈ 37 kH/s.
**[USAGE.md](USAGE.md)** has the full guide: registering at the pool, solo mining with `proxi node mine`, HiveOS / MMPOS / SMOS flight sheets, Docker, Windows, every option, troubleshooting.

## Build

**AMD / other OpenCL GPUs:** `cd opencl && make && ./proxichorizo-cl --selftest && ./proxichorizo-cl --proxi ... --token ... --key-file ...` — needs an OpenCL ICD and headers (ROCm or `ocl-icd-opencl-dev opencl-headers`); the kernels are compiled by the driver at start-up (about a minute) and checked against the embedded test vectors before mining. Measured: RX 7900 XTX ≈ 10.5 MH/s, MI50 ≈ 4.5, RX 7600 XT ≈ 3.6 (9×29-bit field layout, see PLATFORMS.md). Same options as the CUDA miner (`--groups` instead of `--blocks`). See the AMD section of PLATFORMS.md for rates.

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
Reference points (the same binary, platform defaults): RTX 5080 ≈ 17 MH/s, RTX 3070
≈ 6.5 MH/s, CMP 170HX ≈ 4.7 MH/s, one CPU core of the reference Rust seeker ≈ 37 kH/s.

## Layout

- `proxichorizo.cu` — host: seeker protocol client, variant/platform tables, autotune, tests.
- `proxichorizo_device.cuh` — device: field arithmetic (5×51 and 8×32), Ed25519 points, SHA-512,
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

MIT License with one restriction: inclusion in commercial projects requires prior
written permission from suprnova.cc. See `LICENSE`. Author: suprnova.cc.
