# ProxiChorizo — CUDA nonce seeker for Proxima

A GPU miner for [Proxima](https://lunfardo314.github.io/)'s fair-launch mine chain.
Proxima's proof of work is an ECVRF (Ed25519, RFC 9381) output under the key that
signs the transit, so there is no stratum: miners run a *nonce seeker* that fetches
jobs over HTTP from either `proxi node mine` or a pool speaking the same
[nonce-seeker protocol](https://github.com/lunfardo314/proxima/blob/develop/kb/external_nonce_seeker.md),
and post the nonces they find. ProxiChorizo is such a seeker, written for NVIDIA GPUs by suprnova.cc.

Built for and used by the Suprnova Proxima pool: https://proxi.suprnova.cc

## Build

```bash
make                 # all supported generations (Turing .. Blackwell), slow
make ARCHS="-gencode arch=compute_86,code=sm_86"   # just your card, fast
make test            # self-test against the Go reference vectors, random field test, cross-variant verify
```

Requires CUDA 12+ (13 recommended) and a driver that supports it. The binary links
the CUDA runtime statically; it needs only libstdc++/libc on the target.

## Run

```bash
./proxichorizo --proxi http://proxi.suprnova.cc:8100 --token YOUR_TOKEN --key-file proxima.key --name rig1
```

- `--proxi` is the job server, `--token` its bearer token, `--key-file` the Ed25519 key
  the jobs are to be solved under (an unencrypted `proxima.key`; or `--seed-hex`,
  or `NONCE_SEEKER_SEED`). With a pool, use the key file the pool gave you at
  registration, never your own wallet key.
- `--autotune [secs]` measures every kernel variant on each GPU and uses the fastest.
  The built-in platform table already picks good defaults per GPU generation.
- `--gpus 0,1`, `--variant NAME` (`--list-variants`), `--threads N --iters N --blocks N`.
- Several GPUs: one process per GPU with `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<n>`
  gives the steadiest rates.

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
