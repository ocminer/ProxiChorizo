# ProxiChorizo — platform notes and measured optimisations

The work per attempt: 1 SHA-512 + try-and-increment hash-to-curve (≈2 tries,
each a decompression = one field exponentiation), one 254-bit scalar
multiplication on Ed25519 (extended coordinates, a = −1), ×8 cofactor, one
inversion for compression, 1 SHA-512. ≈3,600 field multiplications per attempt.

## Build-time variants (macros in proxichorizo.cu)

| Macro | Values | Meaning |
|---|---|---|
| `PX_FE32` | unset / defined | field element as 5×51-bit limbs with 64-bit multiplies (unset) or 8×32-bit limbs with PTX `mad.lo.cc/madc.hi.cc` carry chains and a dedicated squaring (defined) |
| `PX_WINDOW` | 0, 2, 4 | scalar multiplication: plain double-and-add (0), 2-bit window with a 3-point table, 4-bit window with a 16-point table in local memory |
| `PX_LB_THREADS`, `PX_LB_BLOCKS` | 256, 1/2 | `__launch_bounds__` for mine_kernel; 2 blocks/SM caps registers at 128 and raises occupancy |
| `PX_SQ_MUL` | defined | debugging: squaring falls back to multiplication |
| `PX_SHA_OLD` | defined | debugging: SHA-512 with the 80-word schedule instead of the rolling 16-word one |

## Measurements, 2026-10-19 (German test rig, driver 615.71, CUDA 13.4; the 3070s sit on x1/x4 gen1 risers)

**Correction:** the numbers in the first table were taken with `--bench` before the
fix of 10:45 UTC, when a bench without a key used an all-zero scalar, so the scalar
multiplication skipped every point addition and rates read ~1.45× too high. Relative
comparisons between field implementations still hold; comparisons that change the
number of additions (window size) do not. The second table is with a realistic scalar.

### Honest numbers (realistic scalar), RTX 5080

| Build | MH/s |
|---|---|
| fe51, window 4, lb1 (first version) | 6.69 |
| fe51, window 0, lb2 | 6.37 |
| fe32 + dedicated squaring, window 0, lb2 | 8.67 |
| fe32 + sq, window 2, lb2 | 6.78 |
| **fe32 + sq, window 4, lb2** | **9.39** |
| fe32 + sq, window 4, lb1 | 7.85 |
| fe32 + sq, window 2, lb1 | 6.17 |

### Honest numbers (realistic scalar), RTX 3070 (sm_86, the other 3070 mining)

| Build | MH/s |
|---|---|
| **fe32 + sq, window 0, lb2** | **2.85** |
| fe32 + sq, window 4, lb2 | 2.77 |
| fe32 + sq, window 2, lb2 | 1.97 |
| fe32 + sq, window 2, lb1 | 1.95 |

So the best window differs by generation: Blackwell (sm_120) wants the 4-bit table
(+8 % over plain), Ampere (sm_86) plain double-and-add (+3 %). The 2-bit window's
three-way branch is bad everywhere. This is the per-platform table the single
binary must carry.

### First-round table (zero-scalar bench, inflated ~1.45×)

| Build | RTX 5080 (sm_120) | RTX 3070 (sm_86, other 3070 mining) |
|---|---|---|
| fe51, window 4, lb1 (first version) | 7.95 MH/s | 2.21 (idle rig: 2.4–2.6) |
| fe51, window 0 | 8.55 | |
| fe51, window 2 | 8.37 | |
| fe51, window 0, lb2 | 9.26 | 1.70 |
| fe51, window 2, lb2 | 9.33 | |
| fe32 (mul-based squaring), window 0, lb2 | 8.97 | 2.21 |
| **fe32 + dedicated squaring, window 0, lb2** | **12.50** | **4.25** |
| same, threads 256 iters 4 | 12.59 | |
| same, threads 128 iters 4 | 12.12 | |
| same, threads 192 iters 2 | 11.64 | |

Register/stack (ptxas, sm_120): first version 255 regs, 3168 B stack, spills;
fe32+sq+lb2: 128 regs, 368 B stack, no spills.

Lessons:
- 64-bit integer multiplies are emulated on NVIDIA; the 8×32 carry-chain schoolbook is ~1.5–2× faster per field multiply. Blackwell only benefits once the squaring is dedicated (the mul-based fe32 was 3 % slower there).
- A multi-instruction inline-asm statement with many `"+r"` operands that hold **equal values** (e.g. zero limbs) can get those operands coalesced into one register by the compiler, so writes collide. Keep statements small (one limb pair per statement) and carry explicitly; always use `"=&r"` for pure outputs. The random field test (`--fetest`) did not catch it because the identity checks passed on random inputs; the sqrt(−1) derivation in `init_kernel` did.
- The 16-point window table lives in local memory (2.5 KB/thread) and costs more in occupancy than it saves in additions; plain double-and-add wins on both generations.
- `__launch_bounds__(256, 2)` (≤128 registers) is a free 7–10 %.
- Divergence in the try-and-increment loop (geometric number of tries, warp waits for the slowest lane) is the next structural cost (~25 %): compaction into a separate scalar-multiplication pass is the planned fix.

## Round 2: compacted hash-to-curve (2026-10-19 11:30 UTC)

The try-and-increment loop made warps wait for their slowest lane (expected
≈6 decompressions per warp for ≈2 needed per lane). `PX_COMPACT` splits an
iteration into `h2c_kernel` passes (one SHA-512 + one decompression per item,
failures queued for the next pass, 12 passes max) and `mult_kernel` (×8, scalar
multiplication, ×8, compression, SHA-512) over the found points with no
divergence; both kernels also carry far fewer registers than the fused one.
`--verify` proves every variant finds identical hits over one nonce range.

| GPU | fe32_w4_lb2 | fe32_w0_lb2_c | **fe32_w4_lb2_c** (256×4) |
|---|---|---|---|
| RTX 5080 | 9.41 | 14.58 | **16.60 MH/s** |
| RTX 3070 (other 3070 mining) | 2.79 | 5.16 | **5.98 MH/s** |

Versus the first version (6.7 / ~2.1): **2.5× on Blackwell, ~2.8× on Ampere.**
With the divergence gone the 4-bit window wins on both generations, so the
platform table now defaults every architecture to `fe32_w4_lb2_c` 256×4.

## Round 3: wNAF scalar multiplication (2026-10-19 12:00 UTC)

`PX_WINDOW=5` = width-5 wNAF of the constant scalar (digits computed on the host in
`v_set_scalar`, kept in constant memory), 8 odd multiples cached as (Y+X, Y−X, 2dT, 2Z)
for mixed additions, negation by swapping the sums. ~43 additions instead of 63 + 14
table operations. Variant `fe32_wnaf_lb2_c`: RTX 5080 16.6 → **17.3 MH/s**, RTX 3070
6.0 → **6.2 MH/s** (+4 %). Now the platform default everywhere.

## CMP 170HX (GA100, sm_80) — 2026-10-19 12:50 UTC

Autotune on a CMP 170HX (PCIe gen2 x1 riser): `fe32_w4_lb2_c` 256×2 **4.7 MH/s** (another
unit of the same card: 6.6 MH/s — the cards differ in clocks/power state), `fe32_wnaf_lb2_c`
4.1, `fe32_w0_lb2_c` 3.6, fused kernels 1.3–1.7. So GA100 prefers the 4-bit table over
wNAF, the opposite of GA10x/Blackwell; the platform table has its own row for 8.0.
HBM2 bandwidth plays no role — the work is pure integer ALU; the x1 link does not
matter either (a job is 200 bytes).

## Round 4: fewer multiplications per doubling (2026-10-19 14:30 UTC)

A doubling reads X, Y, Z only; T is needed solely as the input of the next
addition. In the wNAF loop the digit is known (constant memory, uniform), so a
doubling followed by another doubling skips T (3M + 4S instead of 4M + 4S); the
same in the 4-bit and plain variants and in the cofactor ×8. RTX 3070:
`fe32_wnaf_lb2_c` 6.55 → **6.87 MH/s** (+4.9 %), hits identical under `--verify`.

Tried and dropped the same day (RTX 3070, `fe32_wnaf_lb2_c` 256×4):
- two points per thread sharing one inversion (Montgomery's trick): 6.83 vs 6.86 —
  the first point parked in local memory while the second is multiplied costs what
  the saved ~265-operation inversion brings;
- the multiply kernel at 1 block/SM (`PX_LB_MULT_BLOCKS 1`, 148 registers, no
  spills): 5.54 at 256 threads, 6.38 at 128 — occupancy beats registers here.
  The macro stays for other platforms.

ptxas, sm_120, `fe32_wnaf_lb2_c`: h2c_kernel 118 registers, 128 B stack; mult_kernel
128 registers (the cap of 2 blocks/SM), 1 KB stack = the 8 cached odd multiples, 12 B
spills. The table in local memory is the remaining structural cost of wNAF.

## One binary, all variants (since 2026-10-19 11:00 UTC)

`proxichorizo.cu` is the host; `proxichorizo_device.cuh` is the whole device side and is included
once per variant inside a namespace with the macros set. Variants compiled in:

| name | field | window | launch bounds |
|---|---|---|---|
| `fe32_w0_lb2` | 8×32 carry chains + dedicated squaring | plain double-and-add | 256 threads, 2 blocks/SM |
| `fe32_w4_lb2` | same | 4-bit table | same |
| `fe51_w4_lb1` | 5×51 64-bit (the first version) | 4-bit table | 256, 1 |
| `fe32_w0_lb2_c` | 8×32 + sq, compacted pipeline | plain | 256, 2 |
| `fe32_w4_lb2_c` | 8×32 + sq, compacted pipeline | 4-bit table | 256, 2 |
| `fe32_wnaf_lb2_c` | 8×32 + sq, compacted pipeline | width-5 wNAF, cached mixed adds | 256, 2 |

Selection: `PLATFORMS[]` in proxichorizo.cu maps compute capability → variant and
geometry (12.x → fe32_w4_lb2; 8.x/7.5 → fe32_w0_lb2); `--variant NAME` forces one
for all GPUs; `--autotune [secs]` measures every variant × {256×2, 128×4, 256×4} on
each GPU and takes the best; `--list-variants`, `--selftest` and `--fetest` run all.
Distribution build: `-gencode` for sm_75, sm_80, sm_86, sm_89, sm_120 plus
compute_120 PTX for newer cards. Adding a platform = a row in `PLATFORMS[]`;
adding a kernel idea = a macro in proxichorizo_device.cuh and a namespace block.

## AMD / OpenCL port (2026-10-19, `opencl/`)

Same algorithm in OpenCL C 1.2 (`opencl/proxichorizo.cl`), compiled per GPU and variant by the
installed driver at start-up (45–50 s per variant on ROCm; cached in the process). The host
(`opencl/proxichorizo_cl.cpp`) shares the HTTP/job/submit/stats code with the CUDA miner
through `host_common.inc`, generated from `proxichorizo.cu` by `gen_host.py`. Variants:
`fe32_*` = 8×32-bit limbs with 64-bit accumulators (comba), `fe51_*` = 5×51 with `mul_hi`;
`_c` = compacted pipeline; `wnaf`/`w4`/`w0` as on CUDA. Every selected variant must reproduce
the 16 embedded test vectors on the device before it is used (`prog_validated`): ROCm 6.4's
compiler miscompiles the fe51 layout (last byte of the compressed point wrong), ROCm 7.14 does
not; if the platform default fails validation the miner falls back to `fe32_wnaf_c`.

### First round (fe32 / fe51)

| GPU | driver | best variant | MH/s |
|---|---|---|---:|
| RX 7900 XTX (gfx1100, 48 WGP) | ROCm 7.14 OpenCL | fe51_wnaf_c ×8 5.49 / fe32_wnaf_c ×8 5.36 | ≈ 5.4 |
| RX 7600 XT (gfx1102) | ROCm 6.4 | fe32_wnaf_c ×8 | 1.9 |
| MI50 (gfx906, 60 CU) | ROCm 6.4 | fe32_wnaf_c ×8 | 1.9 |

ISA (gfx1100, fe32_wnaf_c): mult_kernel 248 VGPRs, 1.4 KB scratch/lane (the cached wNAF table
and byte arrays), occupancy 5/16, the field multiply compiles to `v_mad_u64_u32` (4127 of them).
RDNA executes 32×32→64 multiply-adds at a quarter of the ALU rate, so the realistic ceiling of a
7900 XTX is well below a comparable NVIDIA part; `-O3` changes nothing. Word-based SHA-512
message assembly (no byte arrays in scratch) brought +3 %.

### Second round: 9×29-bit limbs (`fe29_*`, now the AMD default)

The 8×32 comba needs a carry chain after every 64-bit accumulate because the column sums
exceed 64 bits. With 9 limbs of 29 bits every one of the 81 partial products is < 2^58, so a
whole column (≤ 9 products, or 2^30-limb operands after one lazy addition: 9 × 2^60 < 2^64)
fits a 64-bit accumulator and the schoolbook is 81 bare `v_mad_u64_u32` with no carry handling
at all; the 17 columns are folded once at the end (2^261 ≡ 1216 mod p). Subtraction adds 1024p
limb-wise before subtracting so operands with 30/31-bit limbs stay non-negative (8p was not
enough: its top limb is 2^26 and a lazily added top limb wrapped, which made `fe_eq` lie — found
with the CPU reference build in `opencl/hostref/`, which compiles the very same `.cl` as C and
asserts the limb invariants).

| GPU | driver | fe32_wnaf_c | **fe29_wnaf_c** | fe29_w4_c | fe29_w0_c | fe29_wnaf (fused) |
|---|---|---:|---:|---:|---:|---:|
| RX 7900 XTX (gfx1100) | ROCm 7.14 | 5.43 | **10.81** | 9.64 | 8.68 | 7.99 |
| RX 7600 XT (gfx1102) | ROCm 6.4 | 1.91 | **3.58** | | | |
| MI50 (gfx906, 60 CU) | ROCm 6.4 | 1.95 | **4.52** | | | |

2× across three generations (RDNA3, CDNA-era Vega 20), `--iters 8` best everywhere; live on
the pool the XTX reports 10.3–10.5 MH/s with shares accepted. ISA (gfx1100, fe29_wnaf_c):
mult_kernel 177 VGPRs (was 248), 1.16 KB scratch, occupancy 8/16, 8524 `v_mad_u64_u32`
(2× the fe32 count — all 81 products are mads now, but each is a single instruction instead of
mad + carry chain). The 1 T mad64/s measured ceiling argument below now puts the XTX at
≈ 2.5 T mad64-equivalents/s of the 2.31 T peak: the multiply is at the ALU limit; what is left is
the non-multiply work (carries, SHA-512, the scratch-resident wNAF table).

Measured instruction throughput on the RX 7900 XTX (`tools/clmulbench.cpp`, 8 independent
accumulators, the loop also carries 8 cheap updates per 8 multiplies):

| op | Gop/s |
|---|---:|
| 32-bit add/xor | 3681 |
| `mad24` (24×24+32) | 4489 |
| `mul_lo` 32×32→32 | 2834 |
| `mul_hi` 32×32→32 | 2453 |
| `(ulong)a*b + acc` (`v_mad_u64_u32`) | 2310 |

An attempt costs ≈ 3,600 field multiplications × 64 `v_mad_u64_u32` ≈ 230k of them plus
adds/carries/SHA, so 5.4 MH/s is ≈ 1.25 T mad64/s = 55 % of the measured peak before the other
work is counted — the ceiling is ≈ 10 MH/s theoretical, 7–7.5 MH/s realistic. Radix 2^24 limbs
(11 limbs, 121 lo + 121 hi products at the faster `mad24` rate) cost about twice as many
cycle-equivalents as the 8×32 schoolbook and were not pursued. For comparison an RTX 5080 does
~8 T IMAD32/s in the CUDA kernel: the hardware ratio, not the port, explains most of the gap.
