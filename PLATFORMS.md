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
