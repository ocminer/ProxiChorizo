# ProxiChorizo 2.0.1

**For Proxima library upgrade 6 (Tuesday 2026-11-10).** Same miner as 2.0.0 with a faster
txid-blake2b search on Ampere (RTX 30, sm_86). Update before slot 196000 (about 12:03 UTC); from
slot 197000 (about 14:54 UTC) Proxima mines the BLAKE2b-256 transaction-ID work. 2.0.0 keeps
working; 2.0.1 is an optional update for RTX 30 cards.

- **RTX 30 (sm_86):** the hash search loops over a 32-bit nonce word and precomputes the
  nonce-free steps of BLAKE2b rounds 0 and 1 once per job. RTX 3070 at its 240 W cap,
  `--hash-bench`: 600-byte prefix 3.30 → 3.33 GH/s; +0.6 % to +1.3 % depending on the prefix
  length (both nonce positions in one block and the block-straddling one).
- **Everything else is unchanged:** RTX 50, RTX 40, Turing, A100/H100/CMP and all OpenCL (AMD)
  devices run exactly the 2.0.0 hash kernels (identical machine code), and the VRF kernels of
  every architecture are identical to 2.0.0. On the RTX 5090 and the RX 7900 XTX the search is
  limited by the integer ALU; the alternatives we measured were not faster.
- **Tests:** the self-check vectors, a GPU-vs-CPU comparison of 2^28 nonces over all 32 message
  shapes (including partial launches and 2^32 nonce wraps, `tests/txid/txid_gpu_test.cu fuzz`),
  and testnet mining with blocks confirmed on a private Proxima v1.6.0 network.

## Hash rates (txid-blake2b, `--hash-bench`, 600-byte prefix)

| GPU | Backend | Power cap | Hash rate | Measured power | Core clock |
|---|---|---|---|---|---|
| RTX 5090 | CUDA | 575 W | **17.3 GH/s** | 575 W | 2730 MHz |
| RTX 5090 | CUDA | 450 W | **16.1 GH/s** | 450 W | 2542 MHz |
| RTX 5080 | CUDA | 360 W | **8.96 GH/s** | 309 W | 2857 MHz |
| RTX 5070 Ti | CUDA | 300 W | **7.28 GH/s** | 230 W | 2782 MHz |
| RTX 3070 | CUDA | 240 W | **3.33 GH/s** (2.0.0: 3.30) | 231 W | 1905 MHz |
| RX 7900 XTX | OpenCL | 303 W | **6.33 GH/s** | 280 W | 2465 MHz |
| Instinct MI50 | OpenCL | 225 W | **2.81 GH/s** | 220 W | 1690 MHz |
| RX 7600 XT | OpenCL | 165 W | **2.16 GH/s** | 163 W | 2531 MHz |

VRF rates are those of 1.0.8. ProxiChorizo License 1.0 is unchanged.
