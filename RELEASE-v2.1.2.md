# ProxiChorizo 2.1.2

This release speeds up BLAKE2b mining on NVIDIA and AMD GPUs. The gain depends on the job layout.

Replaying three captured 504-byte padded pool jobs on Linux increased throughput by **1.03% on RTX 5090, 2.68% on RTX 5080 and 3.03% on RTX 5070 Ti** versus v2.1.1. On the RTX 5070 Ti, **948 verified shares** met the deadlines in the two candidate runs, compared with **920** for v2.1.1. On the RTX 5090, the first verified share on a captured padded job arrived in **1.21 seconds**, down from **2.19 seconds**.

On a matched 504-byte padded benchmark, the RX 7900 XTX gained **1.37%** and the RX 7600 XT **1.49%**. The RX 7600 XT figures come from cooled five-second runs. The RTX 3070 and GTX 1080 Ti padded results were within run variation.

Ordinary jobs whose eight-byte nonce crosses a BLAKE2b block boundary can reuse the first compression. At a 381-byte prefix, throughput reached **2.02–2.17×** the v2.1.1 rate on the seven tested GPU models. In a local RTX 5080 replay of seven split layouts, **1,247 verified shares** arrived within three seconds, compared with **630** for v2.1.1. The three captured production jobs all used the padded layout.

The miner selects the new kernels automatically. The primary Linux hardware replays verified **13,405 shares** without a hash mismatch or duplicate. Hardware measurements used the development build at `70f3ef6`. The source benchmark CSV gives the individual rates and executable hashes. Windows hardware throughput was not measured.

Packages cover Linux CUDA/OpenCL, Windows NVIDIA, Windows AMD, HiveOS, mmpOS and SMOS. NVIDIA and AMD Docker images use the tags `2.1.2` and `2.1.2-amd`. Pool login uses your PROX payout address and a worker name.

[Quick start](QUICKSTART-v2.1.2.md) · [Benchmark data](benchmarks/proxima-blake2b-2.1.2.csv)
