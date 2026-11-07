# ProxiChorizo 1.0.7

Per-model kernel selection, improved autotuning and two AMD arithmetic optimizations.

- RX 7600 XT uses a three-block Karatsuba field multiply; RX 7900 XTX uses a shorter
  equivalent scalar with three final cofactor doublings. MI50 retains its faster
  existing kernel. Final measurements and power/clock settings are in the README.
- Measured NVIDIA models keep their qualified kernels. Unmeasured Ampere, Ada and
  Blackwell models compare kernels on the actual card, including RTX 3060 / 3060 Ti.
  The tuning search now includes the 256-blocks-per-SM ladder launch.
- AMD tuning includes ladder and split kernels with their appropriate launch shapes.
  Results are cached separately per physical GPU, processor count, driver API/runtime
  and release, preventing another board's result from being reused.

Matched full-build measurements against 1.0.6: RX 7600 XT **+2.46%**
(4.461 → 4.571 MH/s at 155 W), RX 7900 XTX **+0.71%**
(13.529 → 13.625 MH/s at 303 W). MI50 retains its existing kernel;
the final-build check measured 6.901 MH/s at 150 W. NVIDIA GPU arithmetic is
unchanged from 1.0.6. README tables include final rates and actual power/clocks.

The reported RTX 3070 slowdown against 1.0.1 was not reproduced on either tested
card, including tests at 810 MHz memory. The README includes matched release
comparisons, mining-mode checks and core/memory clock sweeps with actual watts.
3060 and 3060 Ti hardware was unavailable; no physical speed claim is made for them.

Linux, HiveOS, mmpOS and SMOS include NVIDIA CUDA and AMD OpenCL. Windows and Docker
remain NVIDIA. ProxiChorizo License 1.0 and third-party notices are unchanged.

Windows builds include executable startup and CPU profile checks; Windows GPU
hardware was not tested. The installed RX 7900 XTX ROCm runtime still uses about
one CPU core.
