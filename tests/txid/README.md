# Blake2b and solo regression checks

The existing `txid_cpu_test.cpp`, generated vectors and `txid_gpu_test.cu` cover
the full CPU hash and ordinary GPU kernels. The miner's `--hash-selftest` also
checks the RTX 5090 search path when that path is available on the selected GPU.

The solo regression uses the real shared CUDA/OpenCL host code and a loopback
HTTP server; it requires no GPU or external node. From the repository root:

```sh
(cd opencl && python3 gen_host.py)
g++ -O2 -std=c++17 -pthread tests/txid/solo_host_test.cpp -ldl -o /tmp/proxichorizo-solo-test
/tmp/proxichorizo-solo-test
```

It checks a valid improvement after the pool's 256-candidate limit, complete
work-value ordering, CPU verification, unchanged pool limiting, job validation,
zero TTL, delayed beat updates and chunked HTTP responses.

For the RTX 5090 path, the additional GPU test compares the entire CPU hit set
with the GPU hit set, including partial launches and counter boundaries:

```sh
nvcc -O3 -std=c++17 -arch=sm_120 -I. -Xcompiler -pthread \
  tests/txid/aligned_gpu_test.cu -o /tmp/proxichorizo-aligned-test
/tmp/proxichorizo-aligned-test 40
```

This test needs an idle SM120 GPU and runs substantial CPU hash verification.
On the tested RTX 5090, the fixed-seed 40-case run checked 50,081,923 nonces
and 7,241 valid hits, in
addition to 96 small self-check cases. It does not replace an end-to-end test
against the upstream wallet controller and node.
