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

The aligned test also accepts the qualified RTX 3070 (sm_86), RTX 4070 Ti SUPER
(sm_89) and GTX 1080 Ti (sm_61) profiles. Compile for the selected card's architecture.
It prints each hit as JSON on stdout for an independent Python hashlib check;
progress and full CPU/GPU hit-set comparisons go to stderr.

The OpenCL plan has a GPU regression that creates a context only on the selected
PCI bus. It covers all 32 original shapes, the optimized word-14/15 layout,
partial launches, 32-bit and 64-bit wraps, exact targets, full hit buffers and
negative execution statuses. Example for a GPU on PCI bus 3:

```sh
(cd opencl && python3 gen_host.py)
g++ -O2 -std=c++17 tests/txid/opencl_plan_test.cpp -lOpenCL -lpthread -ldl -o /tmp/proxi-opencl-plan
/tmp/proxi-opencl-plan 3
PX_OPENCL_LOW_CPU=1 /tmp/proxi-opencl-plan 3
python3 scripts/generate-txid-plan.py --check
```

`fake_pool.py --prefix-lens 369 --hash-seed example` provides identical synthetic
job prefixes for full-miner comparisons. Every submitted hash is independently
checked with Python hashlib. This is a protocol test, not on-chain confirmation.
