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

For the two-block CUDA path, build the following test for the selected card.
It checks all seven nonce splits, zero-length and partial tiles, multiple grid
strides, counter wrap, exact targets and hit-buffer overflow against full CPU hashes.

```sh
nvcc -O3 -std=c++17 -arch=sm_120 tests/txid/straddle_gpu_test.cu -o /tmp/proxi-straddle
/tmp/proxi-straddle
```

The aligned test needs an idle GPU and runs substantial CPU hash verification.
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
the seven two-block nonce splits across multiple RX 7900 XTX workgroup tiles,
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

The optional `proxima.aligned-tail.v1` extension has host-only regression tests:

```sh
(cd opencl && python3 gen_host.py)
g++ -O2 -std=c++17 tests/txid/zero_tail_test.cpp -o /tmp/proxi-zero-tail
/tmp/proxi-zero-tail
g++ -O2 -std=c++17 tests/txid/aligned_tail_protocol_test.cpp -pthread -ldl -o /tmp/proxi-tail-protocol
/tmp/proxi-tail-protocol
```

The first checks exact zero-tail compression and rejects mutated final blocks or
incompatible shapes. The second uses the actual miner client and a loopback pool:
capability acceptance, refusal, missing reply, malformed reply, unknown-method
disconnect/reconnect, ordinary/padded layout changes and CPU-verified submissions.

`zero_tail_gpu_test.cu` and `zero_tail_opencl_test.cpp` check the new GPU path with
41,000 reference candidates, two 5,000-hit overflow cases, partial launches and
32-/64-bit nonce wrap. Their stdout contains nonce and hash-word pairs for an
independent hashlib check; the diagnostic prefix is `(i*37+11)&255` for its first
384 bytes, followed by 120 zero bytes. The OpenCL test takes the selected PCI bus
number as its sole argument and creates only that device's context.

`zero_tail_gpu_test.cu` accepts an optional `256` or `512` argument to test the
specific launch size. The latter exercises the additional Pascal choice.
`blackwell_zero_tail_test.cu` links the production `txid_blackwell.cu` module and
checks the specialized launch geometries, ordinary fallback, four prefix lengths,
partial grids, counter boundaries and overflow. For example, from the repo root
with CUDA 12.9 and an available Blackwell GPU selected through
`CUDA_VISIBLE_DEVICES`:

```sh
nvcc -O3 -std=c++17 -gencode arch=compute_120,code=compute_120 \
  -c txid_blackwell.cu -o /tmp/proxi-tail-bw.o
nvcc -O3 -std=c++17 -arch=sm_120 -Xcompiler -pthread \
  tests/txid/blackwell_zero_tail_test.cu /tmp/proxi-tail-bw.o -o /tmp/proxi-tail-bw
PX_TXID_BW_ZERO=1 /tmp/proxi-tail-bw > /tmp/proxi-tail-lb2.txt
PX_TXID_BW_ZERO=2 /tmp/proxi-tail-bw > /tmp/proxi-tail-lb3.txt
PX_TXID_BW_ZERO=3 /tmp/proxi-tail-bw > /tmp/proxi-tail-768.txt
/tmp/proxi-tail-bw --ordinary-kernel > /tmp/proxi-tail-ordinary.txt
python3 tests/txid/check_zero_tail_output.py blackwell /tmp/proxi-tail-lb2.txt
python3 tests/txid/check_zero_tail_output.py blackwell /tmp/proxi-tail-lb3.txt
python3 tests/txid/check_zero_tail_output.py blackwell /tmp/proxi-tail-768.txt
python3 tests/txid/check_zero_tail_output.py blackwell /tmp/proxi-tail-ordinary.txt
```

The checker uses Python's independent BLAKE2b implementation to reconstruct the
complete digest and verify the GPU's returned hash word. It also accepts `pascal`
with the output of `zero_tail_gpu_test.cu`. Do not compile these diagnostic tests
with `NDEBUG`, which would remove their assertions.

For controlled layout comparisons use `--hash-bench 30 --hash-prefix-len 504
--hash-zero-tail`. `PX_TXID_ZERO_TAIL=0` retains the previous kernel on the same
zero-tail input; `=1` forces the new path for diagnostics, with startup checks
still required. Without an override, the miner compares both paths on the card
and keeps the previous kernel unless the new one clears the timing margin.
Do not compare an ordinary random 504-byte benchmark with a zero-tail result
and label it a same-input speedup. Normal jobs and hashrate accounting are unchanged.

On a supported RX 7900 XTX or RX 7600 XT, OpenCL startup logs when it enables
the split-nonce and scheduled zero-tail kernels. For a same-device comparison,
set `PX_TXID_RDNA3=0` to use the standard kernels on the same bytes. Exercise
both a 504-byte zero-tail job and an ordinary 381-byte job. The optional kernels
take part in the normal startup hash checks.
