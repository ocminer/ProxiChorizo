# ProxiChorizo — CUDA nonce seeker for Proxima. Multi-stage: build from source, ship a slim runtime.
# Built on Ubuntu 20.04 (glibc 2.31) so the same binary also runs on HiveOS/mmpOS hosts.
ARG CUDA_IMAGE=12.9.1
FROM nvidia/cuda:${CUDA_IMAGE}-devel-ubuntu20.04 AS build
ARG ARCHS="-gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_120,code=sm_120 -gencode arch=compute_120,code=compute_120"
WORKDIR /src
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-nacl && rm -rf /var/lib/apt/lists/*
COPY proxichorizo.cu proxichorizo_device.cuh platform_profiles.h txid_blake2b.h txid_cuda.cuh txid_aligned.cuh txid_vectors.h txid_blackwell.cu txid_blackwell.h txid_bw_math.cuh kernel_protection.cpp kernel_protection.h ./
COPY third_party/monocypher/ third_party/monocypher/
COPY scripts/build-protected.py scripts/protect-kernels.py scripts/
COPY tests/kernel_protection.py tests/
RUN --mount=type=secret,id=PX_KERNEL_BUILD_KEY,required=true \
    PX_KERNEL_BUILD_KEY="$(cat /run/secrets/PX_KERNEL_BUILD_KEY)" python3 scripts/build-protected.py --kind cuda --output proxichorizo --archs="$ARCHS" --require-key && strip proxichorizo

FROM nvidia/cuda:${CUDA_IMAGE}-base-ubuntu20.04
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=build /src/proxichorizo /app/proxichorizo
COPY docker/entrypoint.sh /app/entrypoint.sh
COPY README.md USAGE.md LICENSE LICENSE-Monocypher.txt NOTICE PLATFORMS.md STRATUM-SPEC.md vectors.json vectors-txid-blake2b.json /app/
COPY benchmarks/ /app/benchmarks/
RUN chmod +x /app/entrypoint.sh /app/proxichorizo
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID NVIDIA_DRIVER_CAPABILITIES=compute,utility
EXPOSE 4078
ENTRYPOINT ["/app/entrypoint.sh"]
