# ProxiChorizo — CUDA nonce seeker for Proxima. Multi-stage: build from source, ship a slim runtime.
# Built on Ubuntu 20.04 (glibc 2.31) so the same binary also runs on HiveOS/mmpOS hosts.
ARG CUDA_IMAGE=12.9.1
FROM nvidia/cuda:${CUDA_IMAGE}-devel-ubuntu20.04 AS build
ARG ARCHS="-gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_120,code=sm_120 -gencode arch=compute_120,code=compute_120"
WORKDIR /src
COPY proxichorizo.cu proxichorizo_device.cuh ./
RUN nvcc -O3 -std=c++17 $ARCHS -Xcompiler -static-libstdc++ -Xcompiler -static-libgcc -o proxichorizo proxichorizo.cu -lpthread && strip proxichorizo

FROM nvidia/cuda:${CUDA_IMAGE}-base-ubuntu20.04
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=build /src/proxichorizo /app/proxichorizo
COPY docker/entrypoint.sh /app/entrypoint.sh
COPY README.md USAGE.md LICENSE NOTICE vectors.json /app/
RUN chmod +x /app/entrypoint.sh /app/proxichorizo
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID NVIDIA_DRIVER_CAPABILITIES=compute,utility
EXPOSE 4078
ENTRYPOINT ["/app/entrypoint.sh"]
