# ProxiChorizo in a container. Build:  docker build -f packaging/docker/Dockerfile -t proxichorizo .
# Run:  docker run --rm --gpus all ghcr.io/ocminer/proxichorizo --proxi http://proxi.suprnova.cc:8100 --token TOKEN --seed-hex SEED --name rig1
# (the seed is the first 64 hex characters of "private_key" in the proxima.key the pool gave you; NONCE_SEEKER_SEED works too)
FROM nvidia/cuda:12.9.1-devel-ubuntu20.04 AS build
ARG ARCHS="-gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_120,code=sm_120 -gencode arch=compute_120,code=compute_120"
WORKDIR /src
COPY proxichorizo.cu proxichorizo_device.cuh ./
RUN nvcc -O3 -std=c++17 $ARCHS -Xcompiler -static-libstdc++ -Xcompiler -static-libgcc -o proxichorizo proxichorizo.cu -lpthread

FROM nvidia/cuda:12.9.1-base-ubuntu20.04
COPY --from=build /src/proxichorizo /usr/local/bin/proxichorizo
COPY vectors.json USAGE.md /opt/proxichorizo/
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID NVIDIA_DRIVER_CAPABILITIES=compute,utility
EXPOSE 4078
ENTRYPOINT ["/usr/local/bin/proxichorizo"]
CMD ["--help"]
