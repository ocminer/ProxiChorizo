# ProxiChorizo — CUDA nonce seeker for Proxima
NVCC ?= nvcc
# every supported generation; trim to your card for a faster build (e.g. ARCHS=-gencode arch=compute_86,code=sm_86)
ARCHS ?= -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 \
         -gencode arch=compute_89,code=sm_89 -gencode arch=compute_120,code=sm_120 -gencode arch=compute_120,code=compute_120
proxichorizo: proxichorizo.cu proxichorizo_device.cuh
	$(NVCC) -O3 -std=c++17 $(ARCHS) -o $@ proxichorizo.cu -lpthread
test: proxichorizo
	./proxichorizo --selftest vectors.json && ./proxichorizo --fetest && ./proxichorizo --verify
check:
	scripts/check-secrets.sh --all
clean:
	rm -f proxichorizo
.PHONY: test check clean
