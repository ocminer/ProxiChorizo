# ProxiChorizo OpenCL build. Needs an OpenCL ICD loader + headers (ROCm: /opt/rocm; Debian: ocl-icd-opencl-dev, opencl-headers).
CXX ?= g++
CL_INC ?= $(firstword $(wildcard /opt/rocm/include) /usr/include)
CL_LIB ?= $(firstword $(wildcard /opt/rocm/lib) /usr/lib/x86_64-linux-gnu)
# link the ICD loader by soname when the dev symlink libOpenCL.so is missing (plain ocl-icd-libopencl1)
CL_LDLIBS ?= $(if $(wildcard $(CL_LIB)/libOpenCL.so /usr/lib/x86_64-linux-gnu/libOpenCL.so),-L$(CL_LIB) -lOpenCL,-l:libOpenCL.so.1)
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wno-unused-function -I$(CL_INC)
proxichorizo-cl: proxichorizo_cl.cpp host_common.inc kernel_blob.h
	$(CXX) $(CXXFLAGS) -o $@ proxichorizo_cl.cpp $(CL_LDLIBS) -lpthread
host_common.inc kernel_blob.h: gen_host.py proxichorizo.cl ../proxichorizo.cu
	python3 gen_host.py
test: proxichorizo-cl
	./proxichorizo-cl --selftest && ./proxichorizo-cl --fetest && ./proxichorizo-cl --verify
clean:
	rm -f proxichorizo-cl host_common.inc kernel_blob.h
.PHONY: test clean
