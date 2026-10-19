// clmulbench — peak integer-multiply throughput of an OpenCL GPU (what bounds ProxiChorizo on AMD).
// Build: g++ -O2 -std=c++17 -I/opt/rocm/include -o clmulbench tools/clmulbench.cpp -L/opt/rocm/lib -lOpenCL ; run: ./clmulbench [device]
// peak integer-multiply throughput of an OpenCL device: 32x32->64 mad (what fe_mul uses),
// 32x32->32 mul_lo, 24-bit mad, and 32-bit add for reference. Independent accumulators, unrolled.
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <stdio.h>
#include <string.h>
#include <chrono>
#include <string>
static const char *src = R"(
__kernel void mad64(__global ulong *out, uint iters, uint seed) {
    uint x0 = seed ^ get_global_id(0), x1 = x0 * 3u + 1, x2 = x0 * 5u + 7, x3 = x0 * 7u + 11, x4 = x0*11u+13, x5 = x0*13u+17, x6=x0*17u+19, x7=x0*19u+23;
    ulong a0 = 0, a1 = 1, a2 = 2, a3 = 3, a4 = 4, a5 = 5, a6 = 6, a7 = 7;
    for (uint i = 0; i < iters; i++) {
        a0 += (ulong)x0 * x1; a1 += (ulong)x1 * x2; a2 += (ulong)x2 * x3; a3 += (ulong)x3 * x4;
        a4 += (ulong)x4 * x5; a5 += (ulong)x5 * x6; a6 += (ulong)x6 * x7; a7 += (ulong)x7 * x0;
        x0 += 0x9E3779B9u; x1 += 0x7F4A7C15u; x2 ^= (uint)a0; x3 ^= (uint)a1; x4 += 3u; x5 += 5u; x6 ^= (uint)a2; x7 += 7u;
    }
    out[get_global_id(0)] = a0 ^ a1 ^ a2 ^ a3 ^ a4 ^ a5 ^ a6 ^ a7;
}
__kernel void mul32(__global ulong *out, uint iters, uint seed) {
    uint x0 = seed ^ get_global_id(0), x1 = x0 * 3u + 1, x2 = x0 * 5u + 7, x3 = x0 * 7u + 11, x4 = x0*11u+13, x5 = x0*13u+17, x6=x0*17u+19, x7=x0*19u+23;
    uint a0 = 0, a1 = 1, a2 = 2, a3 = 3, a4 = 4, a5 = 5, a6 = 6, a7 = 7;
    for (uint i = 0; i < iters; i++) {
        a0 += x0 * x1; a1 += x1 * x2; a2 += x2 * x3; a3 += x3 * x4; a4 += x4 * x5; a5 += x5 * x6; a6 += x6 * x7; a7 += x7 * x0;
        x0 += 0x9E3779B9u; x1 += 0x7F4A7C15u; x2 ^= a0; x3 ^= a1; x4 += 3u; x5 += 5u; x6 ^= a2; x7 += 7u;
    }
    out[get_global_id(0)] = a0 ^ a1 ^ a2 ^ a3 ^ a4 ^ a5 ^ a6 ^ a7;
}
__kernel void mulhi32(__global ulong *out, uint iters, uint seed) {
    uint x0 = seed ^ get_global_id(0), x1 = x0 * 3u + 1, x2 = x0 * 5u + 7, x3 = x0 * 7u + 11, x4 = x0*11u+13, x5 = x0*13u+17, x6=x0*17u+19, x7=x0*19u+23;
    uint a0 = 0, a1 = 1, a2 = 2, a3 = 3, a4 = 4, a5 = 5, a6 = 6, a7 = 7;
    for (uint i = 0; i < iters; i++) {
        a0 += mul_hi(x0, x1); a1 += mul_hi(x1, x2); a2 += mul_hi(x2, x3); a3 += mul_hi(x3, x4); a4 += mul_hi(x4, x5); a5 += mul_hi(x5, x6); a6 += mul_hi(x6, x7); a7 += mul_hi(x7, x0);
        x0 += 0x9E3779B9u; x1 += 0x7F4A7C15u; x2 ^= a0; x3 ^= a1; x4 += 3u; x5 += 5u; x6 ^= a2; x7 += 7u;
    }
    out[get_global_id(0)] = a0 ^ a1 ^ a2 ^ a3 ^ a4 ^ a5 ^ a6 ^ a7;
}
__kernel void k_mad24(__global ulong *out, uint iters, uint seed) {
    uint x0 = (seed ^ get_global_id(0)) & 0xffffff, x1 = (x0 * 3u + 1) & 0xffffff, x2 = (x0 * 5u + 7) & 0xffffff, x3 = (x0 * 7u + 11) & 0xffffff, x4=(x0*11u+13)&0xffffff, x5=(x0*13u+17)&0xffffff, x6=(x0*17u+19)&0xffffff, x7=(x0*19u+23)&0xffffff;
    uint a0 = 0, a1 = 1, a2 = 2, a3 = 3, a4 = 4, a5 = 5, a6 = 6, a7 = 7;
    for (uint i = 0; i < iters; i++) {
        a0 = mad24(x0, x1, a0); a1 = mad24(x1, x2, a1); a2 = mad24(x2, x3, a2); a3 = mad24(x3, x4, a3); a4 = mad24(x4, x5, a4); a5 = mad24(x5, x6, a5); a6 = mad24(x6, x7, a6); a7 = mad24(x7, x0, a7);
        x0 = (x0 + 0x9E3779u) & 0xffffff; x1 = (x1 + 0x7F4A7Cu) & 0xffffff; x2 = (x2 ^ a0) & 0xffffff; x3 = (x3 ^ a1) & 0xffffff; x4 = (x4+3u)&0xffffff; x5=(x5+5u)&0xffffff; x6=(x6^a2)&0xffffff; x7=(x7+7u)&0xffffff;
    }
    out[get_global_id(0)] = a0 ^ a1 ^ a2 ^ a3 ^ a4 ^ a5 ^ a6 ^ a7;
}
__kernel void add32(__global ulong *out, uint iters, uint seed) {
    uint x0 = seed ^ get_global_id(0), x1 = x0 * 3u + 1, x2 = x0 * 5u + 7, x3 = x0 * 7u + 11, x4 = x0*11u+13, x5 = x0*13u+17, x6=x0*17u+19, x7=x0*19u+23;
    uint a0 = 0, a1 = 1, a2 = 2, a3 = 3, a4 = 4, a5 = 5, a6 = 6, a7 = 7;
    for (uint i = 0; i < iters; i++) {
        a0 += x0 ^ x1; a1 += x1 ^ x2; a2 += x2 ^ x3; a3 += x3 ^ x4; a4 += x4 ^ x5; a5 += x5 ^ x6; a6 += x6 ^ x7; a7 += x7 ^ x0;
        x0 += 0x9E3779B9u; x1 += 0x7F4A7C15u; x2 ^= a0; x3 ^= a1; x4 += 3u; x5 += 5u; x6 ^= a2; x7 += 7u;
    }
    out[get_global_id(0)] = a0 ^ a1 ^ a2 ^ a3 ^ a4 ^ a5 ^ a6 ^ a7;
}
)";
int main(int argc, char **argv) {
    int devidx = argc > 1 ? atoi(argv[1]) : 0;
    cl_platform_id p; cl_uint np; clGetPlatformIDs(1, &p, &np);
    cl_device_id ds[8]; cl_uint nd; clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 8, ds, &nd); cl_device_id d = ds[devidx];
    char nm[128]; clGetDeviceInfo(d, CL_DEVICE_NAME, sizeof nm, nm, 0); cl_uint cus; clGetDeviceInfo(d, CL_DEVICE_MAX_COMPUTE_UNITS, 4, &cus, 0); cl_uint mhz; clGetDeviceInfo(d, CL_DEVICE_MAX_CLOCK_FREQUENCY, 4, &mhz, 0);
    printf("%s: %u CUs, %u MHz\n", nm, cus, mhz);
    cl_int err; cl_context ctx = clCreateContext(0, 1, &d, 0, 0, &err); cl_command_queue q = clCreateCommandQueue(ctx, d, 0, &err);
    cl_program pr = clCreateProgramWithSource(ctx, 1, &src, 0, &err); if (clBuildProgram(pr, 1, &d, "", 0, 0)) { size_t l; clGetProgramBuildInfo(pr, d, CL_PROGRAM_BUILD_LOG, 0, 0, &l); std::string log(l, 0); clGetProgramBuildInfo(pr, d, CL_PROGRAM_BUILD_LOG, l, log.data(), 0); printf("%s\n", log.c_str()); return 1; }
    size_t g = (size_t)cus * 256 * 16, lsz = 256; cl_uint iters = 4096; cl_uint seed = 12345;
    cl_mem out = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, g * 8, 0, &err);
    for (const char *kn : { "add32", "mul32", "mulhi32", "k_mad24", "mad64" }) {
        cl_kernel k = clCreateKernel(pr, kn, &err);
        clSetKernelArg(k, 0, sizeof out, &out); clSetKernelArg(k, 1, 4, &iters); clSetKernelArg(k, 2, 4, &seed);
        clEnqueueNDRangeKernel(q, k, 1, 0, &g, &lsz, 0, 0, 0); clFinish(q);
        auto t0 = std::chrono::steady_clock::now(); int reps = 5;
        for (int r = 0; r < reps; r++) clEnqueueNDRangeKernel(q, k, 1, 0, &g, &lsz, 0, 0, 0); clFinish(q);
        double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        double ops = (double)g * iters * 8 * reps;
        printf("  %-8s %8.1f Gop/s  = %.2f op/lane-cycle (lanes = CUs*64)\n", kn, ops / s / 1e9, ops / s / ((double)cus * 64 * mhz * 1e6));
    }
    return 0;
}
