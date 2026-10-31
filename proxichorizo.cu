// ProxiChorizo - Copyright (c) 2026 suprnova.cc. ProxiChorizo License 1.0: free to run for mining;
// reuse in commercial or closed-source miners needs written permission. See LICENSE.
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>
#include <algorithm>
#include <cctype>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <random>
#include <fstream>
#include <sstream>
#include <map>
#include <cmath>
#include <cerrno>
#define PX_VERSION "1.0.0"
#include <signal.h>
// sockets: POSIX or Winsock behind four helpers
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <winsock2.h>
#include <ws2tcpip.h>
#include <process.h>
#include <io.h>
#pragma comment(lib, "ws2_32.lib")
#define px_isatty(fd) _isatty(fd)
#define px_fileno(f) _fileno(f)
typedef SOCKET sock_t;
typedef long long ssize_t;
#define SOCK_INVALID INVALID_SOCKET
#ifndef MSG_NOSIGNAL
#define MSG_NOSIGNAL 0
#endif
#define getpid _getpid
static void sock_init() { WSADATA w; WSAStartup(MAKEWORD(2, 2), &w); }
static void sock_close(sock_t s) { closesocket(s); }
static void sock_timeout(sock_t s, int secs) { DWORD ms = (DWORD)secs * 1000; setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, (const char *)&ms, sizeof ms); setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, (const char *)&ms, sizeof ms); }
static void sock_timeout_ms(sock_t s, int ms) { DWORD v = (DWORD)ms; setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, (const char *)&v, sizeof v); }
#else
#include <sys/socket.h>
#include <sys/types.h>
#include <netdb.h>
#include <unistd.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
typedef int sock_t;
#define SOCK_INVALID (-1)
#define px_isatty(fd) isatty(fd)
#define px_fileno(f) fileno(f)
static void sock_init() {}
static void sock_close(sock_t s) { close(s); }
static void sock_timeout(sock_t s, int secs) { struct timeval tv = { secs, 0 }; setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv); setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof tv); }
static void sock_timeout_ms(sock_t s, int ms) { struct timeval tv = { ms / 1000, (ms % 1000) * 1000 }; setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv); }
#endif

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(2);} } while (0)

// ----------------------------------------------------------------------------
// field element, radix 2^51
// ----------------------------------------------------------------------------
struct JobDev {
    uint8_t pk[32];
    uint8_t prefix[64];
    int prefix_len;
    uint8_t beat[64];
    int k;
};
struct Hit { uint64_t nonce; uint8_t beta[64]; };
#define MAX_HITS 256

// ---- kernel variants ---------------------------------------------------------
// Each variant is the whole device side compiled with one setting of the field
// implementation, the scalar-multiplication window and the launch bounds. The
// platform table below picks a default per GPU generation; --variant overrides,
// --autotune measures them all on the card at hand.
struct VariantOps {
    const char *name;
    bool fe32; int window; int lbBlocks; bool compact;
    void (*set_scalar)(const uint8_t *scalar);
    int (*init)(const uint8_t *scalar, uint8_t pk_out[32]);
    void *(*alloc)(size_t n);          // per-device scratch for n nonces per launch (nullptr for variants that need none)
    void (*free_)(void *scratch);
    void (*mine)(int blocks, int threads, const JobDev *job, uint64_t nonce_base, uint64_t stride, int iters, Hit *hits, unsigned *nhits, void *scratch);
    int (*test)(const JobDev *d_job, uint64_t nonce, uint8_t beta[64], uint8_t h[32], uint8_t cg[32]);
    int (*fetest)(uint32_t seed, int n, uint8_t dump[128]);
};

// PX_LEAN: development builds compile only the shipped Blackwell variant and the comba variant (minutes instead of ~7 per arch)
#ifndef PX_LEAN
#define PX_FE32 1
#define PX_WINDOW 0
#define PX_LB_BLOCKS 2
#define PX_VARIANT_NAME "fe32_w0_lb2"
namespace v_fe32_w0_lb2 {
#include "proxichorizo_device.cuh"
}
#undef PX_FE32
#undef PX_WINDOW
#undef PX_LB_BLOCKS
#undef PX_VARIANT_NAME
#undef PX_LB_THREADS
#undef PX_COMPACT

#define PX_FE32 1
#define PX_WINDOW 4
#define PX_LB_BLOCKS 2
#define PX_VARIANT_NAME "fe32_w4_lb2"
namespace v_fe32_w4_lb2 {
#include "proxichorizo_device.cuh"
}
#undef PX_FE32
#undef PX_WINDOW
#undef PX_LB_BLOCKS
#undef PX_VARIANT_NAME
#undef PX_LB_THREADS
#undef PX_COMPACT

#define PX_WINDOW 4
#define PX_LB_BLOCKS 1
#define PX_VARIANT_NAME "fe51_w4_lb1"
namespace v_fe51_w4_lb1 {
#include "proxichorizo_device.cuh"
}
#undef PX_WINDOW
#undef PX_LB_BLOCKS
#undef PX_VARIANT_NAME
#undef PX_LB_THREADS
#undef PX_COMPACT

#define PX_FE32 1
#define PX_WINDOW 0
#define PX_LB_BLOCKS 2
#define PX_COMPACT 1
#define PX_VARIANT_NAME "fe32_w0_lb2_c"
namespace v_fe32_w0_lb2_c {
#include "proxichorizo_device.cuh"
}
#undef PX_FE32
#undef PX_WINDOW
#undef PX_LB_BLOCKS
#undef PX_COMPACT
#undef PX_VARIANT_NAME
#undef PX_LB_THREADS
#undef PX_COMPACT

#define PX_FE32 1
#define PX_WINDOW 4
#define PX_LB_BLOCKS 2
#define PX_COMPACT 1
#define PX_VARIANT_NAME "fe32_w4_lb2_c"
namespace v_fe32_w4_lb2_c {
#include "proxichorizo_device.cuh"
}
#undef PX_FE32
#undef PX_WINDOW
#undef PX_LB_BLOCKS
#undef PX_COMPACT
#undef PX_VARIANT_NAME
#undef PX_LB_THREADS
#undef PX_COMPACT

#endif // PX_LEAN
#define PX_FE32 1
#define PX_WINDOW 5
#define PX_LB_BLOCKS 2
#define PX_COMPACT 1
#define PX_VARIANT_NAME "fe32_wnaf_lb2_c"
namespace v_fe32_wnaf_lb2_c {
#include "proxichorizo_device.cuh"
}
#undef PX_FE32
#undef PX_WINDOW
#undef PX_LB_BLOCKS
#undef PX_COMPACT
#undef PX_VARIANT_NAME
#undef PX_LB_THREADS

#define PX_FE32 1
#define PX_COMBA 1
#define PX_WINDOW 5
#define PX_LB_BLOCKS 2
#define PX_COMPACT 1
#define PX_VARIANT_NAME "fe32c_wnaf_lb2_c"
namespace v_fe32c_wnaf_lb2_c {
#include "proxichorizo_device.cuh"
}
#undef PX_FE32
#undef PX_COMBA
#undef PX_WINDOW
#undef PX_LB_BLOCKS
#undef PX_COMPACT
#undef PX_VARIANT_NAME
#undef PX_LB_THREADS

// the 0.5.8 default for every measured NVIDIA generation: comba field arithmetic, the cofactor folded into
// the wNAF scalar, a separate normalisation stage sharing one inversion over 16 points, the hash-to-curve
// kernel at 3 blocks per SM (PLATFORMS.md, round 10)
#define PX_FE32 1
#define PX_COMBA 1
#define PX_MERGE8 1
#define PX_WINDOW 5
#define PX_LB_BLOCKS 2
#define PX_LB_H2C 3
#define PX_COMPACT 1
#define PX_SPLIT_FINISH 16
#define PX_VARIANT_NAME "fe32c_wnaf_s16_c"
namespace v_fe32c_wnaf_s16_c {
#include "proxichorizo_device.cuh"
}
#undef PX_FE32
#undef PX_COMBA
#undef PX_MERGE8
#undef PX_WINDOW
#undef PX_LB_BLOCKS
#undef PX_LB_H2C
#undef PX_COMPACT
#undef PX_VARIANT_NAME
#undef PX_LB_THREADS
#undef PX_SPLIT_FINISH

#ifdef PX_DEV_EXTRA
// development-only comba combinations (not in release builds)
#define PX_FE32 1
#define PX_COMBA 1
#define PX_WINDOW 5
#define PX_LB_BLOCKS 2
#define PX_LB_MULT_BLOCKS 1
#define PX_COMPACT 1
#define PX_VARIANT_NAME "fe32c_wnaf_lb1_c"
namespace v_fe32c_wnaf_lb1_c {
#include "proxichorizo_device.cuh"
}
#undef PX_FE32
#undef PX_COMBA
#undef PX_WINDOW
#undef PX_LB_BLOCKS
#undef PX_LB_MULT_BLOCKS
#undef PX_COMPACT
#undef PX_VARIANT_NAME
#undef PX_LB_THREADS

#define PX_FE32 1
#define PX_COMBA 1
#define PX_WINDOW 5
#define PX_LB_BLOCKS 2
#define PX_COMPACT 1
#define PX_SPLIT_FINISH 16
#define PX_VARIANT_NAME "fe32c_wnaf_split16_c"
namespace v_fe32c_wnaf_split16_c {
#include "proxichorizo_device.cuh"
}
#undef PX_FE32
#undef PX_COMBA
#undef PX_WINDOW
#undef PX_LB_BLOCKS
#undef PX_COMPACT
#undef PX_VARIANT_NAME
#undef PX_LB_THREADS
#undef PX_SPLIT_FINISH

#define PX_FE32 1
#define PX_COMBA 1
#define PX_MERGE8 1
#define PX_WINDOW 5
#define PX_LB_BLOCKS 2
#define PX_COMPACT 1
#define PX_VARIANT_NAME "fe32c_wnaf_m8_c"
namespace v_fe32c_wnaf_m8_c {
#include "proxichorizo_device.cuh"
}
#undef PX_FE32
#undef PX_COMBA
#undef PX_MERGE8
#undef PX_WINDOW
#undef PX_LB_BLOCKS
#undef PX_COMPACT
#undef PX_VARIANT_NAME
#undef PX_LB_THREADS

#define PX_FE32 1
#define PX_COMBA 1
#define PX_WINDOW 5
#define PX_LB_BLOCKS 2
#define PX_COMPACT 1
#define PX_INV_BATCH 4
#define PX_VARIANT_NAME "fe32c_wnaf_inv4_c"
namespace v_fe32c_wnaf_inv4_c {
#include "proxichorizo_device.cuh"
}
#undef PX_FE32
#undef PX_COMBA
#undef PX_WINDOW
#undef PX_LB_BLOCKS
#undef PX_COMPACT
#undef PX_VARIANT_NAME
#undef PX_LB_THREADS
#define PX_FE32 1
#define PX_COMBA 1
#define PX_MERGE8 1
#define PX_WINDOW 5
#define PX_LB_BLOCKS 2
#define PX_COMPACT 1
#define PX_SPLIT_FINISH 16
#define PX_VARIANT_NAME "fe32c_wnaf_m8s16_c"
namespace v_fe32c_wnaf_m8s16_c {
#include "proxichorizo_device.cuh"
}
#undef PX_FE32
#undef PX_COMBA
#undef PX_MERGE8
#undef PX_WINDOW
#undef PX_LB_BLOCKS
#undef PX_COMPACT
#undef PX_VARIANT_NAME
#undef PX_LB_THREADS
#undef PX_SPLIT_FINISH
#endif // PX_DEV_EXTRA

#ifndef PX_LEAN
#define PX_FE32 1
#define PX_WINDOW 5
#define PX_LB_BLOCKS 2
#define PX_COMPACT 1
#define PX_SPLIT_FINISH 16
#define PX_VARIANT_NAME "fe32_wnaf_split16_c"
namespace v_fe32_wnaf_split16_c {
#include "proxichorizo_device.cuh"
}
#undef PX_FE32
#undef PX_WINDOW
#undef PX_LB_BLOCKS
#undef PX_COMPACT
#undef PX_VARIANT_NAME
#undef PX_LB_THREADS
#undef PX_LB_MULT_BLOCKS
#undef PX_SPLIT_FINISH
#undef PX_LINEAR_W4

#define PX_FE32 1
#define PX_WINDOW 4
#define PX_LB_BLOCKS 2
#define PX_COMPACT 1
#define PX_SPLIT_FINISH 8
#define PX_LINEAR_W4 1
#define PX_VARIANT_NAME "fe32_w4_linear_split8_c"
namespace v_fe32_w4_linear_split8_c {
#include "proxichorizo_device.cuh"
}
#undef PX_FE32
#undef PX_WINDOW
#undef PX_LB_BLOCKS
#undef PX_COMPACT
#undef PX_VARIANT_NAME
#undef PX_LB_THREADS
#undef PX_LB_MULT_BLOCKS
#undef PX_SPLIT_FINISH
#undef PX_LINEAR_W4

#endif // PX_LEAN
#ifdef PX_LEAN
#ifdef PX_DEV_EXTRA
static const VariantOps *VARIANTS[] = { &v_fe32_wnaf_lb2_c::ops, &v_fe32c_wnaf_lb2_c::ops, &v_fe32c_wnaf_s16_c::ops, &v_fe32c_wnaf_lb1_c::ops, &v_fe32c_wnaf_split16_c::ops, &v_fe32c_wnaf_inv4_c::ops, &v_fe32c_wnaf_m8_c::ops, &v_fe32c_wnaf_m8s16_c::ops };
#else
static const VariantOps *VARIANTS[] = { &v_fe32_wnaf_lb2_c::ops, &v_fe32c_wnaf_lb2_c::ops, &v_fe32c_wnaf_s16_c::ops };
#endif
#else
static const VariantOps *VARIANTS[] = { &v_fe32_w0_lb2::ops, &v_fe32_w4_lb2::ops, &v_fe32_w0_lb2_c::ops, &v_fe32_w4_lb2_c::ops, &v_fe32_wnaf_lb2_c::ops, &v_fe51_w4_lb1::ops, &v_fe32_wnaf_split16_c::ops, &v_fe32_w4_linear_split8_c::ops, &v_fe32c_wnaf_lb2_c::ops, &v_fe32c_wnaf_s16_c::ops };
#endif
static const int NVARIANTS = sizeof(VARIANTS) / sizeof(VARIANTS[0]);

// platform table: compute capability -> default variant and launch geometry,
// from the measurements in PLATFORMS.md; anything unlisted gets the last row
// blocksPerSM x multiprocessors is the launch size: the compacted pipeline runs
// its passes over the whole launch, so a bigger launch amortises the pass
// overhead (RTX PRO 6000: 32.4 MH/s at 8 blocks/SM x 4 nonces/thread, 36.3 at
// 64 x 1, 37.0 at 512 x 1; one nonce per thread with more blocks beats looping).
// measured: the row was benchmarked on that generation. For an unmeasured row the
// miner measures the two candidate kernels on first start and caches the result
// (quickTune), so every card gets a kernel chosen on itself rather than assumed.
#include "platform_profiles.h"
static uint8_t g_scalar[32];
static const VariantOps *variantByName(const std::string &n) { for (int i = 0; i < NVARIANTS; i++) if (n == VARIANTS[i]->name) return VARIANTS[i]; return nullptr; }
static const Platform &platformFor(const cudaDeviceProp &p) {
    return platformFor(p.major, p.minor, p.name);
}
// ----------------------------------------------------------------------------
// host side
// ----------------------------------------------------------------------------
static std::string hexs(const uint8_t *p, int n) { static const char *h = "0123456789abcdef"; std::string s; for (int i = 0; i < n; i++) { s += h[p[i] >> 4]; s += h[p[i] & 15]; } return s; }
static bool unhex(const std::string &s, std::vector<uint8_t> &out) {
    out.clear(); if (s.size() % 2) return false;
    auto v = [](char c) -> int { if (c >= '0' && c <= '9') return c - '0'; if (c >= 'a' && c <= 'f') return c - 'a' + 10; if (c >= 'A' && c <= 'F') return c - 'A' + 10; return -1; };
    for (size_t i = 0; i < s.size(); i += 2) { int a = v(s[i]), b = v(s[i + 1]); if (a < 0 || b < 0) return false; out.push_back(a * 16 + b); }
    return true;
}
static std::string now_str() {
    auto t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    char buf[32]; strftime(buf, sizeof buf, "%H:%M:%S", localtime(&t)); return buf;
}
// ---- console -----------------------------------------------------------------
// Every line goes to stderr with a HH:MM:SS prefix (HiveOS/mmpOS tee it into the
// miner log). Colours are ANSI escapes around the message body only: on when
// stderr is a terminal, off with --no-color, NO_COLOR in the environment, or when
// the output is a pipe or file, so logs stay plain text.
static std::mutex g_log_mu;
static bool g_color = false;
#define C_GREEN "\033[32m"
#define C_RED "\033[31m"
#define C_YELLOW "\033[33m"
#define C_CYAN "\033[36m"
#define C_BOLD "\033[1m"
#define C_RESET "\033[0m"
static void logln(const std::string &s) { std::lock_guard<std::mutex> l(g_log_mu); fprintf(stderr, "%s %s\n", now_str().c_str(), s.c_str()); }
static void logc(const char *color, const std::string &s) {
    if (!g_color) { logln(s); return; }
    std::lock_guard<std::mutex> l(g_log_mu); fprintf(stderr, "%s %s%s" C_RESET "\n", now_str().c_str(), color, s.c_str());
}
static void log_ok(const std::string &s) { logc(C_GREEN, s); }
static void log_err(const std::string &s) { logc(C_RED, s); }
static void log_warn(const std::string &s) { logc(C_YELLOW, s); }
// a banner line: no timestamp (the start-up block that ccminer/sgminer users expect)
static void bannerln(const std::string &s) { std::lock_guard<std::mutex> l(g_log_mu); fprintf(stderr, "%s\n", s.c_str()); }
struct ConsoleOpts { bool noColor = false, quiet = false, verbose = false; int logInterval = 30, maxTemp = 88, maxMemTemp = 92; };
static const char *CONSOLE_USAGE =
    "  --max-temp N       pause a GPU at N C core temperature, resume 8 C lower after at least 30 s (default 88; 0 = off); only that GPU pauses\n"
    "  --max-mem-temp N   the same for the memory temperature where the driver reports one (default 92; 0 = off)\n"
    "  --log-interval N   seconds between the per-GPU and total hashrate lines (default 30)\n"
    "  --quiet            only the total line, job changes, warnings and errors (no per-GPU or per-share lines)\n"
    "  --verbose          append kernel occupancy and idle diagnostics to the per-GPU lines\n"
    "  --no-color         plain output (colours are off anyway when stderr is not a terminal or NO_COLOR is set)\n";
// consumes --no-color/--quiet/--verbose/--log-interval/--max-temp/--max-mem-temp; i advances over a value
static bool parse_console_opt(const std::string &a, int &i, int argc, char **argv, ConsoleOpts &o) {
    auto nextint = [&](int dflt) { if (i + 1 < argc) return atoi(argv[++i]); return dflt; };
    if (a == "--no-color" || a == "--no-colour") o.noColor = true;
    else if (a == "--quiet" || a == "-q") o.quiet = true;
    else if (a == "--verbose") o.verbose = true;
    else if (a == "--log-interval") o.logInterval = std::max(1, nextint(30));
    else if (a == "--max-temp") o.maxTemp = nextint(88);
    else if (a == "--max-mem-temp") o.maxMemTemp = nextint(92);
    else return false;
    return true;
}
static void console_setup(const ConsoleOpts &o) {
    g_color = !o.noColor && px_isatty(px_fileno(stderr)) && getenv("NO_COLOR") == nullptr;
#ifdef _WIN32
    if (g_color) {  // Windows 10+: ANSI escapes need virtual terminal processing on the console
        HANDLE h = GetStdHandle(STD_ERROR_HANDLE); DWORD mode = 0;
        if (h == INVALID_HANDLE_VALUE || !GetConsoleMode(h, &mode) || !SetConsoleMode(h, mode | 0x0004)) g_color = false;
    }
#endif
}
static std::string fmt_uptime(uint64_t secs) {
    char b[48];
    if (secs >= 86400) snprintf(b, sizeof b, "%llud %02llu:%02llu:%02llu", (unsigned long long)secs / 86400, (unsigned long long)secs % 86400 / 3600, (unsigned long long)secs % 3600 / 60, (unsigned long long)secs % 60);
    else snprintf(b, sizeof b, "%02llu:%02llu:%02llu", (unsigned long long)secs / 3600, (unsigned long long)secs % 3600 / 60, (unsigned long long)secs % 60);
    return b;
}
// "NVIDIA GeForce RTX 5090" -> "RTX 5090"; AMD names are kept as the driver gives them
static std::string short_gpu_name(const std::string &n) {
    std::string s = n;
    for (const char *p : { "NVIDIA ", "GeForce ", "AMD ", "Radeon " }) if (s.rfind(p, 0) == 0) s = s.substr(strlen(p));
    return s;
}
// "0000:1d:00.0" -> "1D:00.0"
static std::string short_bus(const std::string &bus) {
    std::string s = bus.size() > 5 && bus[4] == ':' ? bus.substr(5) : bus;
    for (auto &c : s) c = (char)toupper((unsigned char)c);
    return s;
}

// host SHA-512 (for key expansion) — tiny, portable
static const uint64_t K512[80] = {
    0x428a2f98d728ae22ULL, 0x7137449123ef65cdULL, 0xb5c0fbcfec4d3b2fULL, 0xe9b5dba58189dbbcULL, 0x3956c25bf348b538ULL, 0x59f111f1b605d019ULL, 0x923f82a4af194f9bULL, 0xab1c5ed5da6d8118ULL,
    0xd807aa98a3030242ULL, 0x12835b0145706fbeULL, 0x243185be4ee4b28cULL, 0x550c7dc3d5ffb4e2ULL, 0x72be5d74f27b896fULL, 0x80deb1fe3b1696b1ULL, 0x9bdc06a725c71235ULL, 0xc19bf174cf692694ULL,
    0xe49b69c19ef14ad2ULL, 0xefbe4786384f25e3ULL, 0x0fc19dc68b8cd5b5ULL, 0x240ca1cc77ac9c65ULL, 0x2de92c6f592b0275ULL, 0x4a7484aa6ea6e483ULL, 0x5cb0a9dcbd41fbd4ULL, 0x76f988da831153b5ULL,
    0x983e5152ee66dfabULL, 0xa831c66d2db43210ULL, 0xb00327c898fb213fULL, 0xbf597fc7beef0ee4ULL, 0xc6e00bf33da88fc2ULL, 0xd5a79147930aa725ULL, 0x06ca6351e003826fULL, 0x142929670a0e6e70ULL,
    0x27b70a8546d22ffcULL, 0x2e1b21385c26c926ULL, 0x4d2c6dfc5ac42aedULL, 0x53380d139d95b3dfULL, 0x650a73548baf63deULL, 0x766a0abb3c77b2a8ULL, 0x81c2c92e47edaee6ULL, 0x92722c851482353bULL,
    0xa2bfe8a14cf10364ULL, 0xa81a664bbc423001ULL, 0xc24b8b70d0f89791ULL, 0xc76c51a30654be30ULL, 0xd192e819d6ef5218ULL, 0xd69906245565a910ULL, 0xf40e35855771202aULL, 0x106aa07032bbd1b8ULL,
    0x19a4c116b8d2d0c8ULL, 0x1e376c085141ab53ULL, 0x2748774cdf8eeb99ULL, 0x34b0bcb5e19b48a8ULL, 0x391c0cb3c5c95a63ULL, 0x4ed8aa4ae3418acbULL, 0x5b9cca4f7763e373ULL, 0x682e6ff3d6b2b8a3ULL,
    0x748f82ee5defb2fcULL, 0x78a5636f43172f60ULL, 0x84c87814a1f0ab72ULL, 0x8cc702081a6439ecULL, 0x90befffa23631e28ULL, 0xa4506cebde82bde9ULL, 0xbef9a3f7b2c67915ULL, 0xc67178f2e372532bULL,
    0xca273eceea26619cULL, 0xd186b8c721c0c207ULL, 0xeada7dd6cde0eb1eULL, 0xf57d4f7fee6ed178ULL, 0x06f067aa72176fbaULL, 0x0a637dc5a2c898a6ULL, 0x113f9804bef90daeULL, 0x1b710b35131c471bULL,
    0x28db77f523047d84ULL, 0x32caab7b40c72493ULL, 0x3c9ebe0a15c9bebcULL, 0x431d67c49c100d4cULL, 0x4cc5d4becb3e42b6ULL, 0x597f299cfc657e2aULL, 0x5fcb6fab3ad6faecULL, 0x6c44198c4a475817ULL
};
static inline uint64_t hrotr(uint64_t x, int n) { return (x >> n) | (x << (64 - n)); }
static void host_sha512(const uint8_t *msg, size_t len, uint8_t out[64]) {
    uint64_t st[8] = { 0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL, 0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL, 0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL, 0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL };
    std::vector<uint8_t> m(msg, msg + len);
    m.push_back(0x80);
    while (m.size() % 128 != 112) m.push_back(0);
    uint64_t bits = (uint64_t)len * 8;
    for (int i = 0; i < 8; i++) m.push_back(0);
    for (int i = 0; i < 8; i++) m.push_back(bits >> (56 - 8 * i));
    for (size_t off = 0; off < m.size(); off += 128) {
        uint64_t W[80];
        for (int i = 0; i < 16; i++) { uint64_t w = 0; for (int j = 0; j < 8; j++) w = (w << 8) | m[off + i * 8 + j]; W[i] = w; }
        for (int i = 16; i < 80; i++) {
            uint64_t s0 = hrotr(W[i - 15], 1) ^ hrotr(W[i - 15], 8) ^ (W[i - 15] >> 7);
            uint64_t s1 = hrotr(W[i - 2], 19) ^ hrotr(W[i - 2], 61) ^ (W[i - 2] >> 6);
            W[i] = W[i - 16] + s0 + W[i - 7] + s1;
        }
        uint64_t a = st[0], b = st[1], c = st[2], d = st[3], e = st[4], f = st[5], g = st[6], h = st[7];
        for (int i = 0; i < 80; i++) {
            uint64_t S1 = hrotr(e, 14) ^ hrotr(e, 18) ^ hrotr(e, 41), ch = (e & f) ^ (~e & g);
            uint64_t t1 = h + S1 + ch + K512[i] + W[i];
            uint64_t S0 = hrotr(a, 28) ^ hrotr(a, 34) ^ hrotr(a, 39), mj = (a & b) ^ (a & c) ^ (b & c);
            uint64_t t2 = S0 + mj;
            h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
        }
        st[0] += a; st[1] += b; st[2] += c; st[3] += d; st[4] += e; st[5] += f; st[6] += g; st[7] += h;
    }
    for (int i = 0; i < 8; i++) for (int j = 0; j < 8; j++) out[i * 8 + j] = st[i] >> (56 - 8 * j);
}

// --- minimal JSON field extraction (flat objects with string/number values) ---
static bool json_find(const std::string &s, const char *key, size_t &pos) {
    std::string k = std::string("\"") + key + "\"";
    size_t p = s.find(k);
    if (p == std::string::npos) return false;
    p = s.find(':', p + k.size());
    if (p == std::string::npos) return false;
    p++;
    while (p < s.size() && (s[p] == ' ' || s[p] == '\t' || s[p] == '\n' || s[p] == '\r')) p++;
    pos = p; return true;
}
static bool json_u64(const std::string &s, const char *key, uint64_t &out) {
    size_t p; if (!json_find(s, key, p)) return false;
    bool neg = false; if (s[p] == '-') { neg = true; p++; }
    if (p >= s.size() || !isdigit((unsigned char)s[p])) return false;
    uint64_t v = 0; while (p < s.size() && isdigit((unsigned char)s[p])) { v = v * 10 + (s[p] - '0'); p++; }
    out = neg ? 0 : v; return true;
}
static bool json_str(const std::string &s, const char *key, std::string &out) {
    size_t p; if (!json_find(s, key, p)) return false;
    if (s[p] != '"') return false;
    size_t e = s.find('"', p + 1); if (e == std::string::npos) return false;
    out = s.substr(p + 1, e - p - 1); return true;
}
static bool json_bool(const std::string &s, const char *key, bool &out) {
    size_t p; if (!json_find(s, key, p)) return false;
    if (s.compare(p, 4, "true") == 0) { out = true; return true; }
    if (s.compare(p, 5, "false") == 0) { out = false; return true; }
    return false;
}

// --- tiny HTTP/1.1 client ---
// stratum+tcp://host:port is a stratum pool (one persistent connection, see stratum_client);
// http://host:port (or host:port) the HTTP seeker protocol of `proxi node mine` and older pools
struct Url { std::string host; int port = 80; bool stratum = false; bool tls = false; };
static bool parse_url(const std::string &u, Url &out) {
    std::string s = u;
    if (s.rfind("stratum+tcp://", 0) == 0) { s = s.substr(14); out.stratum = true; }
    else if (s.rfind("stratum+ssl://", 0) == 0 || s.rfind("stratum+tls://", 0) == 0) { s = s.substr(14); out.stratum = out.tls = true; }
    else if (s.rfind("http://", 0) == 0) s = s.substr(7);
    else if (s.rfind("https://", 0) == 0) { fprintf(stderr, "https is not supported\n"); return false; }
    size_t slash = s.find('/'); if (slash != std::string::npos) s = s.substr(0, slash);
    size_t colon = s.find(':');
    if (colon == std::string::npos) {
        if (out.stratum) { fprintf(stderr, "a stratum URL needs a port: stratum+tcp://host:port or stratum+ssl://host:port\n"); return false; }
        out.host = s; out.port = 80;
    }
    else { out.host = s.substr(0, colon); out.port = atoi(s.substr(colon + 1).c_str()); }
    return !out.host.empty() && out.port > 0 && out.port < 65536;
}
// a TCP connection to u with send/receive timeouts; SOCK_INVALID and err on failure
static sock_t tcp_connect(const Url &u, int timeout_s, std::string &err) {
    struct addrinfo hints = {}, *res = nullptr;
    hints.ai_family = AF_UNSPEC; hints.ai_socktype = SOCK_STREAM;
    char portstr[16]; snprintf(portstr, sizeof portstr, "%d", u.port);
    if (getaddrinfo(u.host.c_str(), portstr, &hints, &res) != 0) { err = "dns failure for " + u.host; return SOCK_INVALID; }
    sock_t fd = SOCK_INVALID;
    for (auto *ai = res; ai; ai = ai->ai_next) {
        fd = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (fd == SOCK_INVALID) continue;
        sock_timeout(fd, timeout_s);
        int one = 1; setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, (const char *)&one, sizeof one);
        if (connect(fd, ai->ai_addr, (int)ai->ai_addrlen) == 0) break;
        sock_close(fd); fd = SOCK_INVALID;
    }
    freeaddrinfo(res);
    if (fd == SOCK_INVALID) err = "cannot connect to " + u.host + ":" + portstr;
    return fd;
}
// a recv that failed because its timeout ran out (not because the connection broke)
// wait until the socket has data to read, at most ms; false on timeout
static bool sock_readable(sock_t s, int ms) {
    fd_set r; FD_ZERO(&r); FD_SET(s, &r);
    struct timeval tv = { ms / 1000, (ms % 1000) * 1000 };
    return select((int)(s + 1), &r, nullptr, nullptr, &tv) > 0;
}
static bool sock_timed_out() {
#ifdef _WIN32
    int e = WSAGetLastError(); return e == WSAETIMEDOUT || e == WSAEWOULDBLOCK;
#else
    return errno == EAGAIN || errno == EWOULDBLOCK;
#endif
}

// ---- TLS for stratum+ssl:// ------------------------------------------------------
// Through OpenSSL, loaded at run time like NVML: the system's libssl (3 or 1.1) on Linux,
// the libssl/libcrypto DLLs shipped next to the miner on Windows. Nothing to install for a
// plain stratum+tcp:// pool. The pool's certificate chain and host name are checked against
// the system's CA store (and cacert.pem next to the miner, or --ca-file); --tls-insecure
// skips the check, for a private pool with its own certificate.
#ifndef _WIN32
#include <dlfcn.h>
#include <limits.h>
#endif
namespace tls {
static int (*init_ssl)(uint64_t, const void *) = nullptr;
static const void *(*client_method)() = nullptr;
static void *(*ctx_new)(const void *) = nullptr;
static int (*ctx_default_paths)(void *) = nullptr;
static int (*ctx_load_locations)(void *, const char *, const char *) = nullptr;
static void (*ctx_set_verify)(void *, int, void *) = nullptr;
static void *(*ssl_new)(void *) = nullptr;
static int (*set_fd)(void *, int) = nullptr;
static long (*ssl_ctrl)(void *, int, long, void *) = nullptr;
static int (*set1_host)(void *, const char *) = nullptr;
static int (*ssl_connect)(void *) = nullptr;
static long (*verify_result)(const void *) = nullptr;
static int (*ssl_read)(void *, void *, int) = nullptr;
static int (*ssl_pending)(const void *) = nullptr;
static int (*ssl_write)(void *, const void *, int) = nullptr;
static int (*get_error)(const void *, int) = nullptr;
static int (*ssl_shutdown)(void *) = nullptr;
static void (*ssl_free)(void *) = nullptr;
static const char *(*verify_err_str)(long) = nullptr;  // libcrypto, for the message only
static std::once_flag once; static bool ok = false; static std::string why;
static void *ctx = nullptr;
static std::string ca_file; static bool insecure = false;   // --ca-file, --tls-insecure
static const int VERIFY_NONE = 0, VERIFY_PEER = 1, CTRL_SET_TLSEXT_HOSTNAME = 55, ERROR_WANT_READ = 2, ERROR_SYSCALL = 5;
static std::string exe_dir() {
#ifdef _WIN32
    char b[MAX_PATH]; DWORD n = GetModuleFileNameA(nullptr, b, sizeof b); std::string p(b, n);
#else
    char b[PATH_MAX]; ssize_t n = readlink("/proc/self/exe", b, sizeof b - 1); std::string p(b, n > 0 ? (size_t)n : 0);
#endif
    size_t sl = p.find_last_of("/\\"); return sl == std::string::npos ? "." : p.substr(0, sl);
}
static bool load() {
    std::call_once(once, [] {
#ifdef _WIN32
        std::string d = exe_dir() + "\\";
        HMODULE c = LoadLibraryA((d + "libcrypto-3-x64.dll").c_str()); if (!c) c = LoadLibraryA("libcrypto-3-x64.dll");
        HMODULE h = LoadLibraryA((d + "libssl-3-x64.dll").c_str()); if (!h) h = LoadLibraryA("libssl-3-x64.dll");
        if (!h) h = LoadLibraryA("libssl-1_1-x64.dll");
        if (!c) c = LoadLibraryA("libcrypto-1_1-x64.dll");
        auto sym = [&](HMODULE m, const char *n) -> void * { return m ? (void *)GetProcAddress(m, n) : nullptr; };
#else
        void *h = nullptr, *c = nullptr;
        for (const char *n : {"libssl.so.3", "libssl.so.1.1", "libssl.so"}) if ((h = dlopen(n, RTLD_NOW))) break;
        for (const char *n : {"libcrypto.so.3", "libcrypto.so.1.1", "libcrypto.so"}) if ((c = dlopen(n, RTLD_NOW))) break;
        auto sym = [&](void *m, const char *n) -> void * { return m ? dlsym(m, n) : nullptr; };
#endif
        if (!h) { why = "stratum+ssl needs OpenSSL (libssl 3 or 1.1), which was not found"; return; }
        init_ssl = (decltype(init_ssl))sym(h, "OPENSSL_init_ssl");
        client_method = (decltype(client_method))sym(h, "TLS_client_method");
        ctx_new = (decltype(ctx_new))sym(h, "SSL_CTX_new");
        ctx_default_paths = (decltype(ctx_default_paths))sym(h, "SSL_CTX_set_default_verify_paths");
        ctx_load_locations = (decltype(ctx_load_locations))sym(h, "SSL_CTX_load_verify_locations");
        ctx_set_verify = (decltype(ctx_set_verify))sym(h, "SSL_CTX_set_verify");
        ssl_new = (decltype(ssl_new))sym(h, "SSL_new");
        set_fd = (decltype(set_fd))sym(h, "SSL_set_fd");
        ssl_ctrl = (decltype(ssl_ctrl))sym(h, "SSL_ctrl");
        set1_host = (decltype(set1_host))sym(h, "SSL_set1_host");
        ssl_connect = (decltype(ssl_connect))sym(h, "SSL_connect");
        verify_result = (decltype(verify_result))sym(h, "SSL_get_verify_result");
        ssl_read = (decltype(ssl_read))sym(h, "SSL_read");
        ssl_pending = (decltype(ssl_pending))sym(h, "SSL_pending");
        ssl_write = (decltype(ssl_write))sym(h, "SSL_write");
        get_error = (decltype(get_error))sym(h, "SSL_get_error");
        ssl_shutdown = (decltype(ssl_shutdown))sym(h, "SSL_shutdown");
        ssl_free = (decltype(ssl_free))sym(h, "SSL_free");
        verify_err_str = (decltype(verify_err_str))sym(c, "X509_verify_cert_error_string");
        if (!init_ssl || !client_method || !ctx_new || !ctx_default_paths || !ctx_load_locations || !ctx_set_verify || !ssl_new || !set_fd ||
            !ssl_ctrl || !set1_host || !ssl_connect || !verify_result || !ssl_read || !ssl_pending || !ssl_write || !get_error || !ssl_shutdown || !ssl_free) {
            why = "the OpenSSL library found lacks functions this miner needs (OpenSSL 1.1 or newer required)"; return;
        }
        init_ssl(0, nullptr);
        ctx = ctx_new(client_method());
        if (!ctx) { why = "OpenSSL: cannot create a TLS context"; return; }
        if (insecure) ctx_set_verify(ctx, VERIFY_NONE, nullptr);
        else {
            ctx_set_verify(ctx, VERIFY_PEER, nullptr);
            ctx_default_paths(ctx);
            std::string f = ca_file;
            if (f.empty()) { std::string g = exe_dir() + "/cacert.pem"; if (FILE *t = fopen(g.c_str(), "rb")) { fclose(t); f = g; } }
            if (!f.empty() && ctx_load_locations(ctx, f.c_str(), nullptr) != 1) { why = "cannot read the CA file " + f; return; }
        }
        ok = true;
    });
    return ok;
}
// the TLS session on a connected socket: handshake with SNI and the host-name check
static void *open(sock_t fd, const std::string &host, std::string &err) {
    if (!load()) { err = why; return nullptr; }
    void *s = ssl_new(ctx);
    if (!s) { err = "OpenSSL: cannot create a session"; return nullptr; }
    set_fd(s, (int)fd);
    ssl_ctrl(s, CTRL_SET_TLSEXT_HOSTNAME, 0, (void *)host.c_str());
    if (!insecure) set1_host(s, host.c_str());
    if (ssl_connect(s) != 1) {
        long v = verify_result(s);
        err = "TLS handshake with " + host + " failed";
        if (v != 0) err += std::string(": certificate not trusted (") + (verify_err_str ? verify_err_str(v) : std::to_string(v).c_str()) + "; --ca-file FILE or --tls-insecure for a private pool)";
        ssl_free(s); return nullptr;
    }
    return s;
}
static void close(void *s) { if (s) { ssl_shutdown(s); ssl_free(s); } }
}  // namespace tls
struct HttpResp { int status = 0; std::string body; std::string err; };
static HttpResp http_do(const Url &u, const std::string &method, const std::string &path, const std::string &body, const std::string &token, int timeout_s) {
    HttpResp r;
    struct addrinfo hints = {}, *res = nullptr;
    hints.ai_family = AF_UNSPEC; hints.ai_socktype = SOCK_STREAM;
    char portstr[16]; snprintf(portstr, sizeof portstr, "%d", u.port);
    if (getaddrinfo(u.host.c_str(), portstr, &hints, &res) != 0) { r.err = "dns failure for " + u.host; return r; }
    sock_t fd = SOCK_INVALID;
    for (auto *ai = res; ai; ai = ai->ai_next) {
        fd = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (fd == SOCK_INVALID) continue;
        sock_timeout(fd, timeout_s);
        int one = 1; setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, (const char *)&one, sizeof one);
        if (connect(fd, ai->ai_addr, (int)ai->ai_addrlen) == 0) break;
        sock_close(fd); fd = SOCK_INVALID;
    }
    freeaddrinfo(res);
    if (fd == SOCK_INVALID) { r.err = "cannot connect to " + u.host + ":" + portstr; return r; }
    std::string req = method + " " + path + " HTTP/1.1\r\nHost: " + u.host + "\r\nConnection: close\r\nUser-Agent: ProxiChorizo/" PX_VERSION "\r\n";
    if (!token.empty()) req += "Authorization: Bearer " + token + "\r\n";
    if (!body.empty() || method == "POST") req += "Content-Type: application/json\r\nContent-Length: " + std::to_string(body.size()) + "\r\n";
    req += "\r\n" + body;
    size_t sent = 0;
    while (sent < req.size()) { ssize_t n = send(fd, req.data() + sent, (int)(req.size() - sent), MSG_NOSIGNAL); if (n <= 0) { sock_close(fd); r.err = "send failed"; return r; } sent += n; }
    std::string resp; char buf[8192];
    for (;;) { ssize_t n = recv(fd, buf, (int)sizeof buf, 0); if (n < 0) { sock_close(fd); r.err = "recv failed/timeout"; return r; } if (n == 0) break; resp.append(buf, n); }
    sock_close(fd);
    size_t hdr_end = resp.find("\r\n\r\n");
    if (hdr_end == std::string::npos) { r.err = "malformed response"; return r; }
    std::string head = resp.substr(0, hdr_end);
    std::string payload = resp.substr(hdr_end + 4);
    if (head.size() > 12) r.status = atoi(head.c_str() + 9);
    std::string lhead = head; for (auto &c : lhead) c = tolower(c);
    if (lhead.find("transfer-encoding: chunked") != std::string::npos) {
        std::string out; size_t p = 0;
        for (;;) {
            size_t e = payload.find("\r\n", p); if (e == std::string::npos) break;
            size_t len = strtoul(payload.substr(p, e - p).c_str(), nullptr, 16);
            if (len == 0) break;
            out += payload.substr(e + 2, len); p = e + 2 + len + 2;
        }
        payload = out;
    }
    r.body = payload;
    return r;
}

// --- job state shared between poller and GPU workers ---
struct Job {
    uint64_t id = 0;
    uint64_t slot = 0;
    uint64_t nonce_base = 0;  // start of this worker's nonce range, if the job server assigns one
    bool has_base = false;
    std::vector<uint8_t> prefix;
    int k = 0;
    bool has_beat = false;
    uint8_t beat[64];
    std::chrono::steady_clock::time_point deadline;
    bool stale = false;
};
// ---- GPU sensors: temperature, fan, power ------------------------------------
// Read where the driver offers them: NVML loaded at run time (NVIDIA, Linux and
// Windows; no build dependency) or the kernel's hwmon files under the device's
// PCI node (AMD on Linux). A GPU without either simply reports no sensors.
struct GpuSensors { int temp = -1; int fan = -1; double power = -1; int mem_temp = -1; int clock = -1; };  // degrees C, percent, watts, degrees C, MHz; -1 = unknown
#ifndef _WIN32
#include <dlfcn.h>
#include <dirent.h>
#endif
namespace nvml {
typedef void *Dev;
// nvmlFieldValue_t as laid out by NVML (40 bytes): memory temperature is field 83
struct FieldValue { unsigned fieldId; unsigned scopeId; long long timestamp; long long latencyUsec; int valueType; int nvmlReturn; union { double d; unsigned u; unsigned long long ull; long long ll; } value; };
static int (*Init)() = nullptr;
static int (*ByBus)(const char *, Dev *) = nullptr;
static int (*Temp)(Dev, int, unsigned *) = nullptr;
static int (*Fan)(Dev, unsigned *) = nullptr;
static int (*Power)(Dev, unsigned *) = nullptr;
static int (*Clock)(Dev, int, unsigned *) = nullptr;
static int (*Fields)(Dev, int, FieldValue *) = nullptr;
static std::once_flag once; static bool ok = false;
static void load() {
#ifdef _WIN32
    HMODULE h = LoadLibraryA("nvml.dll");
    if (!h) h = LoadLibraryA("C:\\Program Files\\NVIDIA Corporation\\NVSMI\\nvml.dll");
    if (!h) return;
    auto sym = [&](const char *n) { return (void *)GetProcAddress(h, n); };
#else
    void *h = dlopen("libnvidia-ml.so.1", RTLD_NOW); if (!h) h = dlopen("libnvidia-ml.so", RTLD_NOW);
    if (!h) return;
    auto sym = [&](const char *n) { return dlsym(h, n); };
#endif
    Init = (int (*)())sym("nvmlInit_v2");
    ByBus = (int (*)(const char *, Dev *))sym("nvmlDeviceGetHandleByPciBusId_v2");
    Temp = (int (*)(Dev, int, unsigned *))sym("nvmlDeviceGetTemperature");
    Fan = (int (*)(Dev, unsigned *))sym("nvmlDeviceGetFanSpeed");
    Power = (int (*)(Dev, unsigned *))sym("nvmlDeviceGetPowerUsage");
    Clock = (int (*)(Dev, int, unsigned *))sym("nvmlDeviceGetClockInfo");
    Fields = (int (*)(Dev, int, FieldValue *))sym("nvmlDeviceGetFieldValues");
    ok = Init && ByBus && Temp && Init() == 0;
}
static bool read(const std::string &bus, GpuSensors &s) {
    std::call_once(once, load);
    if (!ok) return false;
    Dev d; if (ByBus(bus.c_str(), &d) != 0) return false;
    unsigned v;
    if (Temp(d, 0 /* NVML_TEMPERATURE_GPU */, &v) == 0) s.temp = (int)v;
    if (Fan && Fan(d, &v) == 0) s.fan = (int)v;           // fails on passive (server) cards: stays -1
    if (Power && Power(d, &v) == 0) s.power = v / 1000.0;  // milliwatts
    if (Clock && Clock(d, 1 /* NVML_CLOCK_SM */, &v) == 0) s.clock = (int)v;
    if (Fields) {  // memory temperature: HBM/GDDR6X cards report it, most consumer cards do not (stays -1)
        FieldValue f; memset(&f, 0, sizeof f); f.fieldId = 83 /* NVML_FI_DEV_MEMORY_TEMP */;
        if (Fields(d, 1, &f) == 0 && f.nvmlReturn == 0 && f.value.u > 0 && f.value.u < 200) s.mem_temp = (int)f.value.u;
    }
    return s.temp >= 0 || s.power >= 0;
}
}
static bool sysfs_sensors(const std::string &bus, GpuSensors &s) {
#ifdef _WIN32
    (void)bus; (void)s; return false;
#else
    std::string dir = "/sys/bus/pci/devices/" + bus + "/hwmon", hw;
    if (DIR *d = opendir(dir.c_str())) { while (dirent *e = readdir(d)) if (e->d_name[0] != '.') { hw = dir + "/" + e->d_name; break; } closedir(d); }
    if (hw.empty()) return false;
    auto rd = [&](const char *f, double &v) { std::ifstream in(hw + "/" + f); return (bool)(in >> v); };
    double v, m;
    if (rd("temp1_input", v)) s.temp = (int)(v / 1000);
    if (rd("fan1_input", v) && rd("fan1_max", m) && m > 0) s.fan = (int)(v * 100 / m + 0.5);
    else if (rd("pwm1", v)) s.fan = (int)(v / 2.55 + 0.5);
    if (rd("power1_average", v) || rd("power1_input", v)) s.power = v / 1e6;  // microwatts
    if (rd("temp3_input", v)) s.mem_temp = (int)(v / 1000);                     // amdgpu: temp1 edge, temp2 junction, temp3 memory
    if (rd("freq1_input", v)) s.clock = (int)(v / 1e6);                         // Hz
    return s.temp >= 0 || s.power >= 0;
#endif
}
static GpuSensors read_sensors(const std::string &bus) { GpuSensors s; if (!nvml::read(bus, s)) sysfs_sensors(bus, s); return s; }
// "65C, 533 W, 2422 MHz, fan n/a" (memory temperature as "65C/72C mem" where reported) for the per-GPU line
static std::string sensors_str(const GpuSensors &s) {
    char b[96]; std::string o;
    if (s.temp >= 0) { snprintf(b, sizeof b, "%dC", s.temp); o += b; if (s.mem_temp >= 0) { snprintf(b, sizeof b, "/%dC mem", s.mem_temp); o += b; } }
    else o += "temp n/a";
    if (s.power >= 0) { snprintf(b, sizeof b, ", %.0f W", s.power); o += b; } else o += ", power n/a";
    if (s.clock >= 0) { snprintf(b, sizeof b, ", %d MHz", s.clock); o += b; }
    if (s.fan >= 0) { snprintf(b, sizeof b, ", fan %d%%", s.fan); o += b; } else o += ", fan n/a";
    return o;
}
static std::string sensors_json(const GpuSensors &s) {
    auto num = [](double v, bool f) { if (v < 0) return std::string("null"); char b[32]; snprintf(b, sizeof b, f ? "%.1f" : "%.0f", v); return std::string(b); };
    return ",\"temp\":" + num(s.temp, false) + ",\"fan\":" + num(s.fan, false) + ",\"power\":" + num(s.power, true) +
           ",\"mem_temp\":" + num(s.mem_temp, false) + ",\"clock\":" + num(s.clock, false);
}

struct Shared {
    std::mutex mu;
    Job job;                           // current job (id 0 = none)
    std::atomic<uint64_t> attempts{0}; // since last report
    std::atomic<uint64_t> solutions{0};
    std::atomic<uint64_t> shares{0};   // pool shares accepted (results answered without a beat)
    std::atomic<uint64_t> rejected{0}; // answered as not accepted (422 or accepted:false)
    std::atomic<uint64_t> stale{0};    // 409: the pool had moved on to the next job
    std::atomic<bool> stop{false};
    std::string base_url; Url url; std::string token; std::string name;
    uint8_t pk[32];
    uint8_t scalar[32];
    // one entry per selected GPU: for the console lines and the stats API (--api-bind), rate in H/s.
    // The worker adds to the win_* fields after every launch; the console thread turns them into
    // the rate of the last interval. paused/dead/reason carry the thermal pause and device failures.
    struct GpuStat {
        int dev; int ordinal; std::string name; std::string bus; int bus_num; double rate; GpuSensors sens;
        std::string arch, variant, geometry;   // for the banner: "sm_120", kernel variant, "23552x256x1"
        uint64_t win_attempts = 0; double win_kernel_ms = 0, win_idle_ms = 0; int win_launches = 0;
        uint64_t accepted = 0, rejected = 0, stale = 0;
        bool paused = false, dead = false; std::string reason;
        std::chrono::steady_clock::time_point sens_at{}, paused_at{};
    };
    std::vector<GpuStat> gpus;
    std::chrono::steady_clock::time_point started = std::chrono::steady_clock::now();
    std::string api_bind;
    ConsoleOpts con;                   // thermal limits and console options
    uint64_t last_job_id = 0, last_job_k = 0; std::chrono::steady_clock::time_point last_job_at{};  // for the console's total line
    // stratum (stratum+tcp:// URL): the connection, its extranonce1 and the shares awaiting an answer
    std::string user;                  // authorized as: "<address>.<worker>" (--user) or the worker name
    std::mutex wmu;                    // guards sfd for writers and the fields below
    sock_t sfd = SOCK_INVALID;
    void *ssl = nullptr;               // the TLS session for stratum+ssl:// (reads and writes under wmu)
    uint16_t en1 = 0;
    bool subscribed = false;
    uint64_t next_rpc = 100;
    struct Pending { int ordinal; uint64_t job; };
    std::map<uint64_t, Pending> pending;
};
// console wording for one GPU: "GPU #3" by its device index (nvidia-smi order under PCI_BUS_ID)
static std::string gpu_tag(const Shared::GpuStat &g) { return "GPU #" + std::to_string(g.dev); }

static void install_job(Shared &sh, const std::string &body) {
    uint64_t id = 0; json_u64(body, "id", id);
    std::lock_guard<std::mutex> l(sh.mu);
    if (id == 0) {
        if (sh.job.id != 0) logln("no job (miner idle)");
        sh.job = Job(); return;
    }
    std::string pkh, prefixh, beath; uint64_t k = 0, ttl = 0, slot = 0;
    json_str(body, "pubkey", pkh); json_str(body, "alpha_prefix", prefixh); json_str(body, "beat", beath);
    json_u64(body, "k", k); json_u64(body, "ttl_ms", ttl); json_u64(body, "slot", slot);
    if (pkh != hexs(sh.pk, 32)) { logln("the miner mines under key " + pkh + " but this seeker holds " + hexs(sh.pk, 32) + " — refusing the job"); sh.job = Job(); return; }
    Job j; j.id = id; j.slot = slot; j.k = (int)k;
    // a pool hands every worker its own nonce range (top 16 bits); without one
    // (proxi node mine) each GPU starts at a random nonce
    std::string nbh; std::vector<uint8_t> nbv;
    if (json_str(body, "nonce_base", nbh) && unhex(nbh, nbv) && nbv.size() == 8) {
        for (int i = 0; i < 8; i++) j.nonce_base = (j.nonce_base << 8) | nbv[i];
        j.has_base = true;
    }
    if (!unhex(prefixh, j.prefix) || j.prefix.empty() || j.prefix.size() > 64) { logln("bad alpha_prefix in job"); sh.job = Job(); return; }
    std::vector<uint8_t> b;
    if (!beath.empty() && unhex(beath, b) && b.size() == 64) { j.has_beat = true; memcpy(j.beat, b.data(), 64); }
    else { j.has_beat = false; memset(j.beat, 0xff, 64); }
    j.deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(ttl);
    bool changed = (sh.job.id != id);
    sh.job = j; sh.last_job_id = id; sh.last_job_k = k; sh.last_job_at = std::chrono::steady_clock::now();
    if (changed) logln("job " + std::to_string(id) + ": slot " + std::to_string(slot) + ", k=" + std::to_string(k) + ", ttl " + std::to_string(ttl) + " ms" + (j.has_beat ? ", beat " + beath.substr(0, 16) + "…" : ""));
}

static std::string urlencode(const std::string &s) {
    std::string o; char b[4];
    for (unsigned char c : s) {
        if (isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') o += (char)c;
        else { snprintf(b, sizeof b, "%%%02X", c); o += b; }
    }
    return o;
}

static void poller(Shared &sh) {
    uint64_t after = 0; int backoff = 1;
    while (!sh.stop) {
        std::string path = "/seeker/job?after=" + std::to_string(after) + "&wait=5000&worker=" + urlencode(sh.name);
        HttpResp r = http_do(sh.url, "GET", path, "", sh.token, 30);
        if (!r.err.empty() || r.status != 200) {
            logln("cannot fetch the job: " + (r.err.empty() ? "HTTP " + std::to_string(r.status) + " " + r.body.substr(0, 120) : r.err) + " (retry in " + std::to_string(backoff) + "s)");
            std::this_thread::sleep_for(std::chrono::seconds(backoff)); backoff = std::min(backoff * 2, 30);
            continue;
        }
        backoff = 1;
        install_job(sh, r.body);
        { std::lock_guard<std::mutex> l(sh.mu); after = sh.job.id; }
        // A pool retires its job for a few milliseconds at every round boundary
        // (about once per slot). Asking again at once with after=0 long-polls
        // until the next job is published; the short pause only guards against
        // a server that answers an empty job immediately instead of waiting.
        if (after == 0) std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
}

// the pool's hashrate accounting: attempts since the last report, once a second
static void reporter(Shared &sh) {
    uint64_t carried = 0;
    while (!sh.stop) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        uint64_t n = sh.attempts.exchange(0) + carried;
        uint64_t id; { std::lock_guard<std::mutex> l(sh.mu); id = sh.job.id; }
        std::string body = "{\"id\":" + std::to_string(id) + ",\"attempts\":" + std::to_string(n) + ",\"seeker\":\"" + sh.name + "\"}";
        HttpResp r = http_do(sh.url, "POST", "/seeker/report", body, sh.token, 10);
        carried = (r.err.empty() && r.status / 100 == 2) ? 0 : n;
    }
}

// "GPU #0: RTX PRO 6000 Blackwell Server Edition [1D:00.0] 63.52 MH/s, 65C, 533 W, 2422 MHz, fan n/a"
static std::string gpu_line(const Shared::GpuStat &g, double secs, bool verbose) {
    char b[160]; std::string o = gpu_tag(g) + ": " + short_gpu_name(g.name) + " [" + short_bus(g.bus) + "] ";
    if (g.dead) return o + "DEAD (" + g.reason + ")";
    if (g.paused) { o += "PAUSED (" + g.reason + ", " + fmt_uptime((uint64_t)std::chrono::duration<double>(std::chrono::steady_clock::now() - g.paused_at).count()) + ") "; }
    else { snprintf(b, sizeof b, "%.2f MH/s, ", g.rate / 1e6); o += b; }
    o += sensors_str(g.sens);
    if (verbose && secs > 0) {
        snprintf(b, sizeof b, " | kernel %.0f%%, %d launches x %.1f ms, idle %.0f%%", g.win_kernel_ms / 10 / secs, g.win_launches, g.win_launches ? g.win_kernel_ms / g.win_launches : 0.0, g.win_idle_ms / 10 / secs);
        o += b;
    }
    return o;
}
// the periodic console block: one line per GPU, then the total with the share counters and the uptime
static void console(Shared &sh) {
    auto last = std::chrono::steady_clock::now();
    while (!sh.stop) {
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
        auto now = std::chrono::steady_clock::now();
        if (now - last < std::chrono::seconds(sh.con.logInterval)) continue;
        double secs = std::chrono::duration<double>(now - last).count(); last = now;
        std::vector<Shared::GpuStat> snap; uint64_t id, k; int alive = 0;
        {
            std::lock_guard<std::mutex> l(sh.mu);
            for (auto &g : sh.gpus) {
                g.rate = g.dead ? 0 : g.win_attempts / secs;
                snap.push_back(g);
                g.win_attempts = 0; g.win_kernel_ms = g.win_idle_ms = 0; g.win_launches = 0;
                if (!g.dead) alive++;
            }
            id = sh.job.id; k = sh.job.k;
            // a pool retires its job for a few milliseconds at every round boundary: a snapshot
            // inside that gap is not "no job"; say so only after 10 s without one
            if (id == 0 && sh.last_job_id && now - sh.last_job_at < std::chrono::seconds(10)) { id = sh.last_job_id; k = sh.last_job_k; }
        }
        double total = 0;
        for (auto &g : snap) {
            total += g.rate;
            if (sh.con.quiet) continue;
            if (g.dead) log_err(gpu_line(g, secs, sh.con.verbose));
            else if (g.paused) log_warn(gpu_line(g, secs, sh.con.verbose));
            else logln(gpu_line(g, secs, sh.con.verbose));
        }
        char b[256];
        snprintf(b, sizeof b, "Total: %.2f MH/s on %d/%zu GPUs | accepted %llu, rejected %llu, stale %llu", total / 1e6, alive, snap.size(),
                 (unsigned long long)sh.shares.load(), (unsigned long long)sh.rejected.load(), (unsigned long long)sh.stale.load());
        std::string line = b;
        if (sh.solutions.load()) line += ", solutions " + std::to_string(sh.solutions.load());
        line += id ? " | job " + std::to_string(id) + " k=" + std::to_string(k) : " | no job";
        line += " | uptime " + fmt_uptime(std::chrono::duration_cast<std::chrono::seconds>(now - sh.started).count());
        if (id == 0) log_warn(line); else logc(C_BOLD, line);
    }
}

// Thermal gate, called by the worker before every launch: refreshes this GPU's
// sensors every 2 s and pauses only this GPU at --max-temp (core) or --max-mem-temp
// (memory), resuming once it has cooled 8 C below the limit. Returns false while the
// GPU must not launch. A GPU without a readable temperature is never paused.
static bool thermal_gate(Shared &sh, int ordinal) {
    auto now = std::chrono::steady_clock::now();
    std::string bus;
    {
        std::lock_guard<std::mutex> l(sh.mu);
        if ((size_t)ordinal >= sh.gpus.size()) return true;
        auto &g = sh.gpus[ordinal];
        if (now - g.sens_at < std::chrono::seconds(2)) return !g.paused;
        bus = g.bus;
    }
    GpuSensors se = read_sensors(bus);
    // test hook: PX_TEST_TEMP_OFFSET=<device>:<degrees> makes one GPU read that much hotter than it is
    if (const char *e = getenv("PX_TEST_TEMP_OFFSET")) { int d = -1, off = 0; std::lock_guard<std::mutex> l(sh.mu); if (sscanf(e, "%d:%d", &d, &off) == 2 && d == sh.gpus[ordinal].dev && se.temp >= 0) se.temp += off; }
    std::string msg; bool warn = false, paused;
    {
        std::lock_guard<std::mutex> l(sh.mu);
        auto &g = sh.gpus[ordinal];
        g.sens = se; g.sens_at = now;
        int lim = sh.con.maxTemp, mlim = sh.con.maxMemTemp, d = 8;
        bool hot = (lim > 0 && se.temp >= lim) || (mlim > 0 && se.mem_temp >= mlim);
        bool cool = (lim <= 0 || se.temp < 0 || se.temp <= lim - d) && (mlim <= 0 || se.mem_temp < 0 || se.mem_temp <= mlim - d);
        if (!g.paused && hot) {
            g.paused = true; g.reason = "thermal"; g.paused_at = now; warn = true;
            char b[160];
            if (lim > 0 && se.temp >= lim) snprintf(b, sizeof b, "%dC core reaches the limit of %dC: paused, resumes below %dC (the other GPUs keep mining)", se.temp, lim, lim - d);
            else snprintf(b, sizeof b, "%dC memory reaches the limit of %dC: paused, resumes below %dC (the other GPUs keep mining)", se.mem_temp, mlim, mlim - d);
            msg = gpu_tag(g) + ": " + b;
        } else if (g.paused && g.reason == "thermal" && cool && now - g.paused_at >= std::chrono::seconds(30)) {
            // a paused card cools several degrees within seconds and would flap every sensor
            // interval: stay paused at least 30 s so the pause is a duty cycle, not a flutter
            g.paused = false; g.reason.clear();
            char b[120]; snprintf(b, sizeof b, "%dC: cooled down after %s, mining again", se.temp, fmt_uptime((uint64_t)std::chrono::duration<double>(now - g.paused_at).count()).c_str());
            msg = gpu_tag(g) + ": " + b;
        }
        paused = g.paused;
    }
    if (!msg.empty()) { if (warn) log_warn(msg); else log_ok(msg); }
    return !paused;
}
// a device error on one GPU: that GPU is marked dead and its worker ends, the others keep
// mining; the process exits (code 2, for a restart wrapper) only when no GPU is left
static void gpu_failed(Shared &sh, int ordinal, const std::string &what) {
    int alive = 0; std::string tag;
    {
        std::lock_guard<std::mutex> l(sh.mu);
        if ((size_t)ordinal < sh.gpus.size()) { auto &g = sh.gpus[ordinal]; g.dead = true; g.paused = false; g.reason = "device error"; g.rate = 0; tag = gpu_tag(g); }
        for (auto &g : sh.gpus) if (!g.dead) alive++;
    }
    log_err(tag + ": " + what + " — this GPU is disabled" + (alive ? ", mining continues on " + std::to_string(alive) + " GPU(s)" : ""));
    if (!alive) { log_err("no working GPU left, exiting"); fflush(stderr); exit(2); }
}
// start-up banner (ccminer style): version, licence, pool, worker, key, the GPUs, thermal limits, API
static void print_banner(Shared &sh, const char *backend) {
    bannerln("");
    bannerln(std::string(g_color ? C_BOLD C_CYAN : "") + "  *** ProxiChorizo " PX_VERSION " (" + backend + ") — nonce seeker for Proxima by suprnova.cc ***" + (g_color ? C_RESET : ""));
    bannerln("      ProxiChorizo License 1.0: free to run for mining; commercial or closed-source reuse needs written permission");
    if (sh.url.stratum) bannerln(std::string("      pool:    ") + (sh.url.tls ? "stratum+ssl://" : "stratum+tcp://") + sh.url.host + ":" + std::to_string(sh.url.port) + "   user: " + sh.user);
    else bannerln("      pool:    http://" + sh.url.host + ":" + std::to_string(sh.url.port) + "   worker: " + sh.name);
    bannerln("      key:     " + hexs(sh.pk, 32));
    bannerln("      GPUs:    " + std::to_string(sh.gpus.size()) + " selected (GPU numbers follow the PCI bus order, as nvidia-smi lists them)");
    for (auto &g : sh.gpus) bannerln("      " + gpu_tag(g) + ": " + short_gpu_name(g.name) + " [" + short_bus(g.bus) + "] " + g.arch + ", " + g.variant + " " + g.geometry);
    char b[160];
    if (sh.con.maxTemp > 0 || sh.con.maxMemTemp > 0) snprintf(b, sizeof b, "      thermal: pause a GPU at %dC core / %dC memory, resume 8C lower (--max-temp, --max-mem-temp; 0 = off)", sh.con.maxTemp, sh.con.maxMemTemp);
    else snprintf(b, sizeof b, "      thermal: limits off (--max-temp 0)");
    bannerln(b);
    if (!sh.api_bind.empty()) bannerln("      stats:   http://" + (sh.api_bind.find(':') == std::string::npos ? "127.0.0.1:" + sh.api_bind : sh.api_bind) + "/  (JSON; /mmpos for the mmpOS line)");
    bannerln("");
}


// ---- stratum client ----------------------------------------------------------
// The pool's stratum port (NOMP-style stratum v1, newline-delimited JSON-RPC over one
// persistent TCP connection):
//   -> mining.subscribe ["ProxiChorizo/<version>"]   <- [[...], "<extranonce1, 2 bytes hex>", 6]
//   -> mining.authorize ["<address>.<worker>" or "<worker>", "<token>"]   <- true
//   <- mining.set_difficulty [<2^k>]: applies from the next job, as the pool queues it
//   <- mining.notify ["<job id hex>", "<pubkey>", "<pred>", <slot>, "<alpha prefix>", <ttl ms>, <clean_jobs>]
//   -> mining.submit ["<user>", "<job id>", "<extranonce2, 6 bytes hex>", "<beta>"]   <- true or [code, msg, null]
// The nonce is extranonce1 || extranonce2: the connection's own 16 bits on top, then the
// GPU ordinal (8 bits) and 40 bits of counter per GPU and job, as with the HTTP nonce_base.
// Answers: 21 job not found (stale), 22 duplicate, 23 low difficulty, 24 unauthorized, 25 not subscribed.

// a JSON value: just enough of a parser for the stratum messages
struct JV {
    enum T { NUL, BOOL, NUM, STR, ARR, OBJ } t = NUL;
    bool b = false; double n = 0; std::string s;
    std::vector<JV> a; std::vector<std::pair<std::string, JV>> o;
    const JV *get(const char *k) const { for (auto &p : o) if (p.first == k) return &p.second; return nullptr; }
    const JV *at(size_t i) const { return t == ARR && i < a.size() ? &a[i] : nullptr; }
};
static void jws(const std::string &s, size_t &p) { while (p < s.size() && (s[p] == ' ' || s[p] == '\t' || s[p] == '\r' || s[p] == '\n')) p++; }
static bool jparse(const std::string &s, size_t &p, JV &v, int depth = 0) {
    if (depth > 16) return false;
    jws(s, p); if (p >= s.size()) return false;
    char c = s[p];
    if (c == '{') {
        v.t = JV::OBJ; p++; jws(s, p);
        if (p < s.size() && s[p] == '}') { p++; return true; }
        for (;;) {
            JV key; if (!jparse(s, p, key, depth + 1) || key.t != JV::STR) return false;
            jws(s, p); if (p >= s.size() || s[p] != ':') return false; p++;
            JV val; if (!jparse(s, p, val, depth + 1)) return false;
            v.o.emplace_back(key.s, std::move(val));
            jws(s, p); if (p >= s.size()) return false;
            if (s[p] == ',') { p++; continue; }
            if (s[p] == '}') { p++; return true; }
            return false;
        }
    }
    if (c == '[') {
        v.t = JV::ARR; p++; jws(s, p);
        if (p < s.size() && s[p] == ']') { p++; return true; }
        for (;;) {
            JV val; if (!jparse(s, p, val, depth + 1)) return false;
            v.a.push_back(std::move(val));
            jws(s, p); if (p >= s.size()) return false;
            if (s[p] == ',') { p++; continue; }
            if (s[p] == ']') { p++; return true; }
            return false;
        }
    }
    if (c == '"') {
        v.t = JV::STR; p++;
        while (p < s.size() && s[p] != '"') {
            if (s[p] == '\\' && p + 1 < s.size()) {
                char e = s[p + 1]; p += 2;
                if (e == 'n') v.s += '\n'; else if (e == 't') v.s += '\t'; else if (e == 'r') v.s += '\r';
                else if (e == 'u') { v.s += '?'; p += 4; } else v.s += e;
            } else v.s += s[p++];
        }
        if (p >= s.size()) return false;
        p++; return true;
    }
    if (s.compare(p, 4, "true") == 0) { v.t = JV::BOOL; v.b = true; p += 4; return true; }
    if (s.compare(p, 5, "false") == 0) { v.t = JV::BOOL; v.b = false; p += 5; return true; }
    if (s.compare(p, 4, "null") == 0) { v.t = JV::NUL; p += 4; return true; }
    const char *b = s.c_str() + p; char *e = nullptr;
    double d = strtod(b, &e);
    if (e == b) return false;
    v.t = JV::NUM; v.n = d; p += (size_t)(e - b); return true;
}
// a JSON string literal (quotes and backslashes escaped, control characters dropped)
static std::string jstr(const std::string &v) {
    std::string o = "\"";
    for (char c : v) { if (c == '"' || c == '\\') o += '\\'; if ((unsigned char)c >= 0x20) o += c; }
    return o + "\"";
}

// one line to the pool; false when the connection is gone
static bool stratum_send(Shared &sh, const std::string &line) {
    std::lock_guard<std::mutex> l(sh.wmu);
    if (sh.sfd == SOCK_INVALID) return false;
    std::string m = line + "\n"; size_t sent = 0;
    while (sent < m.size()) {
        ssize_t n = sh.ssl ? tls::ssl_write(sh.ssl, m.data() + sent, (int)(m.size() - sent))
                           : send(sh.sfd, m.data() + sent, (int)(m.size() - sent), MSG_NOSIGNAL);
        if (n <= 0) return false;
        sent += (size_t)n;
    }
    return true;
}

// a share found by a GPU: nonce = extranonce1 || extranonce2
static void stratum_submit(Shared &sh, const Job &j, uint64_t nonce, const uint8_t beta[64], int ordinal) {
    std::string tag; { std::lock_guard<std::mutex> l(sh.mu); tag = (size_t)ordinal < sh.gpus.size() ? gpu_tag(sh.gpus[ordinal]) : "GPU"; }
    uint64_t id; uint16_t en1;
    {
        std::lock_guard<std::mutex> l(sh.wmu);
        if (sh.sfd == SOCK_INVALID || !sh.subscribed) { log_warn(tag + ": share of job " + std::to_string(j.id) + " dropped, not connected"); return; }
        en1 = sh.en1; id = sh.next_rpc++;
        sh.pending[id] = Shared::Pending{ordinal, j.id};
    }
    if ((nonce >> 48) != en1) { log_err(tag + ": nonce outside this connection's range, not submitted"); std::lock_guard<std::mutex> l(sh.wmu); sh.pending.erase(id); return; }
    char e2[16], jid[24];
    snprintf(e2, sizeof e2, "%012llx", (unsigned long long)(nonce & 0xffffffffffffULL));
    snprintf(jid, sizeof jid, "%llx", (unsigned long long)j.id);
    std::string line = "{\"id\":" + std::to_string(id) + ",\"method\":\"mining.submit\",\"params\":[" + jstr(sh.user) + "," + jstr(jid) + "," + jstr(e2) + "," + jstr(hexs(beta, 64)) + "]}";
    if (!stratum_send(sh, line)) { std::lock_guard<std::mutex> l(sh.wmu); sh.pending.erase(id); log_warn(tag + ": share of job " + std::to_string(j.id) + " not sent, connection lost"); }
}

// the pool's answer to a submit: count and log it like the HTTP answers
static void stratum_share_answer(Shared &sh, const Shared::Pending &pd, const JV &msg) {
    std::string tag; { std::lock_guard<std::mutex> l(sh.mu); tag = (size_t)pd.ordinal < sh.gpus.size() ? gpu_tag(sh.gpus[pd.ordinal]) : "GPU"; }
    const JV *res = msg.get("result"), *err = msg.get("error");
    int code = 0; std::string why;
    if (err && err->t == JV::ARR) { if (const JV *c = err->at(0)) code = (int)c->n; if (const JV *m = err->at(1)) why = m->s; }
    bool ok = res && res->t == JV::BOOL && res->b && code == 0;
    int which = ok ? 0 : code == 21 ? 2 : 1;  // 0 accepted, 1 rejected, 2 stale
    {
        std::lock_guard<std::mutex> l(sh.mu);
        if (which == 0) sh.shares++; else if (which == 1) sh.rejected++; else sh.stale++;
        if ((size_t)pd.ordinal < sh.gpus.size()) { auto &g = sh.gpus[pd.ordinal]; if (which == 0) g.accepted++; else if (which == 1) g.rejected++; else g.stale++; }
        if (which == 2 && sh.job.id == pd.job) sh.job.stale = true;  // the pool has moved on: idle until the next notify
    }
    char t[64]; snprintf(t, sizeof t, " (%llu/%llu)", (unsigned long long)sh.shares.load(), (unsigned long long)(sh.shares.load() + sh.rejected.load() + sh.stale.load()));
    if (which == 0) { if (!sh.con.quiet) log_ok(tag + ": share accepted" + t); }
    else if (which == 2) log_warn(tag + ": share stale, the pool had moved on from job " + std::to_string(pd.job) + t);
    else log_err(tag + ": share REJECTED (" + std::to_string(code) + " " + why + ") of job " + std::to_string(pd.job) + t);
}

// mining.notify: the job, under the difficulty last set
static void stratum_notify(Shared &sh, const JV &params, int &cur_k, int &pending_k) {
    const JV *jid = params.at(0), *pk = params.at(1), *slot = params.at(3), *alpha = params.at(4), *ttl = params.at(5);
    if (!jid || jid->t != JV::STR || !pk || !slot || !alpha || !ttl) { log_err("stratum: malformed mining.notify"); return; }
    if (pending_k >= 0) { cur_k = pending_k; pending_k = -1; }
    uint64_t id = strtoull(jid->s.c_str(), nullptr, 16);
    if (id == 0 || cur_k <= 0) { log_err("stratum: mining.notify without a job id or before mining.set_difficulty"); return; }
    if (pk->s != hexs(sh.pk, 32)) { log_err("the pool sends jobs for key " + pk->s + " but this miner holds " + hexs(sh.pk, 32) + " — check --key-file and --token"); std::lock_guard<std::mutex> l(sh.mu); sh.job = Job(); return; }
    Job j; j.id = id; j.slot = (uint64_t)slot->n; j.k = cur_k;
    if (!unhex(alpha->s, j.prefix) || j.prefix.empty() || j.prefix.size() > 64) { log_err("stratum: bad alpha prefix in mining.notify"); return; }
    uint16_t en1; { std::lock_guard<std::mutex> l(sh.wmu); en1 = sh.en1; }
    j.nonce_base = (uint64_t)en1 << 48; j.has_base = true;
    j.has_beat = false; memset(j.beat, 0xff, 64);
    j.deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds((int64_t)ttl->n);
    std::lock_guard<std::mutex> l(sh.mu);
    bool changed = sh.job.id != id;
    sh.job = j; sh.last_job_id = id; sh.last_job_k = (uint64_t)cur_k; sh.last_job_at = std::chrono::steady_clock::now();
    if (changed && sh.con.verbose) logln("job " + jid->s + ": slot " + std::to_string(j.slot) + ", k=" + std::to_string(cur_k) + ", ttl " + std::to_string((int64_t)ttl->n) + " ms");
}

// the connection: subscribe, authorize, then jobs in and shares out until it breaks; reconnects with backoff
static void stratum_client(Shared &sh) {
    int backoff = 1;
    while (!sh.stop) {
        std::string err;
        sock_t fd = tcp_connect(sh.url, 10, err);
        if (fd == SOCK_INVALID) {
            log_warn("stratum: " + err + " (retry in " + std::to_string(backoff) + "s)");
            for (int i = 0; i < backoff * 4 && !sh.stop; i++) std::this_thread::sleep_for(std::chrono::milliseconds(250));
            backoff = std::min(backoff * 2, 30); continue;
        }
        int one = 1; setsockopt(fd, SOL_SOCKET, SO_KEEPALIVE, (const char *)&one, sizeof one);
        void *ssl = nullptr;
        if (sh.url.tls) {
            ssl = tls::open(fd, sh.url.host, err);
            if (!ssl) {
                log_err("stratum: " + err + " (retry in " + std::to_string(backoff) + "s)");
                sock_close(fd);
                for (int i = 0; i < backoff * 4 && !sh.stop; i++) std::this_thread::sleep_for(std::chrono::milliseconds(250));
                backoff = std::min(backoff * 2, 30); continue;
            }
        }
        // recv wakes every second to see sh.stop; a TLS read happens under wmu (an OpenSSL
        // session is not shared between threads) only once data is there, so it is short
        sock_timeout_ms(fd, ssl ? 200 : 1000);
        { std::lock_guard<std::mutex> l(sh.wmu); sh.sfd = fd; sh.ssl = ssl; sh.subscribed = false; sh.pending.clear(); }
        auto connected_at = std::chrono::steady_clock::now();
        logln(std::string("stratum: connected to ") + sh.url.host + ":" + std::to_string(sh.url.port) + (ssl ? " (TLS)" : ""));
        stratum_send(sh, "{\"id\":1,\"method\":\"mining.subscribe\",\"params\":[\"ProxiChorizo/" PX_VERSION "\"]}");
        stratum_send(sh, "{\"id\":2,\"method\":\"mining.authorize\",\"params\":[" + jstr(sh.user) + "," + jstr(sh.token) + "]}");
        int cur_k = 0, pending_k = -1; bool authFailed = false, reconnect = false;
        std::string buf; char rb[4096];
        auto last_rx = std::chrono::steady_clock::now();
        while (!sh.stop && !reconnect) {
            ssize_t n; bool timedOut = false;
            if (ssl) {
                // wait for data outside the lock: a read holding wmu would starve the GPU
                // threads, which need it to send their shares
                bool ready; { std::lock_guard<std::mutex> l(sh.wmu); ready = tls::ssl_pending(ssl) > 0; }
                if (!ready && !sock_readable(fd, 1000)) { n = -1; timedOut = true; }
                else {
                    std::lock_guard<std::mutex> l(sh.wmu);
                    int r = tls::ssl_read(ssl, rb, (int)sizeof rb);
                    if (r > 0) n = r;
                    else { int e = tls::get_error(ssl, r); n = -1; timedOut = e == tls::ERROR_WANT_READ || (e == tls::ERROR_SYSCALL && sock_timed_out()); }
                }
            } else { n = recv(fd, rb, (int)sizeof rb, 0); timedOut = n < 0 && sock_timed_out(); }
            if (timedOut) {
                // a pool sends a job about every slot; three minutes of silence is a dead connection
                if (std::chrono::steady_clock::now() - last_rx > std::chrono::minutes(3)) { log_warn("stratum: no message for 3 minutes"); break; }
                continue;
            }
            if (n <= 0) { log_warn("stratum: connection closed by the pool"); break; }
            last_rx = std::chrono::steady_clock::now();
            buf.append(rb, (size_t)n);
            if (buf.size() > 1 << 20) { log_err("stratum: oversized message"); break; }
            size_t nl;
            while ((nl = buf.find('\n')) != std::string::npos) {
                std::string line = buf.substr(0, nl); buf.erase(0, nl + 1);
                if (line.find_first_not_of(" \r\t") == std::string::npos) continue;
                JV msg; size_t p = 0;
                if (!jparse(line, p, msg) || msg.t != JV::OBJ) { log_err("stratum: malformed message: " + line.substr(0, 120)); continue; }
                const JV *method = msg.get("method"), *params = msg.get("params"), *idv = msg.get("id");
                if (method && method->t == JV::STR) {
                    static const JV empty;
                    const JV &pa = params ? *params : empty;
                    if (method->s == "mining.set_difficulty") {
                        const JV *d = pa.at(0);
                        double k = d && d->n > 0 ? std::log2(d->n) : -1;
                        if (k < 1 || k > 255 || std::fabs(k - std::round(k)) > 1e-9) { log_err("stratum: difficulty is not a power of two: " + line.substr(0, 120)); continue; }
                        pending_k = (int)std::lround(k);
                        if (cur_k == 0) cur_k = pending_k;
                        if (sh.con.verbose) logln("stratum: difficulty 2^" + std::to_string(pending_k) + " from the next job");
                    } else if (method->s == "mining.notify") stratum_notify(sh, pa, cur_k, pending_k);
                    else if (method->s == "client.reconnect") { log_warn("stratum: the pool asks to reconnect"); reconnect = true; break; }
                    else if (method->s == "client.show_message") { if (const JV *m = pa.at(0)) logln("pool: " + m->s); }
                    continue;
                }
                uint64_t id = idv && idv->t == JV::NUM ? (uint64_t)idv->n : 0;
                const JV *res = msg.get("result"), *er = msg.get("error");
                std::string why; if (er && er->t == JV::ARR) { if (const JV *m = er->at(1)) why = m->s; }
                if (id == 1) {
                    const JV *en1 = res ? res->at(1) : nullptr, *n2 = res ? res->at(2) : nullptr;
                    if (!en1 || en1->t != JV::STR || en1->s.size() != 4 || !n2 || (int)n2->n != 6) { log_err("stratum: subscribe refused or unsupported" + (why.empty() ? "" : ": " + why)); reconnect = true; break; }
                    std::lock_guard<std::mutex> l(sh.wmu);
                    sh.en1 = (uint16_t)strtoul(en1->s.c_str(), nullptr, 16); sh.subscribed = true;
                } else if (id == 2) {
                    if (res && res->t == JV::BOOL && res->b) logln("stratum: authorized as " + sh.user);
                    else { log_err("stratum: authorization of " + sh.user + " refused" + (why.empty() ? "" : ": " + why) + " — check --token and --user"); authFailed = true; reconnect = true; break; }
                } else if (id >= 100) {
                    Shared::Pending pd{-1, 0}; bool found = false;
                    { std::lock_guard<std::mutex> l(sh.wmu); auto it = sh.pending.find(id); if (it != sh.pending.end()) { pd = it->second; sh.pending.erase(it); found = true; } }
                    if (found) stratum_share_answer(sh, pd, msg);
                }
            }
        }
        { std::lock_guard<std::mutex> l(sh.wmu); sh.sfd = SOCK_INVALID; tls::close(sh.ssl); sh.ssl = nullptr; sh.subscribed = false; sh.pending.clear(); }
        sock_close(fd);
        { std::lock_guard<std::mutex> l(sh.mu); sh.job = Job(); }
        if (sh.stop) break;
        int wait = authFailed ? 30 : reconnect ? 1 : backoff;
        log_warn("stratum: disconnected, reconnecting in " + std::to_string(wait) + "s");
        for (int i = 0; i < wait * 4 && !sh.stop; i++) std::this_thread::sleep_for(std::chrono::milliseconds(250));
        backoff = std::chrono::steady_clock::now() - connected_at > std::chrono::minutes(1) ? 1 : std::min(backoff * 2, 30);
    }
}

// --check-pool URL: connect (with TLS for stratum+ssl), subscribe, print the answer and exit;
// needs no GPU, so it also checks a Windows package's OpenSSL DLLs on a machine without one
static int check_pool(const std::string &u) {
    Url url;
    if (!parse_url(u, url) || !url.stratum) { fprintf(stderr, "--check-pool needs a stratum+tcp:// or stratum+ssl:// URL\n"); return 1; }
    std::string err;
    sock_t fd = tcp_connect(url, 10, err);
    if (fd == SOCK_INVALID) { fprintf(stderr, "check-pool: %s\n", err.c_str()); return 1; }
    void *ssl = nullptr;
    if (url.tls && !(ssl = tls::open(fd, url.host, err))) { fprintf(stderr, "check-pool: %s\n", err.c_str()); sock_close(fd); return 1; }
    std::string req = "{\"id\":1,\"method\":\"mining.subscribe\",\"params\":[\"ProxiChorizo/" PX_VERSION " check\"]}\n";
    bool sent = ssl ? tls::ssl_write(ssl, req.data(), (int)req.size()) == (int)req.size() : send(fd, req.data(), (int)req.size(), MSG_NOSIGNAL) == (ssize_t)req.size();
    std::string line; char b[1024];
    for (int i = 0; sent && i < 50 && line.find('\n') == std::string::npos; i++) {
        int n = ssl ? tls::ssl_read(ssl, b, (int)sizeof b) : (int)recv(fd, b, (int)sizeof b, 0);
        if (n > 0) line.append(b, (size_t)n); else if (!(ssl ? tls::get_error(ssl, n) == tls::ERROR_WANT_READ : sock_timed_out())) break;
    }
    tls::close(ssl); sock_close(fd);
    line = line.substr(0, line.find('\n'));
    JV msg; size_t p = 0; const JV *res = nullptr;
    bool ok = jparse(line, p, msg) && msg.t == JV::OBJ && (res = msg.get("result")) && res->t == JV::ARR;
    fprintf(stderr, "%s %s:%d%s: %s\n", ok ? "pool OK," : "pool FAILED,", url.host.c_str(), url.port, url.tls ? (tls::insecure ? " (TLS, certificate NOT checked)" : " (TLS, certificate checked)") : "", line.empty() ? "no answer" : line.c_str());
    return ok ? 0 : 1;
}

static void submit(Shared &sh, const Job &j, uint64_t nonce, const uint8_t beta[64], uint64_t attempts, int ordinal) {
    if (sh.url.stratum) { stratum_submit(sh, j, nonce, beta, ordinal); return; }
    std::string tag; { std::lock_guard<std::mutex> l(sh.mu); tag = (size_t)ordinal < sh.gpus.size() ? gpu_tag(sh.gpus[ordinal]) : "GPU"; }
    auto count = [&](int which) {  // 0 accepted, 1 rejected, 2 stale: process total and this GPU
        std::lock_guard<std::mutex> l(sh.mu);
        if (which == 0) sh.shares++; else if (which == 1) sh.rejected++; else sh.stale++;
        if ((size_t)ordinal < sh.gpus.size()) { auto &g = sh.gpus[ordinal]; if (which == 0) g.accepted++; else if (which == 1) g.rejected++; else g.stale++; }
    };
    auto tally = [&]() { char b[64]; snprintf(b, sizeof b, " (%llu/%llu)", (unsigned long long)sh.shares.load(), (unsigned long long)(sh.shares.load() + sh.rejected.load() + sh.stale.load())); return std::string(b); };
    uint8_t nb[8]; for (int i = 0; i < 8; i++) nb[i] = nonce >> (56 - 8 * i);
    std::string body = "{\"id\":" + std::to_string(j.id) + ",\"nonce\":\"" + hexs(nb, 8) + "\",\"beta\":\"" + hexs(beta, 64) + "\",\"attempts\":" + std::to_string(attempts) + ",\"seeker\":\"" + sh.name + "\"}";
    HttpResp r = http_do(sh.url, "POST", "/seeker/result", body, sh.token, 10);
    if (!r.err.empty()) { log_err(tag + ": job " + std::to_string(j.id) + ": cannot post the solution: " + r.err); return; }
    std::string beath; bool accepted = false;
    if (r.status == 200) {
        json_bool(r.body, "accepted", accepted); json_str(r.body, "beat", beath);
        std::vector<uint8_t> b;
        {
            std::lock_guard<std::mutex> l(sh.mu);
            if (sh.job.id == j.id && unhex(beath, b) && b.size() == 64) {
                if (!sh.job.has_beat || memcmp(b.data(), sh.job.beat, 64) < 0) { memcpy(sh.job.beat, b.data(), 64); sh.job.has_beat = true; }
            }
        }
        // a pool answers every valid share with accepted and an empty beat; only a
        // solo miner (proxi node mine) sends a beat, and then a result is a solution
        if (beath.empty()) {
            count(accepted ? 0 : 1);
            if (accepted) { if (!sh.con.quiet) log_ok(tag + ": share accepted" + tally()); }
            else log_err(tag + ": share REJECTED by the pool" + tally());
        }
        else if (accepted) { sh.solutions++; log_ok(tag + ": job " + std::to_string(j.id) + " SOLVED with nonce 0x" + hexs(nb, 8) + ", accepted; searching for a smaller output"); }
        else logln(tag + ": job " + std::to_string(j.id) + ": valid but beaten (beat now " + beath.substr(0, 16) + "…)");
    } else if (r.status == 409) {
        { std::lock_guard<std::mutex> l(sh.mu); if (sh.job.id == j.id) sh.job.stale = true; }
        count(2);
        log_warn(tag + ": share stale, the pool had moved on from job " + std::to_string(j.id) + tally());
    } else if (r.status == 422) {
        std::string reason; json_str(r.body, "error", reason);
        count(1);
        log_err(tag + ": share REJECTED, nonce 0x" + hexs(nb, 8) + " of job " + std::to_string(j.id) + ": " + reason + tally());
    } else log_warn(tag + ": job " + std::to_string(j.id) + ": result answered HTTP " + std::to_string(r.status) + " " + r.body.substr(0, 120));
}

struct GpuCfg { int dev; int blocks; int threads; int iters; const VariantOps *ops; int ordinal = 0; };

// A CUDA failure inside the mining loop ends this worker only (gpu_failed marks the
// GPU dead, the other workers keep going); set-up failures before the loop still abort.
#define GPU_TRY(x) if (cudaError_t e_ = (x)) { fail = std::string(cudaGetErrorString(e_)) + " in " #x; break; }
static void gpu_worker(Shared &sh, GpuCfg cfg) {
    CUDA_CHECK(cudaSetDevice(cfg.dev));
    cudaDeviceProp prop; CUDA_CHECK(cudaGetDeviceProperties(&prop, cfg.dev));
    if (cfg.blocks <= 0) cfg.blocks = prop.multiProcessorCount * 8;
    cfg.ops->set_scalar(sh.scalar);
    JobDev *d_job; Hit *d_hits; unsigned *d_n;
    CUDA_CHECK(cudaMalloc(&d_job, sizeof(JobDev)));
    CUDA_CHECK(cudaMalloc(&d_hits, sizeof(Hit) * MAX_HITS));
    CUDA_CHECK(cudaMalloc(&d_n, sizeof(unsigned)));
    std::vector<Hit> hits(MAX_HITS);
    std::mt19937_64 rng(std::random_device{}() ^ (uint64_t)cfg.dev * 0x9E3779B97F4A7C15ULL);
    uint64_t working = 0, nonce = 0;
    uint64_t stride = (uint64_t)cfg.blocks * cfg.threads;
    void *scratch = cfg.ops->alloc(stride * (size_t)cfg.iters);
    cudaEvent_t ev0, ev1; CUDA_CHECK(cudaEventCreate(&ev0)); CUDA_CHECK(cudaEventCreate(&ev1));
    // test hook: PX_TEST_FAIL_GPU=<device>:<seconds> makes this worker issue one invalid CUDA call
    // after that many seconds, to exercise the dead-GPU path without real hardware trouble
    double failAfter = -1; auto t_start = std::chrono::steady_clock::now();
    if (const char *e = getenv("PX_TEST_FAIL_GPU")) { int d = -1; double s = 0; if (sscanf(e, "%d:%lf", &d, &s) == 2 && d == cfg.dev) failAfter = s; }
    std::string fail;
    while (!sh.stop) {
        if (!thermal_gate(sh, cfg.ordinal)) {
            { std::lock_guard<std::mutex> l(sh.mu); sh.gpus[cfg.ordinal].win_idle_ms += 1000; }
            std::this_thread::sleep_for(std::chrono::seconds(1)); continue;
        }
        Job j; { std::lock_guard<std::mutex> l(sh.mu); j = sh.job; }
        // past its ttl a job is still the current one until the miner replaces it or
        // answers 409, so keep searching: an idle GPU earns nothing, a late solution
        // costs one refused post
        if (j.id == 0 || j.stale) {
            { std::lock_guard<std::mutex> l(sh.mu); sh.gpus[cfg.ordinal].win_idle_ms += 10; }
            std::this_thread::sleep_for(std::chrono::milliseconds(10)); continue;
        }
        // a new job: this GPU's own range of the worker's nonce space (8 bits of GPU
        // ordinal under the pool's 16-bit worker index, 2^40 nonces per GPU per job),
        // or a random start when the job server assigns no range
        if (j.id != working) { working = j.id; nonce = j.has_base ? (j.nonce_base | ((uint64_t)(cfg.ordinal & 0xff) << 40)) : rng(); }
        JobDev jd; memset(&jd, 0, sizeof jd);
        memcpy(jd.pk, sh.pk, 32); memcpy(jd.prefix, j.prefix.data(), j.prefix.size()); jd.prefix_len = (int)j.prefix.size();
        memcpy(jd.beat, j.beat, 64); jd.k = j.k;
        if (failAfter >= 0 && std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count() >= failAfter) {
            failAfter = -1; log_warn("GPU #" + std::to_string(cfg.dev) + ": PX_TEST_FAIL_GPU: injecting an invalid CUDA call");
            GPU_TRY(cudaMemcpy(d_job, nullptr, sizeof jd, cudaMemcpyHostToDevice));
        }
        GPU_TRY(cudaMemcpy(d_job, &jd, sizeof jd, cudaMemcpyHostToDevice));
        GPU_TRY(cudaMemset(d_n, 0, sizeof(unsigned)));
        GPU_TRY(cudaEventRecord(ev0));
        cfg.ops->mine(cfg.blocks, cfg.threads, d_job, nonce, stride, cfg.iters, d_hits, d_n, scratch);
        GPU_TRY(cudaGetLastError());
        GPU_TRY(cudaEventRecord(ev1));
        GPU_TRY(cudaDeviceSynchronize());
        float ms = 0; GPU_TRY(cudaEventElapsedTime(&ms, ev0, ev1));
        uint64_t done = stride * cfg.iters;
        nonce += done;
        sh.attempts += done;
        { std::lock_guard<std::mutex> l(sh.mu); auto &g = sh.gpus[cfg.ordinal]; g.win_attempts += done; g.win_kernel_ms += ms; g.win_launches++; }
        unsigned n = 0; GPU_TRY(cudaMemcpy(&n, d_n, sizeof n, cudaMemcpyDeviceToHost));
        if (n > 0) {
            if (n > MAX_HITS) n = MAX_HITS;
            GPU_TRY(cudaMemcpy(hits.data(), d_hits, sizeof(Hit) * n, cudaMemcpyDeviceToHost));
            std::sort(hits.begin(), hits.begin() + n, [](const Hit &a, const Hit &b) { return memcmp(a.beta, b.beta, 64) < 0; });
            for (unsigned i = 0; i < n; i++) {
                Job cur; { std::lock_guard<std::mutex> l(sh.mu); cur = sh.job; }
                if (cur.id != j.id || cur.stale) break;
                if (cur.has_beat && memcmp(hits[i].beta, cur.beat, 64) >= 0) continue;
                submit(sh, cur, hits[i].nonce, hits[i].beta, sh.attempts.exchange(0), cfg.ordinal);
            }
        }
    }
    if (!fail.empty()) gpu_failed(sh, cfg.ordinal, "CUDA error: " + fail);
}

// ----------------------------------------------------------------------------
// stats API: GET / (or /stats) -> JSON for dashboards, GET /mmpos -> the mmpOS
// custom-miner line. Read-only, meant for 127.0.0.1 (HiveOS h-stats.sh, mmp-stats.sh).
// ----------------------------------------------------------------------------
static std::string json_escape(const std::string &v) {
    std::string o;
    for (char c : v) { if (c == '"' || c == '\\') o += '\\'; if ((unsigned char)c >= 0x20) o += c; }
    return o;
}
static std::string stats_json(Shared &sh, bool mmpos) {
    std::lock_guard<std::mutex> l(sh.mu);
    uint64_t up = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - sh.started).count();
    uint64_t acc = sh.shares.load() + sh.solutions.load(), rej = sh.rejected.load(), stale = sh.stale.load();
    double total = 0; std::string hs, bus, gpus, temp, fan, power; int active = 0;
    for (auto &g : sh.gpus) {
        total += g.rate;
        if (!g.dead && !g.paused) active++;
        char r[64]; snprintf(r, sizeof r, "%.0f", g.rate);
        if (!hs.empty()) { hs += ","; bus += ","; gpus += ","; temp += ","; fan += ","; power += ","; }
        hs += r; bus += std::to_string(g.bus_num);
        temp += std::to_string(g.sens.temp < 0 ? 0 : g.sens.temp); fan += std::to_string(g.sens.fan < 0 ? 0 : g.sens.fan);
        power += std::to_string(g.sens.power < 0 ? 0 : (int)(g.sens.power + 0.5));
        // 0.5.9 additions: status/paused/reason (thermal pause, device error), per-GPU share counters, mem_temp and clock in sensors_json
        gpus += "{\"id\":" + std::to_string(g.ordinal) + ",\"cuda_device\":" + std::to_string(g.dev) + ",\"name\":\"" + json_escape(g.name) +
                "\",\"bus_id\":\"" + g.bus + "\",\"bus_number\":" + std::to_string(g.bus_num) + ",\"hashrate\":" + r + sensors_json(g.sens) +
                ",\"status\":\"" + (g.dead ? "dead" : g.paused ? "paused" : "mining") + "\",\"paused\":" + (g.paused ? "true" : "false") +
                ",\"reason\":" + (g.reason.empty() ? std::string("null") : "\"" + json_escape(g.reason) + "\"") +
                ",\"accepted\":" + std::to_string(g.accepted) + ",\"rejected\":" + std::to_string(g.rejected) + ",\"stale\":" + std::to_string(g.stale) + "}";
    }
    if (mmpos)
        return "{\"busid\":[" + bus + "],\"hash\":[" + hs + "],\"units\":\"hs\",\"temp\":[" + temp + "],\"fan\":[" + fan + "],\"power\":[" + power +
               "],\"air\":[\"" + std::to_string(acc) + "\",\"0\",\"" + std::to_string(rej) +
               "\"],\"miner_name\":\"ProxiChorizo\",\"miner_version\":\"" PX_VERSION "\"}";
    char t[64]; snprintf(t, sizeof t, "%.0f", total);
    return "{\"miner\":\"ProxiChorizo\",\"version\":\"" PX_VERSION "\",\"uptime\":" + std::to_string(up) + ",\"hashrate\":" + t +
           ",\"hashrate_units\":\"hs\",\"shares\":{\"accepted\":" + std::to_string(acc) + ",\"rejected\":" + std::to_string(rej) + ",\"stale\":" + std::to_string(stale) + "},\"solutions\":" +
           std::to_string(sh.solutions.load()) + ",\"job_server\":\"" + json_escape(sh.url.host) + ":" + std::to_string(sh.url.port) + "\",\"worker\":\"" +
           json_escape(sh.name) + "\",\"job_k\":" + std::to_string(sh.job.k) + ",\"job_id\":" + std::to_string(sh.job.id) +
           ",\"gpus_active\":" + std::to_string(active) + ",\"max_temp\":" + std::to_string(sh.con.maxTemp) + ",\"max_mem_temp\":" + std::to_string(sh.con.maxMemTemp) +
           ",\"gpus\":[" + gpus + "]}";
}
static void api_server(Shared &sh) {
    std::string host = "127.0.0.1"; int port = 0;
    size_t c = sh.api_bind.rfind(':');
    if (c == std::string::npos) port = atoi(sh.api_bind.c_str()); else { host = sh.api_bind.substr(0, c); port = atoi(sh.api_bind.c_str() + c + 1); }
    if (host.empty()) host = "0.0.0.0";
    sock_t ls = socket(AF_INET, SOCK_STREAM, 0);
    if (ls == SOCK_INVALID) { logln("stats API: cannot create a socket"); return; }
    int one = 1; setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, (const char *)&one, sizeof one);
    sockaddr_in a; memset(&a, 0, sizeof a); a.sin_family = AF_INET; a.sin_port = htons((unsigned short)port);
    if (inet_pton(AF_INET, host.c_str(), &a.sin_addr) != 1) { logln("stats API: bad bind address " + host); sock_close(ls); return; }
    if (bind(ls, (sockaddr *)&a, sizeof a) != 0 || listen(ls, 8) != 0) { logln("stats API: cannot listen on " + sh.api_bind); sock_close(ls); return; }
    logln("stats API on http://" + host + ":" + std::to_string(port) + "/  (/mmpos for the mmpOS line)");
    while (!sh.stop) {
        sock_t fd = accept(ls, nullptr, nullptr);
        if (fd == SOCK_INVALID) continue;
        sock_timeout(fd, 3);
        char buf[2048]; ssize_t n = recv(fd, buf, (int)sizeof buf - 1, 0);
        std::string path = "/";
        if (n > 0) { buf[n] = 0; std::string req(buf); size_t sp = req.find(' '), sp2 = req.find(' ', sp + 1); if (sp != std::string::npos && sp2 != std::string::npos) path = req.substr(sp + 1, sp2 - sp - 1); }
        std::string body = stats_json(sh, path.rfind("/mmpos", 0) == 0);
        std::string resp = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: " + std::to_string(body.size()) + "\r\nConnection: close\r\n\r\n" + body;
        send(fd, resp.data(), (int)resp.size(), MSG_NOSIGNAL);
        sock_close(fd);
    }
    sock_close(ls);
}

static bool load_seed(const std::string &seedhex, const std::string &keyfile, uint8_t seed[32]) {
    std::string sh = seedhex;
    if (sh.empty()) { const char *e = getenv("NONCE_SEEKER_SEED"); if (e) sh = e; }
    if (sh.empty() && !keyfile.empty()) {
        std::ifstream f(keyfile); if (!f) { fprintf(stderr, "cannot open key file %s\n", keyfile.c_str()); return false; }
        std::stringstream ss; ss << f.rdbuf(); std::string js = ss.str();
        bool isenc = false; json_bool(js, "encrypted", isenc);
        if (isenc) { fprintf(stderr, "encrypted key files are not supported yet; pass --seed-hex or NONCE_SEEKER_SEED\n"); return false; }
        std::string pkh;
        if (!json_str(js, "private_key", pkh)) { fprintf(stderr, "no private_key in %s\n", keyfile.c_str()); return false; }
        sh = pkh.substr(0, 64);
    }
    std::vector<uint8_t> b;
    if (!unhex(sh, b) || b.size() < 32) { fprintf(stderr, "seed must be >= 32 bytes hex (got %zu bytes)\n", b.size()); return false; }
    memcpy(seed, b.data(), 32);
    return true;
}

// selftest: every vector through the variant, intermediates compared with the Go reference
static int run_selftest(const VariantOps *ops, const std::string &path) {
    std::ifstream f(path); if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); return 2; }
    std::string line; int n = 0, bad = 0;
    JobDev *d_job; CUDA_CHECK(cudaMalloc(&d_job, sizeof(JobDev)));
    std::string cur_x;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::string pk, x, alpha, h, cg, beta; uint64_t ctr = 0;
        json_str(line, "pk", pk); json_str(line, "x", x); json_str(line, "alpha", alpha);
        json_str(line, "h", h); json_str(line, "cg", cg); json_str(line, "beta", beta); json_u64(line, "ctr", ctr);
        std::vector<uint8_t> xb, pkb, ab;
        unhex(x, xb); unhex(pk, pkb); unhex(alpha, ab);
        if (xb.size() != 32 || pkb.size() != 32 || ab.size() != 45) { fprintf(stderr, "bad vector line\n"); bad++; continue; }
        if (x != cur_x) { ops->set_scalar(xb.data()); cur_x = x; }
        JobDev jd; memset(&jd, 0, sizeof jd);
        memcpy(jd.pk, pkb.data(), 32); memcpy(jd.prefix, ab.data(), 37); jd.prefix_len = 37;
        uint64_t nonce = 0; for (int i = 0; i < 8; i++) nonce = (nonce << 8) | ab[37 + i];
        CUDA_CHECK(cudaMemcpy(d_job, &jd, sizeof jd, cudaMemcpyHostToDevice));
        uint8_t ob[64], oh[32], ocg[32];
        int ok = ops->test(d_job, nonce, ob, oh, ocg);
        n++;
        bool hok = hexs(oh, 32) == h, cgok = hexs(ocg, 32) == cg, bok = hexs(ob, 64) == beta;
        if (!ok || !hok || !cgok || !bok) {
            bad++;
            fprintf(stderr, "[%s] vector %d FAILED (ctr=%llu): H %s, 8G %s, beta %s\n", ops->name, n, (unsigned long long)ctr, hok ? "ok" : "MISMATCH", cgok ? "ok" : "MISMATCH", bok ? "ok" : "MISMATCH");
        }
    }
    fprintf(stderr, "[%s] selftest: %d vectors, %d failed\n", ops->name, n, bad);
    cudaFree(d_job);
    return bad ? 1 : 0;
}

// bench: realistic job shape under the loaded scalar; returns MH/s
static double run_bench(const VariantOps *ops, const uint8_t pk[32], int dev, int blocks, int threads, int iters, double seconds, bool verbose) {
    CUDA_CHECK(cudaSetDevice(dev));
    cudaDeviceProp prop; CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    if (blocks <= 0) blocks = prop.multiProcessorCount * 8;
    JobDev jd; memset(&jd, 0, sizeof jd); jd.prefix_len = 37; jd.k = 20; memset(jd.beat, 0xff, 64);
    memcpy(jd.pk, pk, 32);
    for (int i = 0; i < 37; i++) jd.prefix[i] = (uint8_t)(i * 31 + 7);
    JobDev *d_job; Hit *d_hits; unsigned *d_n;
    CUDA_CHECK(cudaMalloc(&d_job, sizeof(JobDev))); CUDA_CHECK(cudaMalloc(&d_hits, sizeof(Hit) * MAX_HITS)); CUDA_CHECK(cudaMalloc(&d_n, 4));
    CUDA_CHECK(cudaMemcpy(d_job, &jd, sizeof jd, cudaMemcpyHostToDevice));
    uint64_t stride = (uint64_t)blocks * threads, total = 0, nonce = 0x123456789ULL;
    void *scratch = ops->alloc(stride * (size_t)iters);
    // one warm-up launch, then timed
    CUDA_CHECK(cudaMemset(d_n, 0, 4)); ops->mine(blocks, threads, d_job, nonce, stride, iters, d_hits, d_n, scratch); CUDA_CHECK(cudaDeviceSynchronize());
    auto t0 = std::chrono::steady_clock::now();
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < seconds) {
        CUDA_CHECK(cudaMemset(d_n, 0, 4));
        ops->mine(blocks, threads, d_job, nonce, stride, iters, d_hits, d_n, scratch);
        CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
        nonce += stride * iters; total += stride * iters;
    }
    ops->free_(scratch);
    double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    double mhs = total / s / 1e6;
    if (verbose) fprintf(stderr, "GPU %d %s [%s %dx%d iters %d]: %.3f MH/s\n", dev, prop.name, ops->name, blocks, threads, iters, mhs);
    cudaFree(d_job); cudaFree(d_hits); cudaFree(d_n);
    return mhs;
}

// autotune: every variant and a few geometries on this device, best wins
static GpuCfg autotune(int dev, const uint8_t scalar[32], const uint8_t pk[32], double seconds, int fixedBlocks, bool quick = false) {
    CUDA_CHECK(cudaSetDevice(dev));
    cudaDeviceProp prop; CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    const Platform &pl = platformFor(prop);
    GpuCfg best{dev, fixedBlocks > 0 ? fixedBlocks : prop.multiProcessorCount * pl.blocksPerSM, pl.threads, pl.iters, variantByName(pl.variant)};
    double bestRate = 0;
    // threads, nonces per thread, blocks per multiprocessor (ignored under --blocks)
    const int geoAll[][3] = { {256, 4, 8}, {256, 2, 8}, {128, 4, 8}, {256, 1, 64}, {256, 1, 128}, {256, 2, 64} };
    // quick: the kernels that win somewhere (0.5.8 comba, wNAF, 4-bit table), at the two launch shapes that do
    const int geoQuick[][3] = { {256, 1, 128}, {256, 4, 8} };
    const int (*geo)[3] = quick ? geoQuick : geoAll;
    const int ngeo = quick ? 2 : 6;
    fprintf(stderr, "autotune GPU %d %s (sm_%d%d), %.0fs per trial:\n", dev, prop.name, prop.major, prop.minor, seconds);
    for (int v = 0; v < NVARIANTS; v++) {
        if (quick && strcmp(VARIANTS[v]->name, "fe32c_wnaf_s16_c") != 0 && strcmp(VARIANTS[v]->name, "fe32_wnaf_lb2_c") != 0 && strcmp(VARIANTS[v]->name, "fe32_w4_lb2_c") != 0) continue;
        uint8_t pk2[32];
        if (VARIANTS[v]->init(scalar, pk2) != 1 || memcmp(pk2, pk, 32) != 0) { fprintf(stderr, "  %-12s: init failed, skipped\n", VARIANTS[v]->name); continue; }
        for (int gi = 0; gi < ngeo; gi++) {
            const int *g = geo[gi];
            int blocks = fixedBlocks > 0 ? fixedBlocks : prop.multiProcessorCount * g[2];
            double r = run_bench(VARIANTS[v], pk, dev, blocks, g[0], g[1], seconds, false);
            fprintf(stderr, "  %-16s %5dx%3dx%d: %.3f MH/s%s\n", VARIANTS[v]->name, blocks, g[0], g[1], r, r > bestRate ? "  <- best so far" : "");
            if (r > bestRate) { bestRate = r; best = GpuCfg{dev, blocks, g[0], g[1], VARIANTS[v]}; }
        }
    }
    fprintf(stderr, "autotune GPU %d: %s %dx%dx%d (%.3f MH/s)\n", dev, best.ops->name, best.blocks, best.threads, best.iters, bestRate);
    return best;
}

// Tuning cache for generations this build has no measurements for: one line per
// card and build, "<name>|sm<xy>|<version>|<variant> <blocks> <threads> <iters>",
// in proxichorizo-autotune.cache next to where the miner runs. A new release
// re-measures (its defaults may have moved); --autotune / --variant / --blocks
// bypass it.
static const char *TUNE_CACHE = "proxichorizo-autotune.cache";
static std::string tuneKey(const cudaDeviceProp &prop) {
    return std::string(prop.name) + "|sm" + std::to_string(prop.major) + std::to_string(prop.minor) + "|" PX_VERSION "|";
}
static bool loadTuned(const cudaDeviceProp &prop, int dev, GpuCfg &cfg) {
    std::ifstream in(TUNE_CACHE);
    std::string line, key = tuneKey(prop);
    while (std::getline(in, line)) {
        if (line.compare(0, key.size(), key) != 0) continue;
        std::istringstream ss(line.substr(key.size()));
        std::string v; int b = 0, t = 0, it = 0;
        if (!(ss >> v >> b >> t >> it) || b <= 0 || t <= 0 || it <= 0) return false;
        const VariantOps *ops = variantByName(v);
        if (!ops) return false;
        cfg = GpuCfg{dev, b, t, it, ops};
        return true;
    }
    return false;
}
static void saveTuned(const cudaDeviceProp &prop, const GpuCfg &cfg) {
    std::ofstream out(TUNE_CACHE, std::ios::app);
    if (!out) { logln(std::string("cannot write ") + TUNE_CACHE + ": the result is not kept for the next start"); return; }
    out << tuneKey(prop) << cfg.ops->name << ' ' << cfg.blocks << ' ' << cfg.threads << ' ' << cfg.iters << '\n';
}


// Compare complete outputs for small, awkward launch shapes. The compacted
// reference has the same bounded hash-to-curve retry policy as the candidates.
// k=0 retains every point that the pipeline computes, rather than sampling hits.
static int run_pipeline_edges(const uint8_t pk[32], int dev) {
    CUDA_CHECK(cudaSetDevice(dev));
    const VariantOps *reference = variantByName("fe32_w4_lb2_c");
    if (!reference) for (int v = 0; v < NVARIANTS && !reference; v++) if (VARIANTS[v]->compact) reference = VARIANTS[v];
    if (!reference) return 2;
    struct Shape { int blocks, threads, iters; };
    std::vector<Shape> shapes;
    for (int n : {1,2,3,7,8,9,15,16,17,31,32,33,63,64,65,127,128,129,191,192,193,255,256}) {
        shapes.push_back({1,n,1});
        shapes.push_back({n,1,1});
    }
    for (int it : {1,2,3,7,8}) shapes.push_back({1,32,it});
    for (int it : {1,3,7,11}) shapes.push_back({3,7,it});
    for (int it : {1,3,7}) shapes.push_back({7,5,it});
    std::vector<const VariantOps *> opslist{reference};
    for (int v=0;v<NVARIANTS;v++) if (VARIANTS[v]->compact && VARIANTS[v]!=reference) opslist.push_back(VARIANTS[v]);
    JobDev *d_job; Hit *d_hits; unsigned *d_n;
    CUDA_CHECK(cudaMalloc(&d_job,sizeof(JobDev)));
    CUDA_CHECK(cudaMalloc(&d_hits,sizeof(Hit)*MAX_HITS));
    CUDA_CHECK(cudaMalloc(&d_n,sizeof(unsigned)));
    std::vector<std::vector<std::string>> expected(shapes.size());
    int rc=0;
    for (const VariantOps *ops : opslist) {
        uint8_t pk2[32]; if(ops->init(g_scalar,pk2)!=1 || memcmp(pk,pk2,32)) {rc=1;continue;}
        size_t points=0;
        for (size_t c=0;c<shapes.size();c++) {
            Shape s=shapes[c]; uint64_t stride=(uint64_t)s.blocks*s.threads;
            size_t count=stride*s.iters;
            if(count>MAX_HITS) {rc=1;continue;}
            JobDev jd{};jd.prefix_len=37;jd.k=0;memcpy(jd.pk,pk,32);memset(jd.beat,255,64);
            for(int i=0;i<37;i++)jd.prefix[i]=(uint8_t)(i*31+7+c*19);
            CUDA_CHECK(cudaMemcpy(d_job,&jd,sizeof jd,cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemset(d_n,0,sizeof(unsigned)));
            void *scratch=ops->alloc(count);
            ops->mine(s.blocks,s.threads,d_job,0xabcdef0000ULL+c*1009,stride,s.iters,d_hits,d_n,scratch);
            CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaDeviceSynchronize());
            unsigned n=0;CUDA_CHECK(cudaMemcpy(&n,d_n,sizeof n,cudaMemcpyDeviceToHost));
            if(n>MAX_HITS){fprintf(stderr,"[%s] pipeline edge overflow\n",ops->name);ops->free_(scratch);rc=1;continue;}
            std::vector<Hit> hits(n);if(n)CUDA_CHECK(cudaMemcpy(hits.data(),d_hits,n*sizeof(Hit),cudaMemcpyDeviceToHost));
            ops->free_(scratch);
            std::vector<std::string> found;
            for(auto &hit:hits)found.push_back(std::to_string(hit.nonce)+":"+hexs(hit.beta,64));
            std::sort(found.begin(),found.end());points+=found.size();
            if(ops==reference)expected[c]=found;
            else if(found!=expected[c]){fprintf(stderr,"[%s] pipeline edge %zu (%dx%dx%d) DIFFERENT\n",ops->name,c,s.blocks,s.threads,s.iters);rc=1;}
        }
        fprintf(stderr,"[%s] pipeline edges: %zu shapes, %zu complete outputs%s\n",ops->name,shapes.size(),points,ops==reference?" (reference)":" compared");
    }
    cudaFree(d_job);cudaFree(d_hits);cudaFree(d_n);
    return rc;
}


// verify: every variant over the same nonce range at low difficulty must find exactly the same hits
static int run_verify(const uint8_t pk[32], int dev) {
    CUDA_CHECK(cudaSetDevice(dev));
    cudaDeviceProp prop; CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    int blocks = prop.multiProcessorCount * 4, threads = 256, iters = 2, launches = 3;
    JobDev jd; memset(&jd, 0, sizeof jd); jd.prefix_len = 37; jd.k = 15; memset(jd.beat, 0xff, 64);
    memcpy(jd.pk, pk, 32);
    for (int i = 0; i < 37; i++) jd.prefix[i] = (uint8_t)(i * 31 + 7);
    JobDev *d_job; Hit *d_hits; unsigned *d_n;
    CUDA_CHECK(cudaMalloc(&d_job, sizeof(JobDev))); CUDA_CHECK(cudaMalloc(&d_hits, sizeof(Hit) * MAX_HITS)); CUDA_CHECK(cudaMalloc(&d_n, 4));
    CUDA_CHECK(cudaMemcpy(d_job, &jd, sizeof jd, cudaMemcpyHostToDevice));
    uint64_t stride = (uint64_t)blocks * threads;
    std::vector<std::string> ref; int rc = 0;
    for (int v = 0; v < NVARIANTS; v++) {
        const VariantOps *ops = VARIANTS[v];
        uint8_t pk2[32]; uint8_t sc0[32]; (void)sc0; if (ops->init(g_scalar, pk2) != 1) { fprintf(stderr, "[%s] init failed\n", ops->name); rc = 1; continue; }
        void *scratch = ops->alloc(stride * (size_t)iters);
        std::vector<std::string> found;
        uint64_t nonce = 0xABCDEF0000ULL;
        for (int l = 0; l < launches; l++) {
            CUDA_CHECK(cudaMemset(d_n, 0, 4));
            ops->mine(blocks, threads, d_job, nonce, stride, iters, d_hits, d_n, scratch);
            CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
            unsigned n; CUDA_CHECK(cudaMemcpy(&n, d_n, 4, cudaMemcpyDeviceToHost)); if (n > MAX_HITS) n = MAX_HITS;
            std::vector<Hit> hits(n); if (n) CUDA_CHECK(cudaMemcpy(hits.data(), d_hits, sizeof(Hit) * n, cudaMemcpyDeviceToHost));
            for (auto &h : hits) { uint8_t nb[8]; for (int i = 0; i < 8; i++) nb[i] = h.nonce >> (56 - 8 * i); found.push_back(hexs(nb, 8) + ":" + hexs(h.beta, 16)); }
            nonce += stride * iters;
        }
        ops->free_(scratch);
        std::sort(found.begin(), found.end());
        if (v == 0) { ref = found; fprintf(stderr, "[%s] %zu hits over %llu nonces (reference)\n", ops->name, found.size(), (unsigned long long)(stride * iters * launches)); }
        else if (found == ref) fprintf(stderr, "[%s] %zu hits, identical\n", ops->name, found.size());
        else { fprintf(stderr, "[%s] %zu hits, DIFFERENT from reference (%zu)\n", ops->name, found.size(), ref.size()); rc = 1; }
    }
    return rc;
}

static void usage() {
    fprintf(stderr,
        "ProxiChorizo " PX_VERSION " — CUDA nonce seeker for Proxima (suprnova.cc)\n"
        "  --proxi URL        the pool: stratum+tcp://HOST:PORT or stratum+ssl://HOST:PORT (stratum), or the HTTP job server of `proxi node mine`\n"
        "                     and older pools (http://HOST:PORT); -o and --url are the same option\n"
        "  --user ADDR.WORKER stratum login (-u): your payout address and worker name, or just the worker name;\n"
        "                     default: the --name\n"
        "  --ca-file FILE     CA certificates for stratum+ssl:// besides the system store (PEM)\n"
        "  --tls-insecure     stratum+ssl:// without checking the pool's certificate (private pools only)\n"
        "  --check-pool URL   connect to a stratum pool, subscribe, print the answer and exit (no GPU needed)\n"
        "  --token T          the account token from the pool registration (stratum password; --pass and -p are the same)\n"
        "  --token-file FILE  read the bearer token from FILE (first line; keeps it off the command line), or PROXI_TOKEN in the environment\n"
        "  --key-file FILE    unencrypted proxima.key (or --seed-hex HEX / NONCE_SEEKER_SEED)\n"
        "  --seed-hex HEX     32-byte seed\n"
        "  --name NAME        seeker name in the pool's statistics (default ProxiChorizo-<pid>)\n"
        "  --api-bind [H:]P   stats API, JSON at http://H:P/ and the mmpOS line at /mmpos (default off; H defaults to 127.0.0.1)\n"
        "  -d 0,1,2,3         GPUs to use; without it every GPU mines, in one process, numbered 0.. in PCI bus order as nvidia-smi\n"
        "                     lists them (--gpus and --devices are the same option)\n"
        "  --variant NAME     kernel variant for all GPUs (default: per-platform table; --list-variants)\n"
        "  --autotune [SECS]  measure every variant on each GPU first and use the fastest (default 6s per trial)\n"
        "  --blocks N --threads N --iters N   launch geometry (defaults from the platform table)\n"
        "%s"
        "  --selftest FILE    verify every variant against vectors.json (from vrfvec) and exit\n"
        "  --fetest           randomized field-arithmetic test of every variant and exit\n"
        "  --verify           every variant over one nonce range at low difficulty must find identical hits; exit\n"
        "  --bench [SECS]     measure the hashrate of the chosen variant and exit\n", CONSOLE_USAGE);
}

// -d 0,1,2,3 (also "all"): device indices, each once, within the device count
static bool parse_devices(const std::string &gpus, int ndev, std::vector<int> &devs) {
    devs.clear();
    if (gpus.empty() || gpus == "all") { for (int i = 0; i < ndev; i++) devs.push_back(i); return true; }
    std::stringstream ss(gpus); std::string t;
    while (std::getline(ss, t, ',')) {
        while (!t.empty() && t.front() == ' ') t.erase(0, 1);
        while (!t.empty() && t.back() == ' ') t.pop_back();
        if (t.empty()) continue;
        char *end = nullptr; long d = strtol(t.c_str(), &end, 10);
        if (*end || d < 0 || d >= ndev) { fprintf(stderr, "-d %s: no GPU %s (this machine has %d, numbered 0..%d)\n", gpus.c_str(), t.c_str(), ndev, ndev - 1); return false; }
        if (std::find(devs.begin(), devs.end(), (int)d) != devs.end()) continue;
        devs.push_back((int)d);
    }
    if (devs.empty()) { fprintf(stderr, "-d %s selects no GPU\n", gpus.c_str()); return false; }
    return true;
}

// Ctrl-C / SIGTERM: one summary line, then out (the workers hold no state worth flushing)
static Shared *g_sh = nullptr;
static void on_signal(int) {
    if (g_sh) {
        char b[200];
        snprintf(b, sizeof b, "\n%s stopped: accepted %llu, rejected %llu, stale %llu, uptime %s\n", now_str().c_str(), (unsigned long long)g_sh->shares.load(),
                 (unsigned long long)g_sh->rejected.load(), (unsigned long long)g_sh->stale.load(), fmt_uptime(std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - g_sh->started).count()).c_str());
        fputs(b, stderr); fflush(stderr);
    }
    _Exit(0);
}

int main(int argc, char **argv) {
    std::string proxi, token, tokenFile, keyfile, seedhex, name, selftest, gpus, variant, apiBind, user, checkPool;
    sock_init();
    // GPU numbers follow the PCI bus order (as nvidia-smi lists the cards) unless the caller
    // chose an order; the runtime reads this before its first call
#ifdef _WIN32
    if (!getenv("CUDA_DEVICE_ORDER")) _putenv_s("CUDA_DEVICE_ORDER", "PCI_BUS_ID");
#else
    setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID", 0);
#endif
    int blocks = 0, threads = 0, iters = 0; double bench = 0, autotuneSecs = 0; bool fetest = false, listVariants = false, verify = false;
    ConsoleOpts con;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&](std::string &dst) { if (i + 1 < argc) dst = argv[++i]; };
        auto optnum = [&](double dflt) { if (i + 1 < argc && isdigit((unsigned char)argv[i + 1][0])) return atof(argv[++i]); return dflt; };
        if (a == "--help" || a == "-h") { usage(); return 0; }
        else if (a == "--proxi" || a == "-o" || a == "--url") next(proxi);
        else if (a == "--user" || a == "-u") next(user);
        else if (a == "--ca-file") next(tls::ca_file);
        else if (a == "--tls-insecure") tls::insecure = true;
        else if (a == "--check-pool") next(checkPool);
        else if (a == "--token" || a == "--pass" || a == "-p") next(token);
        else if (a == "--token-file") next(tokenFile);
        else if (a == "--key-file") next(keyfile);
        else if (a == "--seed-hex") next(seedhex);
        else if (a == "--name") next(name);
        else if (a == "--api-bind" || a == "--api-port") next(apiBind);
        else if (a == "--gpus" || a == "-d" || a == "--devices") next(gpus);
        else if (a == "--variant") next(variant);
        else if (a == "--selftest") next(selftest);
        else if (a == "--fetest") fetest = true;
        else if (a == "--verify") verify = true;
        else if (a == "--list-variants") listVariants = true;
        else if (a == "--blocks") { std::string s; next(s); blocks = atoi(s.c_str()); }
        else if (a == "--threads") { std::string s; next(s); threads = atoi(s.c_str()); }
        else if (a == "--iters") { std::string s; next(s); iters = atoi(s.c_str()); }
        else if (a == "--bench") bench = optnum(20);
        else if (a == "--autotune") autotuneSecs = optnum(6);
        else if (parse_console_opt(a, i, argc, argv, con)) {}
        else { fprintf(stderr, "unknown option %s\n\n", a.c_str()); usage(); return 1; }
    }
    console_setup(con);
    if (!checkPool.empty()) return check_pool(checkPool);
    if (listVariants) { for (int v = 0; v < NVARIANTS; v++) fprintf(stderr, "%s\n", VARIANTS[v]->name); return 0; }
    if (token.empty() && !tokenFile.empty()) {
        std::ifstream tf(tokenFile); if (!tf) { fprintf(stderr, "cannot open token file %s\n", tokenFile.c_str()); return 1; }
        std::getline(tf, token); while (!token.empty() && (token.back() == '\r' || token.back() == '\n' || token.back() == ' ')) token.pop_back();
    }
    if (token.empty()) { const char *e = getenv("PROXI_TOKEN"); if (e) token = e; }
    int ndev = 0; CUDA_CHECK(cudaGetDeviceCount(&ndev));
    if (ndev == 0) { fprintf(stderr, "no CUDA devices\n"); return 2; }
    std::vector<int> devs;
    if (!parse_devices(gpus, ndev, devs)) return 1;

    uint8_t seed[32], scalar[32] = {0}, pk[32] = {0};
    bool keyGiven = !seedhex.empty() || !keyfile.empty() || getenv("NONCE_SEEKER_SEED") != nullptr;
    bool have_key = (!selftest.empty() || fetest || verify || !keyGiven) ? false : load_seed(seedhex, keyfile, seed);
    if (keyGiven && !have_key && selftest.empty() && !fetest && !verify) return 1;
    if (!have_key && selftest.empty() && !fetest && !verify && bench == 0 && autotuneSecs == 0) { usage(); return 1; }
    if (!have_key) {
        // bench/tests without a key: a fixed pseudo-random seed, so the scalar
        // multiplication costs what a real key costs (an all-zero scalar skips every addition)
        const char *fake = "ProxiChorizo bench scalar";
        uint8_t digest[64];
        host_sha512((const uint8_t *)fake, strlen(fake), digest);
        memcpy(seed, digest, sizeof seed);
    }
    { uint8_t h[64]; host_sha512(seed, 32, h); memcpy(scalar, h, 32); scalar[0] &= 248; scalar[31] &= 127; scalar[31] |= 64; memcpy(g_scalar, scalar, 32); }

    const VariantOps *forced = nullptr;
    if (!variant.empty() && variant != "all") { forced = variantByName(variant); if (!forced) { fprintf(stderr, "unknown variant %s (see --list-variants)\n", variant.c_str()); return 1; } }

    if (verify) {
        CUDA_CHECK(cudaSetDevice(devs[0]));
        uint8_t pk2[32]; if (VARIANTS[0]->init(scalar, pk2) != 1) { fprintf(stderr, "init failed\n"); return 2; }
        int edge_rc = run_pipeline_edges(pk2, devs[0]);
        return edge_rc | run_verify(pk2, devs[0]);
    }
    if (!selftest.empty() || fetest) {
        CUDA_CHECK(cudaSetDevice(devs[0]));
        int rc = 0;
        for (int v = 0; v < NVARIANTS; v++) {
            if (forced && VARIANTS[v] != forced) continue;
            uint8_t pk2[32]; int ok = VARIANTS[v]->init(scalar, pk2);
            if (ok != 1) { fprintf(stderr, "[%s] constant derivation FAILED (%s)\n", VARIANTS[v]->name, ok == -2 ? "fe_sq != fe_mul" : "sqrt(-1) check"); rc = 1; continue; }
            if (!selftest.empty()) rc |= run_selftest(VARIANTS[v], selftest);
            if (fetest) {
                int worst = 0;
                for (int sd = 1; sd <= 64 && !worst; sd++) { uint8_t dump[128]; int fail = VARIANTS[v]->fetest(sd, 2000, dump); if (fail) { worst = fail; fprintf(stderr, "[%s] fetest FAIL code %d at iter %d\n x %s\n", VARIANTS[v]->name, fail / 1000000, fail % 1000000, hexs(dump, 32).c_str()); } }
                if (!worst) fprintf(stderr, "[%s] fetest: 128000 random values ok\n", VARIANTS[v]->name); else rc = 1;
            }
        }
        return rc;
    }

    // per device: variant and geometry from the platform table, --variant, or autotune; constants derived and the key checked
    std::vector<GpuCfg> cfgs;
    std::vector<Shared::GpuStat> gstats;
    for (int d : devs) {
        CUDA_CHECK(cudaSetDevice(d));
        cudaDeviceProp prop; CUDA_CHECK(cudaGetDeviceProperties(&prop, d));
        { char busid[32] = "?"; cudaDeviceGetPCIBusId(busid, sizeof busid, d); int bn = (int)strtol(busid + 5, nullptr, 16);
          gstats.push_back(Shared::GpuStat{d, (int)cfgs.size(), prop.name, busid, bn, 0.0}); }
        const Platform &pl = platformFor(prop);
        GpuCfg cfg{d, blocks > 0 ? blocks : prop.multiProcessorCount * pl.blocksPerSM, threads > 0 ? threads : pl.threads, iters > 0 ? iters : pl.iters, forced ? forced : variantByName(pl.variant)};
        cfg.ordinal = (int)cfgs.size();
        if (!cfg.ops) cfg.ops = VARIANTS[0];
        uint8_t pk2[32]; int ok = cfg.ops->init(scalar, pk2);
        if (ok != 1) { fprintf(stderr, "device %d: constant derivation failed on %s\n", d, cfg.ops->name); return 2; }
        if (d == devs[0]) memcpy(pk, pk2, 32); else if (memcmp(pk, pk2, 32) != 0) { fprintf(stderr, "device %d derives a different public key\n", d); return 2; }
        if (autotuneSecs > 0) { cfg = autotune(d, scalar, pk, autotuneSecs, blocks); if (threads > 0) cfg.threads = threads; if (iters > 0) cfg.iters = iters; cfg.ops->init(scalar, pk2); }
        else if (!pl.measured && !forced && blocks <= 0 && threads <= 0 && iters <= 0 && bench <= 0) {
            GpuCfg tuned;
            if (loadTuned(prop, d, tuned)) {
                cfg = tuned; cfg.ordinal = (int)cfgs.size(); cfg.ops->init(scalar, pk2);
                fprintf(stderr, "GPU %d %s (sm_%d%d): %s, %dx%dx%d — tuned on this card earlier (%s)\n", d, prop.name, prop.major, prop.minor, cfg.ops->name, cfg.blocks, cfg.threads, cfg.iters, TUNE_CACHE);
            } else {
                fprintf(stderr, "GPU %d %s (sm_%d%d): %s; this build has no measurements for this generation, measuring the candidate kernels now (~20 s, once; kept in %s)\n", d, prop.name, prop.major, prop.minor, pl.note, TUNE_CACHE);
                cfg = autotune(d, scalar, pk, 3, 0, true); cfg.ordinal = (int)cfgs.size(); cfg.ops->init(scalar, pk2);
                saveTuned(prop, cfg);
            }
        }
        else fprintf(stderr, "GPU %d %s (sm_%d%d): %s, %dx%dx%d — %s\n", d, prop.name, prop.major, prop.minor, cfg.ops->name, cfg.blocks, cfg.threads, cfg.iters, pl.note);
        { auto &g = gstats.back(); g.arch = "sm_" + std::to_string(prop.major) + std::to_string(prop.minor); g.variant = cfg.ops->name;
          g.geometry = std::to_string(cfg.blocks) + "x" + std::to_string(cfg.threads) + "x" + std::to_string(cfg.iters); }
        cfgs.push_back(cfg);
    }
    if (bench > 0) { for (auto &c : cfgs) run_bench(c.ops, pk, c.dev, c.blocks, c.threads, c.iters, bench, true); return 0; }
    if (proxi.empty()) { if (autotuneSecs > 0) return 0; usage(); return 1; }

    Shared sh;
    sh.base_url = proxi; if (!parse_url(proxi, sh.url)) return 1;
    sh.token = token; sh.name = name.empty() ? "proxichorizo-" + std::to_string(getpid()) : name;
    sh.user = user.empty() ? sh.name : user;
    memcpy(sh.pk, pk, 32); memcpy(sh.scalar, scalar, 32);
    sh.gpus = gstats; sh.api_bind = apiBind; sh.con = con;
    print_banner(sh, "CUDA");
    logln("ProxiChorizo " PX_VERSION ": key " + hexs(pk, 32) + ", " + (sh.url.stratum ? "stratum " : "job server ") + sh.url.host + ":" + std::to_string(sh.url.port) + ", " + std::to_string(devs.size()) + " GPU(s), " + (sh.url.stratum ? "user " + sh.user : "name " + sh.name));
    for (auto &g : sh.gpus) if (read_sensors(g.bus).temp < 0) log_warn(gpu_tag(g) + ": no temperature sensor readable (NVML/hwmon), the thermal limit is not enforced on it");
    g_sh = &sh; signal(SIGINT, on_signal); signal(SIGTERM, on_signal);
    std::vector<std::thread> th;
    if (sh.url.stratum) th.emplace_back(stratum_client, std::ref(sh));
    else { th.emplace_back(poller, std::ref(sh)); th.emplace_back(reporter, std::ref(sh)); }
    th.emplace_back(console, std::ref(sh));
    if (!apiBind.empty()) th.emplace_back(api_server, std::ref(sh));
    for (auto &c : cfgs) th.emplace_back(gpu_worker, std::ref(sh), c);
    for (auto &t : th) t.join();
    return 0;
}
