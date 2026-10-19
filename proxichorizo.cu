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
#define PX_VERSION "0.5.0"
// sockets: POSIX or Winsock behind four helpers
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <winsock2.h>
#include <ws2tcpip.h>
#include <process.h>
#pragma comment(lib, "ws2_32.lib")
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
static void sock_init() {}
static void sock_close(sock_t s) { close(s); }
static void sock_timeout(sock_t s, int secs) { struct timeval tv = { secs, 0 }; setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv); setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof tv); }
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

static const VariantOps *VARIANTS[] = { &v_fe32_w0_lb2::ops, &v_fe32_w4_lb2::ops, &v_fe32_w0_lb2_c::ops, &v_fe32_w4_lb2_c::ops, &v_fe32_wnaf_lb2_c::ops, &v_fe51_w4_lb1::ops };
static const int NVARIANTS = 6;

// platform table: compute capability -> default variant and launch geometry,
// from the measurements in PLATFORMS.md; anything unlisted gets the last row
struct Platform { int major, minor; const char *variant; int threads, iters; const char *note; };
static const Platform PLATFORMS[] = {
    { 12, -1, "fe32_wnaf_lb2_c", 256, 4, "Blackwell (RTX 50xx): compacted pipeline + wNAF (17.3 MH/s on a 5080)" },
    {  8,  9, "fe32_wnaf_lb2_c", 256, 4, "Ada (RTX 40xx): assumed like Ampere/Blackwell, run --autotune once" },
    {  8,  6, "fe32_wnaf_lb2_c", 256, 4, "Ampere GA10x (RTX 30xx): compacted pipeline + wNAF (6.2 MH/s on a 3070)" },
    {  8,  0, "fe32_w4_lb2_c", 256, 2, "Ampere GA100 (A100, CMP 170HX): compacted 4-bit window; wNAF is slower here (4.7 vs 4.1 MH/s on a CMP 170HX)" },
    {  7,  5, "fe32_wnaf_lb2_c", 256, 4, "Turing: assumed, run --autotune once" },
    { -1, -1, "fe32_wnaf_lb2_c", 256, 4, "default" },
};
static uint8_t g_scalar[32];
static const VariantOps *variantByName(const std::string &n) { for (int i = 0; i < NVARIANTS; i++) if (n == VARIANTS[i]->name) return VARIANTS[i]; return nullptr; }
static const Platform &platformFor(const cudaDeviceProp &p) {
    for (const Platform &pl : PLATFORMS) if ((pl.major == -1) || (pl.major == p.major && (pl.minor == -1 || pl.minor == p.minor))) return pl;
    return PLATFORMS[sizeof(PLATFORMS) / sizeof(PLATFORMS[0]) - 1];
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
static std::mutex g_log_mu;
static void logln(const std::string &s) { std::lock_guard<std::mutex> l(g_log_mu); fprintf(stderr, "%s %s\n", now_str().c_str(), s.c_str()); }

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
struct Url { std::string host; int port = 80; };
static bool parse_url(const std::string &u, Url &out) {
    std::string s = u;
    if (s.rfind("http://", 0) == 0) s = s.substr(7);
    else if (s.rfind("https://", 0) == 0) { fprintf(stderr, "https is not supported\n"); return false; }
    size_t slash = s.find('/'); if (slash != std::string::npos) s = s.substr(0, slash);
    size_t colon = s.find(':');
    if (colon == std::string::npos) { out.host = s; out.port = 80; }
    else { out.host = s.substr(0, colon); out.port = atoi(s.substr(colon + 1).c_str()); }
    return !out.host.empty();
}
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
struct Shared {
    std::mutex mu;
    Job job;                           // current job (id 0 = none)
    std::atomic<uint64_t> attempts{0}; // since last report
    std::atomic<uint64_t> solutions{0};
    std::atomic<uint64_t> shares{0};   // pool shares accepted (results answered without a beat)
    std::atomic<uint64_t> rejected{0};
    std::atomic<bool> stop{false};
    std::string base_url; Url url; std::string token; std::string name;
    uint8_t pk[32];
    uint8_t scalar[32];
    // for the stats API (--api-bind): one entry per mining GPU, rate in H/s
    struct GpuStat { int dev; int ordinal; std::string name; std::string bus; int bus_num; double rate; };
    std::vector<GpuStat> gpus;
    std::chrono::steady_clock::time_point started = std::chrono::steady_clock::now();
    std::string api_bind;
};

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
    sh.job = j;
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
        if (after == 0) std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
}

static void reporter(Shared &sh) {
    uint64_t carried = 0; auto win_start = std::chrono::steady_clock::now(); uint64_t win_attempts = 0;
    while (!sh.stop) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        uint64_t n = sh.attempts.exchange(0) + carried;
        win_attempts += n - carried;
        uint64_t id; { std::lock_guard<std::mutex> l(sh.mu); id = sh.job.id; }
        std::string body = "{\"id\":" + std::to_string(id) + ",\"attempts\":" + std::to_string(n) + ",\"seeker\":\"" + sh.name + "\"}";
        HttpResp r = http_do(sh.url, "POST", "/seeker/report", body, sh.token, 10);
        carried = (r.err.empty() && r.status / 100 == 2) ? 0 : n;
        auto el = std::chrono::steady_clock::now() - win_start;
        if (el >= std::chrono::seconds(10)) {
            double secs = std::chrono::duration<double>(el).count();
            char line[200]; snprintf(line, sizeof line, "job %llu: %.0f H/s (%.2f MH/s), %llu shares (%llu rejected), %llu solo solutions", (unsigned long long)id, win_attempts / secs, win_attempts / secs / 1e6,
                (unsigned long long)sh.shares.load(), (unsigned long long)sh.rejected.load(), (unsigned long long)sh.solutions.load());
            if (id != 0) logln(line);
            win_start = std::chrono::steady_clock::now(); win_attempts = 0;
        }
    }
}

static void submit(Shared &sh, const Job &j, uint64_t nonce, const uint8_t beta[64], uint64_t attempts) {
    uint8_t nb[8]; for (int i = 0; i < 8; i++) nb[i] = nonce >> (56 - 8 * i);
    std::string body = "{\"id\":" + std::to_string(j.id) + ",\"nonce\":\"" + hexs(nb, 8) + "\",\"beta\":\"" + hexs(beta, 64) + "\",\"attempts\":" + std::to_string(attempts) + ",\"seeker\":\"" + sh.name + "\"}";
    HttpResp r = http_do(sh.url, "POST", "/seeker/result", body, sh.token, 10);
    if (!r.err.empty()) { logln("job " + std::to_string(j.id) + ": cannot post the solution: " + r.err); return; }
    std::string beath; bool accepted = false;
    if (r.status == 200) {
        json_bool(r.body, "accepted", accepted); json_str(r.body, "beat", beath);
        std::vector<uint8_t> b;
        std::lock_guard<std::mutex> l(sh.mu);
        if (sh.job.id == j.id && unhex(beath, b) && b.size() == 64) {
            if (!sh.job.has_beat || memcmp(b.data(), sh.job.beat, 64) < 0) { memcpy(sh.job.beat, b.data(), 64); sh.job.has_beat = true; }
        }
        // a pool answers every valid share with accepted and an empty beat; only a
        // solo miner (proxi node mine) sends a beat, and then a result is a solution
        if (beath.empty()) { sh.shares++; if (!accepted) sh.rejected++; }
        else if (accepted) { sh.solutions++; logln("job " + std::to_string(j.id) + ": SOLVED with nonce 0x" + hexs(nb, 8) + ", accepted; searching for a smaller output"); }
        else logln("job " + std::to_string(j.id) + ": valid but beaten (beat now " + beath.substr(0, 16) + "…)");
    } else if (r.status == 409) {
        std::lock_guard<std::mutex> l(sh.mu);
        if (sh.job.id == j.id) sh.job.stale = true;
        logln("job " + std::to_string(j.id) + ": solved, but the miner has moved on");
    } else if (r.status == 422) {
        std::string reason; json_str(r.body, "error", reason);
        logln("job " + std::to_string(j.id) + ": REJECTED nonce 0x" + hexs(nb, 8) + ": " + reason);
    } else logln("job " + std::to_string(j.id) + ": result answered HTTP " + std::to_string(r.status) + " " + r.body.substr(0, 120));
}

struct GpuCfg { int dev; int blocks; int threads; int iters; const VariantOps *ops; int ordinal = 0; };

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
    logln(std::string("GPU ") + std::to_string(cfg.dev) + " " + prop.name + " (sm_" + std::to_string(prop.major) + std::to_string(prop.minor) + "): variant " + cfg.ops->name + ", " +
          std::to_string(cfg.blocks) + "x" + std::to_string(cfg.threads) + " threads, " + std::to_string(cfg.iters) + " nonces/thread/launch");
    auto t_last = std::chrono::steady_clock::now(); uint64_t local_attempts = 0;
    double idle_nojob = 0, idle_stale = 0, kernel_ms = 0, gpu_ms = 0; int launches = 0;
    cudaEvent_t ev0, ev1; CUDA_CHECK(cudaEventCreate(&ev0)); CUDA_CHECK(cudaEventCreate(&ev1));
    while (!sh.stop) {
        Job j; { std::lock_guard<std::mutex> l(sh.mu); j = sh.job; }
        // past its ttl a job is still the current one until the miner replaces it or
        // answers 409, so keep searching: an idle GPU earns nothing, a late solution
        // costs one refused post
        if (j.id == 0 || j.stale) {
            if (j.id == 0) idle_nojob += 10; else idle_stale += 10;
            std::this_thread::sleep_for(std::chrono::milliseconds(10)); continue;
        }
        // a new job: this GPU's own range of the worker's nonce space (8 bits of GPU
        // ordinal under the pool's 16-bit worker index, 2^40 nonces per GPU per job),
        // or a random start when the job server assigns no range
        if (j.id != working) { working = j.id; nonce = j.has_base ? (j.nonce_base | ((uint64_t)(cfg.ordinal & 0xff) << 40)) : rng(); }
        auto k0 = std::chrono::steady_clock::now();
        JobDev jd; memset(&jd, 0, sizeof jd);
        memcpy(jd.pk, sh.pk, 32); memcpy(jd.prefix, j.prefix.data(), j.prefix.size()); jd.prefix_len = (int)j.prefix.size();
        memcpy(jd.beat, j.beat, 64); jd.k = j.k;
        CUDA_CHECK(cudaMemcpy(d_job, &jd, sizeof jd, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_n, 0, sizeof(unsigned)));
        CUDA_CHECK(cudaEventRecord(ev0));
        cfg.ops->mine(cfg.blocks, cfg.threads, d_job, nonce, stride, cfg.iters, d_hits, d_n, scratch);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(ev1));
        CUDA_CHECK(cudaDeviceSynchronize());
        { float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1)); gpu_ms += ms; }
        kernel_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - k0).count(); launches++;
        uint64_t done = stride * cfg.iters;
        nonce += done;
        sh.attempts += done; local_attempts += done;
        unsigned n = 0; CUDA_CHECK(cudaMemcpy(&n, d_n, sizeof n, cudaMemcpyDeviceToHost));
        if (n > 0) {
            if (n > MAX_HITS) n = MAX_HITS;
            CUDA_CHECK(cudaMemcpy(hits.data(), d_hits, sizeof(Hit) * n, cudaMemcpyDeviceToHost));
            std::sort(hits.begin(), hits.begin() + n, [](const Hit &a, const Hit &b) { return memcmp(a.beta, b.beta, 64) < 0; });
            for (unsigned i = 0; i < n; i++) {
                Job cur; { std::lock_guard<std::mutex> l(sh.mu); cur = sh.job; }
                if (cur.id != j.id || cur.stale) break;
                if (cur.has_beat && memcmp(hits[i].beta, cur.beat, 64) >= 0) continue;
                submit(sh, cur, hits[i].nonce, hits[i].beta, sh.attempts.exchange(0));
            }
        }
        auto now = std::chrono::steady_clock::now();
        if (now - t_last >= std::chrono::seconds(30)) {
            double s = std::chrono::duration<double>(now - t_last).count();
            char line[300]; snprintf(line, sizeof line, "GPU %d: %.2f MH/s avg over %.0fs | in kernel %.0f%% (%d launches, %.1f ms each, %.2f MH/s in-kernel) | idle: no job %.0f%%, stale %.0f%%",
                cfg.dev, local_attempts / s / 1e6, s, kernel_ms / 10 / s, launches, launches ? gpu_ms / launches : 0,
                gpu_ms > 0 ? (double)launches * stride * cfg.iters / (gpu_ms / 1000.0) / 1e6 : 0, idle_nojob / 10 / s, idle_stale / 10 / s);
            logln(line);
            { std::lock_guard<std::mutex> l(sh.mu); if ((size_t)cfg.ordinal < sh.gpus.size()) sh.gpus[cfg.ordinal].rate = local_attempts / s; }
            t_last = now; local_attempts = 0; idle_nojob = idle_stale = kernel_ms = gpu_ms = 0; launches = 0;
        }
    }
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
    uint64_t acc = sh.shares.load() + sh.solutions.load(), rej = sh.rejected.load();
    double total = 0; std::string hs, bus, gpus;
    for (auto &g : sh.gpus) {
        total += g.rate;
        char r[64]; snprintf(r, sizeof r, "%.0f", g.rate);
        if (!hs.empty()) { hs += ","; bus += ","; gpus += ","; }
        hs += r; bus += std::to_string(g.bus_num);
        gpus += "{\"id\":" + std::to_string(g.ordinal) + ",\"cuda_device\":" + std::to_string(g.dev) + ",\"name\":\"" + json_escape(g.name) +
                "\",\"bus_id\":\"" + g.bus + "\",\"bus_number\":" + std::to_string(g.bus_num) + ",\"hashrate\":" + r + "}";
    }
    if (mmpos)
        return "{\"busid\":[" + bus + "],\"hash\":[" + hs + "],\"units\":\"hs\",\"air\":[\"" + std::to_string(acc) + "\",\"0\",\"" + std::to_string(rej) +
               "\"],\"miner_name\":\"ProxiChorizo\",\"miner_version\":\"" PX_VERSION "\"}";
    char t[64]; snprintf(t, sizeof t, "%.0f", total);
    return "{\"miner\":\"ProxiChorizo\",\"version\":\"" PX_VERSION "\",\"uptime\":" + std::to_string(up) + ",\"hashrate\":" + t +
           ",\"hashrate_units\":\"hs\",\"shares\":{\"accepted\":" + std::to_string(acc) + ",\"rejected\":" + std::to_string(rej) + "},\"solutions\":" +
           std::to_string(sh.solutions.load()) + ",\"job_server\":\"" + json_escape(sh.url.host) + ":" + std::to_string(sh.url.port) + "\",\"worker\":\"" +
           json_escape(sh.name) + "\",\"job_k\":" + std::to_string(sh.job.k) + ",\"gpus\":[" + gpus + "]}";
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
static GpuCfg autotune(int dev, const uint8_t scalar[32], const uint8_t pk[32], double seconds, int fixedBlocks) {
    CUDA_CHECK(cudaSetDevice(dev));
    cudaDeviceProp prop; CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    const Platform &pl = platformFor(prop);
    GpuCfg best{dev, fixedBlocks, pl.threads, pl.iters, variantByName(pl.variant)};
    double bestRate = 0;
    const int geo[][2] = { {256, 2}, {128, 4}, {256, 4} };
    fprintf(stderr, "autotune GPU %d %s (sm_%d%d), %.0fs per trial:\n", dev, prop.name, prop.major, prop.minor, seconds);
    for (int v = 0; v < NVARIANTS; v++) {
        uint8_t pk2[32];
        if (VARIANTS[v]->init(scalar, pk2) != 1 || memcmp(pk2, pk, 32) != 0) { fprintf(stderr, "  %-12s: init failed, skipped\n", VARIANTS[v]->name); continue; }
        for (auto &g : geo) {
            double r = run_bench(VARIANTS[v], pk, dev, fixedBlocks, g[0], g[1], seconds, false);
            fprintf(stderr, "  %-12s %3dx%d: %.3f MH/s%s\n", VARIANTS[v]->name, g[0], g[1], r, r > bestRate ? "  <- best so far" : "");
            if (r > bestRate) { bestRate = r; best = GpuCfg{dev, fixedBlocks, g[0], g[1], VARIANTS[v]}; }
        }
    }
    fprintf(stderr, "autotune GPU %d: %s %dx%d (%.3f MH/s)\n", dev, best.ops->name, best.threads, best.iters, bestRate);
    return best;
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
        "  --proxi URL        job server of the pool or of `proxi node mine` (e.g. http://proxi.suprnova.cc:8100)\n"
        "  --token T          bearer token (pool registration / mine.seeker.token)\n"
        "  --key-file FILE    unencrypted proxima.key (or --seed-hex HEX / NONCE_SEEKER_SEED)\n"
        "  --seed-hex HEX     32-byte seed\n"
        "  --name NAME        seeker name in the pool's statistics (default ProxiChorizo-<pid>)\n"
        "  --api-bind [H:]P   stats API, JSON at http://H:P/ and the mmpOS line at /mmpos (default off; H defaults to 127.0.0.1)\n"
        "  --gpus 0,1         devices to use (default all; indices as CUDA orders them, see CUDA_DEVICE_ORDER)\n"
        "  --variant NAME     kernel variant for all GPUs (default: per-platform table; --list-variants)\n"
        "  --autotune [SECS]  measure every variant on each GPU first and use the fastest (default 6s per trial)\n"
        "  --blocks N --threads N --iters N   launch geometry (defaults from the platform table)\n"
        "  --selftest FILE    verify every variant against vectors.json (from vrfvec) and exit\n"
        "  --fetest           randomized field-arithmetic test of every variant and exit\n"
        "  --verify           every variant over one nonce range at low difficulty must find identical hits; exit\n"
        "  --bench [SECS]     measure the hashrate of the chosen variant and exit\n");
}

int main(int argc, char **argv) {
    std::string proxi, token, keyfile, seedhex, name, selftest, gpus, variant, apiBind;
    sock_init();
    int blocks = 0, threads = 0, iters = 0; double bench = 0, autotuneSecs = 0; bool fetest = false, listVariants = false, verify = false;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&](std::string &dst) { if (i + 1 < argc) dst = argv[++i]; };
        auto optnum = [&](double dflt) { if (i + 1 < argc && isdigit((unsigned char)argv[i + 1][0])) return atof(argv[++i]); return dflt; };
        if (a == "--proxi") next(proxi);
        else if (a == "--token") next(token);
        else if (a == "--key-file") next(keyfile);
        else if (a == "--seed-hex") next(seedhex);
        else if (a == "--name") next(name);
        else if (a == "--api-bind" || a == "--api-port") next(apiBind);
        else if (a == "--gpus") next(gpus);
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
        else { usage(); return 1; }
    }
    if (listVariants) { for (int v = 0; v < NVARIANTS; v++) fprintf(stderr, "%s\n", VARIANTS[v]->name); return 0; }
    int ndev = 0; CUDA_CHECK(cudaGetDeviceCount(&ndev));
    if (ndev == 0) { fprintf(stderr, "no CUDA devices\n"); return 2; }
    std::vector<int> devs;
    if (gpus.empty()) for (int i = 0; i < ndev; i++) devs.push_back(i);
    else { std::stringstream ss(gpus); std::string t; while (std::getline(ss, t, ',')) devs.push_back(atoi(t.c_str())); }

    uint8_t seed[32], scalar[32] = {0}, pk[32] = {0};
    bool keyGiven = !seedhex.empty() || !keyfile.empty() || getenv("NONCE_SEEKER_SEED") != nullptr;
    bool have_key = (!selftest.empty() || fetest || verify || !keyGiven) ? false : load_seed(seedhex, keyfile, seed);
    if (keyGiven && !have_key && selftest.empty() && !fetest && !verify) return 1;
    if (!have_key && selftest.empty() && !fetest && !verify && bench == 0 && autotuneSecs == 0) { usage(); return 1; }
    if (!have_key) {
        // bench/tests without a key: a fixed pseudo-random seed, so the scalar
        // multiplication costs what a real key costs (an all-zero scalar skips every addition)
        const char *fake = "ProxiChorizo bench scalar";
        host_sha512((const uint8_t *)fake, strlen(fake), seed);
    }
    { uint8_t h[64]; host_sha512(seed, 32, h); memcpy(scalar, h, 32); scalar[0] &= 248; scalar[31] &= 127; scalar[31] |= 64; memcpy(g_scalar, scalar, 32); }

    const VariantOps *forced = nullptr;
    if (!variant.empty() && variant != "all") { forced = variantByName(variant); if (!forced) { fprintf(stderr, "unknown variant %s (see --list-variants)\n", variant.c_str()); return 1; } }

    if (verify) {
        CUDA_CHECK(cudaSetDevice(devs[0]));
        uint8_t pk2[32]; if (VARIANTS[0]->init(scalar, pk2) != 1) { fprintf(stderr, "init failed\n"); return 2; }
        return run_verify(pk2, devs[0]);
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
        GpuCfg cfg{d, blocks, threads > 0 ? threads : pl.threads, iters > 0 ? iters : pl.iters, forced ? forced : variantByName(pl.variant)};
        cfg.ordinal = (int)cfgs.size();
        if (!cfg.ops) cfg.ops = VARIANTS[0];
        uint8_t pk2[32]; int ok = cfg.ops->init(scalar, pk2);
        if (ok != 1) { fprintf(stderr, "device %d: constant derivation failed on %s\n", d, cfg.ops->name); return 2; }
        if (d == devs[0]) memcpy(pk, pk2, 32); else if (memcmp(pk, pk2, 32) != 0) { fprintf(stderr, "device %d derives a different public key\n", d); return 2; }
        if (autotuneSecs > 0) { cfg = autotune(d, scalar, pk, autotuneSecs, blocks); if (threads > 0) cfg.threads = threads; if (iters > 0) cfg.iters = iters; cfg.ops->init(scalar, pk2); }
        else fprintf(stderr, "GPU %d %s (sm_%d%d): %s, %dx%d — %s\n", d, prop.name, prop.major, prop.minor, cfg.ops->name, cfg.threads, cfg.iters, pl.note);
        cfgs.push_back(cfg);
    }
    if (bench > 0) { for (auto &c : cfgs) run_bench(c.ops, pk, c.dev, c.blocks, c.threads, c.iters, bench, true); return 0; }
    if (proxi.empty()) { if (autotuneSecs > 0) return 0; usage(); return 1; }

    Shared sh;
    sh.base_url = proxi; if (!parse_url(proxi, sh.url)) return 1;
    sh.token = token; sh.name = name.empty() ? "proxichorizo-" + std::to_string(getpid()) : name;
    memcpy(sh.pk, pk, 32); memcpy(sh.scalar, scalar, 32);
    sh.gpus = gstats; sh.api_bind = apiBind;
    logln("ProxiChorizo " PX_VERSION ": key " + hexs(pk, 32) + ", job server " + sh.url.host + ":" + std::to_string(sh.url.port) + ", " + std::to_string(devs.size()) + " GPU(s), name " + sh.name);
    std::vector<std::thread> th;
    th.emplace_back(poller, std::ref(sh));
    th.emplace_back(reporter, std::ref(sh));
    if (!apiBind.empty()) th.emplace_back(api_server, std::ref(sh));
    for (auto &c : cfgs) th.emplace_back(gpu_worker, std::ref(sh), c);
    for (auto &t : th) t.join();
    return 0;
}
