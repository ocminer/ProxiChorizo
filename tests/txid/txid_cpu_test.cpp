// CPU test of txid_blake2b.h: the reference hash against the upgrade-6 vectors and Python hashlib cases,
// and the GPU algorithm (per-job precomputation + txid_h3) emulated on the CPU against the reference.
// Build: g++ -O2 -std=c++17 -I../.. -o txid_cpu_test txid_cpu_test.cpp
// Run:   python3 gen_fuzz.py > fuzz.txt && ./txid_cpu_test ../../vectors-txid-blake2b.json fuzz.txt
#include "txid_blake2b.h"
#include <cstdio>
#include <fstream>
#include <sstream>
#include <random>
using namespace txid;
static std::string hx(const uint8_t *p, size_t n) { static const char *h = "0123456789abcdef"; std::string s; for (size_t i = 0; i < n; i++) { s += h[p[i] >> 4]; s += h[p[i] & 15]; } return s; }
static std::vector<uint8_t> unhx(const std::string &s) { std::vector<uint8_t> o; for (size_t i = 0; i + 1 < s.size(); i += 2) o.push_back((uint8_t)strtoul(s.substr(i, 2).c_str(), nullptr, 16)); return o; }
static std::string jget(const std::string &obj, const std::string &key, size_t from = 0) {
    size_t p = obj.find("\"" + key + "\"", from); if (p == std::string::npos) return "";
    p = obj.find(':', p) + 1; while (obj[p] == ' ') p++;
    if (obj[p] == '"') { size_t e = obj.find('"', p + 1); return obj.substr(p + 1, e - p - 1); }
    size_t e = obj.find_first_of(",}\n", p); return obj.substr(p, e - p);
}
int main(int argc, char **argv) {
    int bad = 0, n = 0;
    if (argc > 1) {
        std::ifstream f(argv[1]); std::stringstream ss; ss << f.rdbuf(); std::string js = ss.str();
        size_t pos = 0;
        while ((pos = js.find("\"prefix_len\"", pos)) != std::string::npos) {
            size_t end = js.find("\"prefix_len\"", pos + 1); std::string obj = js.substr(pos, end == std::string::npos ? std::string::npos : end - pos); pos += 10;
            int plen = atoi(jget(obj, "prefix_len").c_str());
            std::vector<uint8_t> prefix(plen); for (int i = 0; i < plen; i++) prefix[i] = (uint8_t)((i * 7 + 3) & 0xff);
            uint64_t en1 = strtoull(jget(obj, "extranonce1").c_str(), nullptr, 16), en2 = strtoull(jget(obj, "extranonce2").c_str(), nullptr, 16);
            uint64_t N = en1 << 48 | en2; uint8_t h[32]; hash_nonce(prefix, N, h);
            size_t fp = obj.find("first_en2_with_tz_ge_12");
            std::string want = jget(obj, "hash"), fen2 = jget(obj, "extranonce2", fp), fhash = jget(obj, "hash", fp);
            int tz = atoi(jget(obj, "trailing_zeros").c_str());
            bool ok = hx(h, 32) == want && work_bits(h) == tz;
            // the GPU algorithm on the CPU, and the first en2 >= 0 with >= 12 zero bits
            Params p; Shape s = prepare(prefix.data(), plen, 12, p);
            ok = ok && emulate_h3(p, N) == load64le(h + 24);
            uint64_t first = ~0ULL;
            for (uint64_t e = 0; e < (1u << 20); e++) { uint64_t h3 = emulate_h3(p, en1 << 48 | e); if ((h3 & p.mask) == 0) { first = e; break; } }
            uint8_t fh[32]; hash_nonce(prefix, en1 << 48 | first, fh);
            bool fok = first == strtoull(fen2.c_str(), nullptr, 16) && hx(fh, 32) == fhash;
            printf("vector len %3d (kernel %2d%s): hash %s, first en2 %012llx %s\n", plen + 8, s.kernel, s.straddle ? " straddle" : "", ok ? "ok" : "MISMATCH", (unsigned long long)first, fok ? "ok" : "MISMATCH");
            if (!ok || !fok) bad++;
            n++;
        }
    }
    if (argc > 2) {
        std::ifstream f(argv[2]); std::string pre, non, hh; int cases = 0, fails = 0;
        std::mt19937_64 rng(7);
        while (f >> pre >> non >> hh) {
            if (pre == "-") pre.clear();
            std::vector<uint8_t> prefix = unhx(pre); uint64_t N = strtoull(non.c_str(), nullptr, 16);
            uint8_t h[32]; hash_nonce(prefix, N, h);
            bool ok = hx(h, 32) == hh;
            Params p; prepare(prefix.data(), (int)prefix.size(), 0, p);
            ok = ok && emulate_h3(p, N) == load64le(h + 24);
            for (int r = 0; r < 4 && ok; r++) { uint64_t M = rng(); uint8_t h2[32]; hash_nonce(prefix, M, h2); ok = emulate_h3(p, M) == load64le(h2 + 24); }
            if (!ok) { fails++; if (fails < 10) printf("fuzz FAIL prefix_len %zu nonce %s\n", prefix.size(), non.c_str()); }
            cases++;
        }
        printf("fuzz: %d cases (hashlib reference + emulated GPU algorithm), %d failed\n", cases, fails);
        bad += fails;
    }
    printf("%s: %d vectors, %d failures\n", bad ? "FAIL" : "PASS", n, bad);
    return bad ? 1 : 0;
}
