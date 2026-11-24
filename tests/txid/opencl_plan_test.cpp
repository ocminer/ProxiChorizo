// GPU regression test: selected-card context, exact nonce windows and failed events.
// Run gen_host.py, then build with the same OpenCL headers/runtime as the miner.
#define main proxichorizo_program_main
#include "../../opencl/proxichorizo_cl.cpp"
#undef main
#include <set>
int main(int argc, char **argv) {
    if (argc != 2) return 2;
    const int bus = atoi(argv[1]);
    enumerate_devices();
    int selected = -1;
    for (size_t i = 0; i < g_devs.size(); ++i) if (g_devs[i].bus == bus) {
        if (selected >= 0) return 2;
        selected = (int)i;
    }
    if (selected < 0) return 2;
    ClDev &d = g_devs[selected]; cl_int err;
    d.ctx = clCreateContext(nullptr, 1, &d.id, nullptr, nullptr, &err); CL_CHECK(err);
    d.q = clCreateCommandQueue(d.ctx, d.id, 0, &err); CL_CHECK(err); configure_wait(d);
    ClTxSearch search(d); search.tp->use_plan29 = true;
    std::string why;
    if (txid::selfcheck(std::ref(search), why)) { fprintf(stderr, "%s\n", why.c_str()); return 1; }
    size_t cases = 0, checked = 0;
    for (int blocks : {0,2,5}) for (int offset = 113; offset <= 127; ++offset)
    for (uint64_t first : {0xa1b27856fffffe01ULL,0xfffffffffffffe01ULL})
    for (int bits : {0,3,63}) {
        if (offset == 120) continue; // covered separately by zero_tail_opencl_test
        // Span more than one shared straddle tile, including an unaligned
        // first nonce and the last partial group. Keep zero-target runs short
        // enough to compare every hit with the bounded device output buffer.
        const uint64_t count = offset > 120 && bits ? 8195 : 1025;
        std::vector<uint8_t> prefix(blocks * 128 + offset);
        for (size_t i = 0; i < prefix.size(); ++i) prefix[i] = (uint8_t)(i*37 + offset);
        txid::Params p; txid::prepare(prefix.data(), (int)prefix.size(), bits, p);
        auto got = search(p, first, count);
        std::set<uint64_t> expected, seen;
        for (uint64_t i = 0; i < count; ++i) {
            uint8_t h[32]; txid::hash_nonce(prefix,first+i,h);
            if (txid::work_bits(h) >= bits) expected.insert(first+i);
        }
        for (auto hit : got) {
            uint8_t h[32]; txid::hash_nonce(prefix,hit.n,h);
            if (hit.h3 != txid::load64le(h+24) || !seen.insert(hit.n).second) return 1;
            printf("{\"prefix\":\"%s\",\"nonce\":\"%016llx\",\"h3\":\"%016llx\"}\n",
                   hexs(prefix.data(),prefix.size()).c_str(),(unsigned long long)hit.n,(unsigned long long)hit.h3);
        }
        if (seen != expected) return 1;
        ++cases; checked += count;
    }
    for (int length : {369,377,383}) {
        std::vector<uint8_t> prefix(length, 0x93);
        txid::Params p;
        txid::prepare(prefix.data(), length, 0, p);
        if (!search(p,0,0).empty()) return 1;
        const uint64_t first = 0xfffffffffffff801ULL;
        auto overflow = search(p, first, 5000);
        cl_uint hits = 0;
        CL_CHECK(clEnqueueReadBuffer(d.q, search.nhit, CL_TRUE, 0, sizeof hits, &hits, 0, nullptr, nullptr));
        if (hits != 5000 || overflow.size() != TXID_MAX_HITS) return 1;
        std::set<uint64_t> distinct;
        for (auto hit : overflow) {
            uint8_t h[32];
            txid::hash_nonce(prefix, hit.n, h);
            if (hit.n - first >= 5000 || hit.h3 != txid::load64le(h+24) || !distinct.insert(hit.n).second) return 1;
        }
    }
    // A successful result read must not hide an earlier failed dispatch event.
    for (bool read : {false, true}) for (cl_int status : {CL_COMPLETE, -42}) {
        if (read) read_wait_queues.insert(d.q); else read_wait_queues.erase(d.q);
        cl_event event = clCreateUserEvent(d.ctx,&err); CL_CHECK(err);
        CL_CHECK(clSetUserEventStatus(event,status)); mining_events[d.q].push_back(event);
        if (mining_wait(d.q,search.nhit) != status || !mining_events[d.q].empty()) return 1;
    }
    if (mining_wait(d.q,nullptr) != CL_INVALID_MEM_OBJECT) return 1;
    fprintf(stderr,"PASS %zu boundary/target cases, %zu candidate nonces, zero/overflow and negative-event propagation\n",cases,checked);
    return 0;
}
