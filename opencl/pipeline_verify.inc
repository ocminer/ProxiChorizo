// Compare complete outputs against the original compacted pipeline, including
// partial workgroups, repeated work per item and nonce wraparound. Both paths
// retain the same bounded hash-to-curve retry policy.
static int run_pipeline_edges(const uint8_t scalar[32], const uint8_t pk[32], int dev) {
    ClDev &d = g_devs[dev]; cl_int err;
    const int sizes[] = {1,2,7,8,9,15,16,17,31,32,33,63,64,65,127,128,129,
                        255,256,257,511,512,513,767,1023,1024,1025,2047,2048,
                        2049,4095,4096,4097,8191,8192};
    const size_t count = sizeof sizes / sizeof sizes[0];
    std::vector<const Variant *> variants{variantByName("fe29_wnaf_c")};
    for (const auto &v : VARIANTS) if (v.split) variants.push_back(&v);
    std::vector<std::vector<std::pair<uint64_t, std::string>>> expected(count * 2);
    cl_mem hitsbuf = clCreateBuffer(d.ctx, CL_MEM_READ_WRITE, sizeof(Hit) * MAX_HITS, nullptr, &err); CL_CHECK(err);
    cl_mem nbuf = clCreateBuffer(d.ctx, CL_MEM_READ_WRITE, sizeof(cl_uint), nullptr, &err); CL_CHECK(err);
    int rc = 0;
    for (size_t v = 0; v < variants.size(); v++) {
        Prog *pr = get_prog(d, variants[v]); uint8_t pk2[32];
        if (prog_init(pr, scalar, pk2) != 1 || memcmp(pk, pk2, 32)) { rc = 1; break; }
        size_t total = 0;
        for (size_t c = 0; c < count * 2; c++) {
            size_t nitems = sizes[c % count];
            JobDev jd{}; jd.prefix_len = 37; jd.k = 0;
            memcpy(jd.pk, pk, 32); memset(jd.beat, 255, sizeof jd.beat);
            for (int i = 0; i < 37; i++) jd.prefix[i] = (uint8_t)(i * 31 + 7 + c * 19);
            cl_mem job = make_job(d, jd); cl_uint zero = 0;
            CL_CHECK(clEnqueueFillBuffer(d.q, nbuf, &zero, sizeof zero, 0, sizeof zero, 0, nullptr, nullptr));
            Scratch *sc = scratch_alloc(pr, nitems);
            uint64_t stride = c < count ? ((nitems + 255) / 256) * 256 : 256;
            uint64_t base = c % 3 ? 0xabcdef0000ULL + c * 1009 : UINT64_MAX - nitems / 2;
            prog_mine(pr, job, base, stride, 1, hitsbuf, nbuf, sc);
            cl_uint n = 0; CL_CHECK(clEnqueueReadBuffer(d.q, nbuf, CL_TRUE, 0, sizeof n, &n, 0, nullptr, nullptr));
            if (n > MAX_HITS) {
                fprintf(stderr, "[%s] pipeline verification output overflow\n", pr->v->name);
                rc = 1;
            } else {
                std::vector<Hit> hits(n);
                if (n) CL_CHECK(clEnqueueReadBuffer(d.q, hitsbuf, CL_TRUE, 0, n * sizeof(Hit), hits.data(), 0, nullptr, nullptr));
                std::vector<std::pair<uint64_t, std::string>> found;
                for (const auto &h : hits) found.push_back({h.nonce, hexs(h.beta, sizeof h.beta)});
                std::sort(found.begin(), found.end()); total += n;
                if (!v) expected[c] = found;
                else if (found != expected[c]) {
                    fprintf(stderr, "[%s] pipeline case %zu (%zu items) DIFFERENT\n", pr->v->name, c, nitems);
                    rc = 1;
                }
            }
            scratch_free(sc); clReleaseMemObject(job);
        }
        fprintf(stderr, "[%s] dense pipeline: %zu shapes, %zu complete outputs %s\n",
                pr->v->name, count * 2, total, v ? "compared" : "reference");
        if (!total) rc = 1;
    }
    clReleaseMemObject(hitsbuf); clReleaseMemObject(nbuf);
    return rc;
}
