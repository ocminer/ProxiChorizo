// CPU-only regression: tuning for measured cards must not change adjacent models.
#define main proxichorizo_cli_main
#include "../proxichorizo_cl.cpp"
#undef main
#include <cassert>

int main() {
    struct Case { const char *gfx, *board; unsigned cus; bool optimized; const char *variant = "fe29_wnaf_c"; };
    const Case cases[] = {
        {"gfx1100", "AMD Radeon RX 7900 XTX", 48, true, "fe29_mlsp_cf3_c"},
        {"gfx1100", "AMD Radeon RX 7900 XT", 42, false},
        {"gfx1100", "", 48, false},
        {"gfx1102", "AMD Radeon RX 7600 XT", 16, true, "fe29_mlsp_kara_c"},
        {"gfx1102", "AMD Radeon RX 7600", 16, false},
        {"gfx906:sramecc+:xnack-", "AMD Instinct MI50/MI60", 60, true, "fe29_mlsp_c"},
        {"gfx906:sramecc+:xnack-", "AMD Instinct MI50/MI60", 64, false},
        {"gfx90a", "AMD Instinct MI210", 104, false},
        {"gfx1201", "AMD Radeon RX 9070 XT", 32, false},
    };
    for (const auto &c : cases) {
        ClDev d; d.name = c.gfx; d.board = c.board; d.cus = c.cus;
        const Platform &p = platformFor(d);
        assert(std::string(p.variant) == c.variant);
        assert(p.measured == c.optimized);
        assert(p.iters == (c.optimized ? 1 : 8));
        assert(p.groups_per_cu == (c.optimized ? 128 : 8));
    }
    ClDev a; a.name="gfx1100"; a.board="RX 7900 XT"; a.cus=42; a.tuning_id="uuid-A"; a.driver="runtime-A";
    ClDev b=a; b.tuning_id="uuid-B"; assert(tuneKey(a)!=tuneKey(b));
    b=a; b.cus=48; assert(tuneKey(a)!=tuneKey(b));
    b=a; b.driver="runtime-B"; assert(tuneKey(a)!=tuneKey(b));
    puts("OpenCL model profiles: 9 cases passed (CPU only)");
}
