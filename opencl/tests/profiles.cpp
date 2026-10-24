// CPU-only regression: tuning for measured cards must not change adjacent models.
#define main proxichorizo_cli_main
#include "../proxichorizo_cl.cpp"
#undef main
#include <cassert>

int main() {
    struct Case { const char *gfx, *board; unsigned cus; bool optimized; };
    const Case cases[] = {
        {"gfx1100", "AMD Radeon RX 7900 XTX", 48, true},
        {"gfx1100", "AMD Radeon RX 7900 XT", 42, false},
        {"gfx1100", "", 48, false},
        {"gfx1102", "AMD Radeon RX 7600 XT", 16, true},
        {"gfx1102", "AMD Radeon RX 7600", 16, false},
        {"gfx906:sramecc+:xnack-", "AMD Instinct MI50/MI60", 60, true},
        {"gfx906:sramecc+:xnack-", "AMD Instinct MI50/MI60", 64, false},
        {"gfx90a", "AMD Instinct MI210", 104, false},
        {"gfx1201", "AMD Radeon RX 9070 XT", 32, false},
    };
    for (const auto &c : cases) {
        ClDev d; d.name = c.gfx; d.board = c.board; d.cus = c.cus;
        const Platform &p = platformFor(d);
        assert(std::string(p.variant) == (c.optimized ? "fe29_wnaf_split8_c" : "fe29_wnaf_c"));
        assert(p.iters == (c.optimized ? 1 : 8));
        assert(p.groups_per_cu == (c.optimized ? 128 : 8));
    }
    puts("OpenCL model profiles: 9 cases passed (CPU only)");
}
