#include "shim.h"
#include "proxichorizo.cl"   // found via -I.. (the real kernel source, unmodified)
static void hex(const char *n, const u8 *b, int l) { printf("%s ", n); for (int i = 0; i < l; i++) printf("%02x", b[i]); printf("\n"); }
int main(void) {
    for (int sd = 1; sd <= 4; sd++) { int fail = -1; u8 dump[128] = {0}; Consts K; memset(&K, 0, sizeof K);
        fe_test_kernel(sd, 2000, &fail, dump, &K);
        printf("fetest seed %d -> code %d iter %d\n", sd, fail / 1000000, fail % 1000000);
        if (fail) { hex(" x", dump, 32); hex(" y", dump + 32, 32); hex(" a", dump + 64, 32); hex(" b", dump + 96, 32); break; } }
    u8 scalar[32] = {0}; scalar[0] = 9; u8 pk[32] = {0}; int ok = -9; Consts K; memset(&K, 0, sizeof K);
    init_kernel(scalar, pk, &ok, &K);
    printf("init ok=%d\n", ok); hex(" pk/diag", pk, 32);
    return 0;
}
