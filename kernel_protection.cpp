// ProxiChorizo License 1.0; see LICENSE.
// Authenticated, in-memory kernel loading. No decrypted file is written.
#include "kernel_protection.h"
#include "third_party/monocypher/monocypher.h"
#include "kernel_key.h" // generated locally, never packaged or committed
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <mutex>

struct PxPayload {
    uint64_t size;
    unsigned char nonce[24], mac[16];
    uint32_t kind, reserved[3];
};
static_assert(sizeof(PxPayload) == 64, "payload format");
struct PxManifest {
    char magic[32];
    uint32_t count, version;
    PxPayload payloads[16];
};
// The sealing tool fills this reserved table after linking. Volatile prevents
// constant folding of the unsealed placeholder, on both MSVC and GCC.
static volatile PxManifest px_manifest = {"PX_KERNEL_MANIFEST_V1_20261115", 0, 1, {}};
[[noreturn]] static void protection_error() {
    std::fputs("Kernel protection check failed; reinstall an official release.\n", stderr);
    std::exit(3);
}
void px_protection_check() {
    if (px_manifest.version != 1 || !px_manifest.count || px_manifest.count > 16)
        protection_error();
}
static void open_payload(unsigned char *out, const unsigned char *in, size_t n, unsigned kind) {
    px_protection_check();
    PxPayload item{}; unsigned matches = 0;
    for (unsigned i = 0; i < px_manifest.count; ++i) {
        const volatile PxPayload &p = px_manifest.payloads[i];
        if (p.size != n || p.kind != kind) continue;
        ++matches;
        for (unsigned j = 0; j < 24; ++j) item.nonce[j] = p.nonce[j];
        for (unsigned j = 0; j < 16; ++j) item.mac[j] = p.mac[j];
    }
    if (matches != 1 || !n || n > (1ULL << 30)) protection_error();
    unsigned char key[32], ad[12];
    for (unsigned i = 0; i < 32; ++i) key[i] = px_key_a[i] ^ px_key_b[i];
    for (unsigned i = 0; i < 8; ++i) ad[i] = (unsigned char)((uint64_t)n >> (8*i));
    for (unsigned i = 0; i < 4; ++i) ad[8+i] = (unsigned char)(kind >> (8*i));
    int rc = crypto_aead_unlock(out, item.mac, key, item.nonce, ad, sizeof ad, in, n);
    crypto_wipe(key, sizeof key);
    if (rc) protection_error();
}
std::string px_open_kernel(const unsigned char *in, size_t n, unsigned kind) {
    std::string out(n, '\0');
    open_payload(reinterpret_cast<unsigned char *>(&out[0]), in, n, kind);
    return out;
}

#ifdef PX_CUDA_PROTECTION
// CUDA's host registration ABI. The normal runtime still owns function,
// variable and context registration. Keep each decoded image until its
// matching unregister call, including static-constructor/atexit ordering.
extern "C" void **__cudaRegisterFatBinary(void *);
extern "C" void __cudaUnregisterFatBinary(void **);
struct PxCudaWrapper { int32_t magic, version; const void *data; void *filename; };
struct PxCudaImage { PxCudaWrapper wrapper; unsigned char *bytes; size_t size; void **handle; PxCudaImage *next; };
static PxCudaImage *images = nullptr;
static std::mutex &image_mutex() { static std::mutex m; return m; }
extern "C" void **px_register_fatbin(void *raw) {
    auto *source = static_cast<const PxCudaWrapper *>(raw);
    if (!source || source->magic != 0x466243b1 || source->version != 1 || !source->data)
        protection_error();
    const auto *p = static_cast<const unsigned char *>(source->data);
    uint32_t magic; uint16_t version, header; uint64_t n;
    std::memcpy(&magic, p, 4); std::memcpy(&version, p+4, 2);
    std::memcpy(&header, p+6, 2); std::memcpy(&n, p+8, 8);
    if (magic != 0xba55ed50 || version != 1 || header != 16 || !n || n > (1ULL << 30))
        protection_error();
    auto *image = static_cast<PxCudaImage *>(std::calloc(1, sizeof(PxCudaImage)));
    if (!image) protection_error();
    image->size = (size_t)n + header;
    image->bytes = static_cast<unsigned char *>(std::malloc(image->size));
    if (!image->bytes) protection_error();
    std::memcpy(image->bytes, p, header);
    open_payload(image->bytes + header, p + header, (size_t)n, 1);
    image->wrapper = *source; image->wrapper.data = image->bytes;
    image->handle = __cudaRegisterFatBinary(&image->wrapper);
    if (!image->handle) protection_error();
    std::lock_guard<std::mutex> lock(image_mutex());
    image->next = images; images = image;
    return image->handle;
}
extern "C" void px_unregister_fatbin(void **handle) {
    __cudaUnregisterFatBinary(handle);
    std::lock_guard<std::mutex> lock(image_mutex());
    PxCudaImage **link = &images;
    while (*link && (*link)->handle != handle) link = &(*link)->next;
    if (!*link) protection_error();
    PxCudaImage *image = *link; *link = image->next;
    crypto_wipe(image->bytes, image->size); std::free(image->bytes); std::free(image);
}
#endif
