// ProxiChorizo License 1.0; see LICENSE.
#pragma once
#include <cstddef>
#include <string>
#ifdef PX_KERNEL_PROTECTION
void px_protection_check();
std::string px_open_kernel(const unsigned char *, size_t, unsigned);
#else
inline void px_protection_check() {}
inline std::string px_open_kernel(const unsigned char *p, size_t n, unsigned) {
    return std::string(reinterpret_cast<const char *>(p), n);
}
#endif
