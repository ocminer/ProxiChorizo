#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
typedef uint8_t uchar; typedef uint32_t uint; typedef uint64_t ulong;
#define __kernel
#define __global
#define __constant static const
#define __attribute__(x)
static inline size_t get_global_id(int d) { (void)d; return 0; }
static inline uint atomic_inc(uint *p) { return (*p)++; }
static inline ulong mul_hi(ulong a, ulong b) { return (ulong)(((unsigned __int128)a * b) >> 64); }
static inline int clz(ulong x) { return x ? __builtin_clzll(x) : 64; }
#define PX_HOST_REF 1
static inline size_t get_global_size(int d) { (void)d; return 1; }
