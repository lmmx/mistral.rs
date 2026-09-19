// Index and decode math for PTQ1_0 (Prism ternary, group 128); host-compilable
// so it can be tested without a GPU.
#pragma once

#include <cstdint>

#ifdef __CUDACC__
#define PTQ1_0_HD __host__ __device__ __forceinline__
#else
#define PTQ1_0_HD inline
#endif

namespace ptq1_0 {

constexpr int BLOCK_ELEMS = 128;
constexpr int BLOCK_BYTES = 28;
constexpr int SCALE_OFFSET = 26;
constexpr int WIDE_LANES = 16;
constexpr int NARROW_LANES = 8;
constexpr int NARROW_START = 80;
constexpr int QH_START = 120;
constexpr int ACTIVE_LANES = 26; // lane index == byte index within qs then qh
constexpr int FWHT_BLOCK = 1024;

struct LaneMap {
  int elem_base;
  int elem_stride;
  int trits;
};

PTQ1_0_HD LaneMap lane_map(int lane) {
  if (lane < WIDE_LANES) {
    return {lane, 16, 5};
  }
  if (lane < WIDE_LANES + NARROW_LANES) {
    return {NARROW_START + (lane - WIDE_LANES), 8, 5};
  }
  return {QH_START + (lane - WIDE_LANES - NARROW_LANES), 2, 4};
}

// Next base-3 digit of a stored byte as a weight in -1..1; `v` carries the
// remainder.
PTQ1_0_HD int next_trit(uint32_t &v) {
  const uint32_t p = v * 3u;
  v = p & 255u;
  return static_cast<int>(p >> 8) - 1;
}

// Lower index of butterfly pair `pair` at stride `h` (h a power of two).
PTQ1_0_HD int fwht_low_index(int pair, int h) {
  return ((pair & ~(h - 1)) << 1) | (pair & (h - 1));
}

#ifdef __CUDACC__
static __device__ __forceinline__ float block_scale(const uint8_t *blk) {
  return __half2float(__ushort_as_half(
      *reinterpret_cast<const unsigned short *>(blk + SCALE_OFFSET)));
}

// Adds one block's contribution for this lane; `xs` holds the TT activation
// rows.
template <int TT>
static __device__ __forceinline__ void
accumulate_block(uint32_t byte, float d, int b, const LaneMap &map, bool active,
                 const float *const *xs, float *acc) {
  if (!active) {
    return;
  }
  const int e0 = b * BLOCK_ELEMS + map.elem_base;
  float part[TT] = {};
  for (int n = 0; n < map.trits; ++n) {
    const float trit = static_cast<float>(next_trit(byte));
    const int e = e0 + n * map.elem_stride;
#pragma unroll
    for (int t = 0; t < TT; ++t) {
      part[t] += trit * __ldg(xs[t] + e);
    }
  }
#pragma unroll
  for (int t = 0; t < TT; ++t) {
    acc[t] += d * part[t];
  }
}

static __device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    x += __shfl_xor_sync(0xffffffff, x, mask, 32);
  }
  return x;
}
#endif

} // namespace ptq1_0
