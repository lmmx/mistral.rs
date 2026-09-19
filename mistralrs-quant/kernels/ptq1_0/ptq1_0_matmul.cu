// Prism ternary (PTQ1_0) matmul with the Hadamard weight fold applied to activations.

#include "cuda_bf16.h"
#include "cuda_fp16.h"
#include <stdint.h>

#include "ptq1_0_math.cuh"

#define WARP_SIZE 32
#define FWHT_THREADS 256
#define MATMUL_WARPS 8
#define MATMUL_TOKENS_PER_PASS 4
#define FWHT_SCALE 0.03125f // 1 / sqrt(FWHT_BLOCK)

static __device__ __forceinline__ float to_float(float v) { return v; }
static __device__ __forceinline__ float to_float(__half v) {
  return __half2float(v);
}
static __device__ __forceinline__ float to_float(__nv_bfloat16 v) {
  return __bfloat162float(v);
}

static __device__ __forceinline__ void store_float(float *p, float v) {
  *p = v;
}
static __device__ __forceinline__ void store_float(__half *p, float v) {
  *p = __float2half(v);
}
static __device__ __forceinline__ void store_float(__nv_bfloat16 *p, float v) {
  *p = __float2bfloat16(v);
}

// One CTA per (FWHT_BLOCK columns, token). Without do_fwht it only widens to f32.
template <typename T>
static __global__ void
ptq1_0_prepare_kernel(const T *__restrict__ x, const float *__restrict__ signs,
                      const uint32_t *__restrict__ gather,
                      float *__restrict__ out, int k, int do_fwht) {
  __shared__ float s[ptq1_0::FWHT_BLOCK];
  const int tid = threadIdx.x;
  const int col0 = blockIdx.x * ptq1_0::FWHT_BLOCK;
  const size_t base = static_cast<size_t>(blockIdx.y) * k;

  for (int i = tid; i < ptq1_0::FWHT_BLOCK; i += FWHT_THREADS) {
    const int p = col0 + i;
    float v = 0.0f;
    if (p < k) {
      v = to_float(x[base + (gather ? gather[p] : p)]);
      if (signs) {
        v *= signs[p];
      }
    }
    s[i] = v;
  }
  __syncthreads();

  if (do_fwht) {
    for (int h = 1; h < ptq1_0::FWHT_BLOCK; h <<= 1) {
      for (int pair = tid; pair < ptq1_0::FWHT_BLOCK / 2;
           pair += FWHT_THREADS) {
        const int i0 = ptq1_0::fwht_low_index(pair, h);
        const float a = s[i0];
        const float b = s[i0 + h];
        s[i0] = a + b;
        s[i0 + h] = a - b;
      }
      __syncthreads();
    }
  }

  const float scale = do_fwht ? FWHT_SCALE : 1.0f;
  for (int i = tid; i < ptq1_0::FWHT_BLOCK; i += FWHT_THREADS) {
    const int p = col0 + i;
    if (p < k) {
      out[base + p] = s[i] * scale;
    }
  }
}

// One warp per output row, TT tokens per pass. Lane L owns byte L of every block, so neighbouring lanes read
// neighbouring activations.
template <typename OutT, int TT>
static __global__ void
ptq1_0_matmul_kernel(const uint8_t *__restrict__ w, const float *__restrict__ x,
                     OutT *__restrict__ dst, int ncols_x, int nrows_x,
                     int b_size) {
  const int lane = threadIdx.x % WARP_SIZE;
  const int row = blockIdx.x * MATMUL_WARPS + threadIdx.x / WARP_SIZE;
  const int t0 = blockIdx.y * TT;
  if (row >= nrows_x) {
    return;
  }
  const int nblk = ncols_x / ptq1_0::BLOCK_ELEMS;
  const uint8_t *wrow =
      w + static_cast<size_t>(row) * nblk * ptq1_0::BLOCK_BYTES;
  const ptq1_0::LaneMap map = ptq1_0::lane_map(lane);
  const bool active = lane < ptq1_0::ACTIVE_LANES;

  const float *xs[TT];
  float acc[TT];
#pragma unroll
  for (int t = 0; t < TT; ++t) {
    xs[t] = x + static_cast<size_t>(min(t0 + t, b_size - 1)) * ncols_x;
    acc[t] = 0.0f;
  }

  for (int b = 0; b < nblk; ++b) {
    const uint8_t *blk = wrow + static_cast<size_t>(b) * ptq1_0::BLOCK_BYTES;
    if (!active) {
      continue;
    }
    const float d = __half2float(__ushort_as_half(
        *reinterpret_cast<const unsigned short *>(blk + ptq1_0::SCALE_OFFSET)));
    uint32_t v = blk[lane];
    const int e0 = b * ptq1_0::BLOCK_ELEMS + map.elem_base;
    float part[TT] = {};
    for (int n = 0; n < map.trits; ++n) {
      const float trit = static_cast<float>(ptq1_0::next_trit(v));
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

#pragma unroll
  for (int t = 0; t < TT; ++t) {
    acc[t] = ptq1_0::warp_sum(acc[t]);
  }
  if (lane == 0) {
#pragma unroll
    for (int t = 0; t < TT; ++t) {
      if (t0 + t < b_size) {
        store_float(dst + static_cast<size_t>(t0 + t) * nrows_x + row, acc[t]);
      }
    }
  }
}

template <typename T>
static void ptq1_0_launch(const void *x, const void *w, const void *signs,
                          const void *gather, void *scratch, void *dst,
                          int ncols_x, int nrows_x, int b_size, int do_fwht,
                          void *stream) {
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  dim3 prep_grid((ncols_x + ptq1_0::FWHT_BLOCK - 1) / ptq1_0::FWHT_BLOCK,
                 b_size, 1);
  ptq1_0_prepare_kernel<T><<<prep_grid, FWHT_THREADS, 0, s>>>(
      static_cast<const T *>(x), static_cast<const float *>(signs),
      static_cast<const uint32_t *>(gather), static_cast<float *>(scratch),
      ncols_x, do_fwht);

  const int block = MATMUL_WARPS * WARP_SIZE;
  const unsigned int row_blocks = (nrows_x + MATMUL_WARPS - 1) / MATMUL_WARPS;
  if (b_size == 1) {
    ptq1_0_matmul_kernel<T, 1><<<dim3(row_blocks, 1, 1), block, 0, s>>>(
        static_cast<const uint8_t *>(w), static_cast<const float *>(scratch),
        static_cast<T *>(dst), ncols_x, nrows_x, b_size);
  } else {
    const unsigned int passes =
        (b_size + MATMUL_TOKENS_PER_PASS - 1) / MATMUL_TOKENS_PER_PASS;
    ptq1_0_matmul_kernel<T, MATMUL_TOKENS_PER_PASS>
        <<<dim3(row_blocks, passes, 1), block, 0, s>>>(
            static_cast<const uint8_t *>(w),
            static_cast<const float *>(scratch), static_cast<T *>(dst), ncols_x,
            nrows_x, b_size);
  }
}

// Host-side launchers used by `mistralrs-quant/src/gguf/ffi.rs`.

#define PTQ1_0_LAUNCHER(tag, c_type)                                           \
  extern "C" void launch_ptq1_0_matmul_##tag(                                  \
      const void *x, const void *w, const void *signs, const void *gather,     \
      void *scratch, void *dst, int ncols_x, int nrows_x, int b_size,          \
      int do_fwht, void *stream) {                                             \
    ptq1_0_launch<c_type>(x, w, signs, gather, scratch, dst, ncols_x,          \
                          nrows_x, b_size, do_fwht, stream);                   \
  }

PTQ1_0_LAUNCHER(f32, float)
PTQ1_0_LAUNCHER(f16, __half)
PTQ1_0_LAUNCHER(bf16, __nv_bfloat16)
