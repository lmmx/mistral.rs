// Prism ternary (PTQ1_0) matmul with the Hadamard weight fold applied to
// activations. Activations are quantized to int8 (one scale per weight block)
// so the dot product runs on dp4a.

#include "cuda_bf16.h"
#include "cuda_fp16.h"
#include <stdint.h>

#include "ptq1_0_math.cuh"

#define WARP_SIZE 32
#define PREPARE_THREADS 256
#define MATMUL_WARPS 8
#define MATMUL_TOKENS_PER_PASS 8
#define MATMUL_PREFETCH 2
#define BLOCKS_PER_WARP_STEP 4 // 4 blocks x 7 words fill 28 of 32 lanes
#define BLOCKS_PER_ITER (BLOCKS_PER_WARP_STEP * MATMUL_PREFETCH)
#define SCALE_WORD (ptq1_0::BLOCK_WORDS - 1) // scale is in its top half
#define FWHT_SCALE 0.03125f    // 1 / sqrt(FWHT_BLOCK)
#define INT8_MAX_F 127.0f

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

// One CTA per (FWHT_BLOCK columns, token): gather, signs and FWHT in shared
// memory, then int8 quantization with one scale per 128 columns. Without
// do_fwht it only quantizes.
template <typename T>
static __global__ void
ptq1_0_prepare_kernel(const T *__restrict__ x, const float *__restrict__ signs,
                      const uint32_t *__restrict__ gather,
                      int8_t *__restrict__ xq, float *__restrict__ xscale,
                      int k, int do_fwht) {
  __shared__ float s[ptq1_0::FWHT_BLOCK];
  const int tid = threadIdx.x;
  const int col0 = blockIdx.x * ptq1_0::FWHT_BLOCK;
  const size_t base = static_cast<size_t>(blockIdx.y) * k;

  for (int i = tid; i < ptq1_0::FWHT_BLOCK; i += PREPARE_THREADS) {
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
           pair += PREPARE_THREADS) {
        const int i0 = ptq1_0::fwht_low_index(pair, h);
        const float a = s[i0];
        const float b = s[i0 + h];
        s[i0] = a + b;
        s[i0 + h] = a - b;
      }
      __syncthreads();
    }
  }

  // Each warp quantizes one 128-column segment, four columns per lane.
  const int warp = tid / WARP_SIZE;
  const int lane = tid % WARP_SIZE;
  const int seg = col0 + warp * ptq1_0::BLOCK_ELEMS;
  if (seg < k) {
    const float scale = do_fwht ? FWHT_SCALE : 1.0f;
    float v[4];
    float amax = 0.0f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      v[j] = s[warp * ptq1_0::BLOCK_ELEMS + lane * 4 + j] * scale;
      amax = fmaxf(amax, fabsf(v[j]));
    }
    amax = ptq1_0::warp_max(amax);
    const float inv = amax > 0.0f ? INT8_MAX_F / amax : 0.0f;
    uint32_t packed = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      packed |= (static_cast<uint32_t>(__float2int_rn(v[j] * inv)) & 0xFFu)
                << (8 * j);
    }
    reinterpret_cast<uint32_t *>(xq + base + seg)[lane] = packed;
    if (lane == 0) {
      xscale[static_cast<size_t>(blockIdx.y) * (k / ptq1_0::BLOCK_ELEMS) +
             seg / ptq1_0::BLOCK_ELEMS] = amax / INT8_MAX_F;
    }
  }
}

// One warp per output row. Lane L reads word L % 7 of block (step * 4 + L / 7),
// so a warp step loads 112 contiguous bytes.
template <typename OutT, int TT>
static __global__ void
ptq1_0_matmul_kernel(const uint32_t *__restrict__ w,
                     const int8_t *__restrict__ xq,
                     const float *__restrict__ xscale, OutT *__restrict__ dst,
                     int ncols_x, int nrows_x, int b_size) {
  const int lane = threadIdx.x % WARP_SIZE;
  const int row = blockIdx.x * MATMUL_WARPS + threadIdx.x / WARP_SIZE;
  const int t0 = blockIdx.y * TT;
  if (row >= nrows_x) {
    return;
  }
  const int nblk = ncols_x / ptq1_0::BLOCK_ELEMS;
  const uint32_t *wrow =
      w + static_cast<size_t>(row) * nblk * ptq1_0::BLOCK_WORDS;

  const int slot = lane / ptq1_0::BLOCK_WORDS;
  const int word_index = lane % ptq1_0::BLOCK_WORDS;
  const ptq1_0::DpLane dl = ptq1_0::dp_lane(word_index);

  const int *xrow[TT];
  const float *srow[TT];
  float acc[TT];
#pragma unroll
  for (int t = 0; t < TT; ++t) {
    const int tok = min(t0 + t, b_size - 1);
    const int8_t *xbytes = xq + static_cast<size_t>(tok) * ncols_x;
    xrow[t] = reinterpret_cast<const int *>(xbytes);
    srow[t] = xscale + static_cast<size_t>(tok) * nblk;
    acc[t] = 0.0f;
  }

  for (int step = 0; step < nblk; step += BLOCKS_PER_ITER) {
    uint32_t words[MATMUL_PREFETCH];
#pragma unroll
    for (int p = 0; p < MATMUL_PREFETCH; ++p) {
      const int b = step + p * BLOCKS_PER_WARP_STEP + slot;
      const size_t at =
          static_cast<size_t>(b) * ptq1_0::BLOCK_WORDS + word_index;
      words[p] = (slot < BLOCKS_PER_WARP_STEP && b < nblk) ? __ldg(wrow + at)
                                                           : 0u;
    }
#pragma unroll
    for (int p = 0; p < MATMUL_PREFETCH; ++p) {
      const int b = step + p * BLOCKS_PER_WARP_STEP + slot;
      const bool valid = slot < BLOCKS_PER_WARP_STEP && b < nblk;
      const int scale_lane = slot * ptq1_0::BLOCK_WORDS + SCALE_WORD;
      const uint32_t last = __shfl_sync(0xffffffffu, words[p], scale_lane);
      const float d = __half2float(
          __ushort_as_half(static_cast<unsigned short>(last >> 16)));

      uint32_t v_lo, v_hi;
      ptq1_0::init_state(words[p], word_index, v_lo, v_hi);
      int isum[TT] = {};
#pragma unroll
      for (int n = 0; n < ptq1_0::DECODE_STEPS; ++n) {
        const int q = ptq1_0::decode_step(v_lo, v_hi, dl.mult);
        if (n < dl.steps) {
          const int e = valid ? b * ptq1_0::BLOCK_ELEMS + dl.elem_base +
                                    n * dl.elem_stride
                              : 0;
#pragma unroll
          for (int t = 0; t < TT; ++t) {
            isum[t] = ptq1_0::dot4(q, __ldg(xrow[t] + (e >> 2)), isum[t]);
          }
        }
      }
      if (valid) {
#pragma unroll
        for (int t = 0; t < TT; ++t) {
          acc[t] += d * __ldg(srow[t] + b) * static_cast<float>(isum[t]);
        }
      }
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

// `scratch` holds b_size * ncols_x int8 activations, then one f32 scale per
// 128 columns per token.
template <typename T>
static void ptq1_0_launch(const void *x, const void *w, const void *signs,
                          const void *gather, void *scratch, void *dst,
                          int ncols_x, int nrows_x, int b_size, int do_fwht,
                          void *stream) {
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  int8_t *xq = static_cast<int8_t *>(scratch);
  float *xscale =
      reinterpret_cast<float *>(xq + static_cast<size_t>(b_size) * ncols_x);

  dim3 prep_grid((ncols_x + ptq1_0::FWHT_BLOCK - 1) / ptq1_0::FWHT_BLOCK,
                 b_size, 1);
  ptq1_0_prepare_kernel<T><<<prep_grid, PREPARE_THREADS, 0, s>>>(
      static_cast<const T *>(x), static_cast<const float *>(signs),
      static_cast<const uint32_t *>(gather), xq, xscale, ncols_x, do_fwht);

  const int block = MATMUL_WARPS * WARP_SIZE;
  const unsigned int row_blocks = (nrows_x + MATMUL_WARPS - 1) / MATMUL_WARPS;
  const uint32_t *wp = static_cast<const uint32_t *>(w);
  if (b_size == 1) {
    ptq1_0_matmul_kernel<T, 1><<<dim3(row_blocks, 1, 1), block, 0, s>>>(
        wp, xq, xscale, static_cast<T *>(dst), ncols_x, nrows_x, b_size);
  } else {
    const unsigned int passes =
        (b_size + MATMUL_TOKENS_PER_PASS - 1) / MATMUL_TOKENS_PER_PASS;
    ptq1_0_matmul_kernel<T, MATMUL_TOKENS_PER_PASS>
        <<<dim3(row_blocks, passes, 1), block, 0, s>>>(
            wp, xq, xscale, static_cast<T *>(dst), ncols_x, nrows_x, b_size);
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
