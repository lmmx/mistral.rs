//! CPU linear over PTQ1_0 blocks read straight from the GGUF mmap, applying the Hadamard fold to activations.

use std::sync::{atomic::AtomicUsize, Arc};

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::Linear;
use rayon::prelude::*;

#[cfg(feature = "cuda")]
use super::ptq1_0_cuda::PackedWeights;
use super::{
    archive::GgufArchive,
    hadamard::RowTransform,
    ptq1_0::{dequantize_row, trit, unpack_block_signed, PTQ1_0_BLOCK_BYTES, PTQ1_0_BLOCK_ELEMS},
};
use crate::{
    IsqPlanParams, IsqRequest, IsqType, QuantMethod, QuantMethodConfig, QuantizeOntoGuard,
    QuantizedSerde, UnquantLinear,
};

#[derive(Debug)]
pub struct Ptq1_0Linear {
    archive: Arc<GgufArchive>,
    name: String,
    out_dim: usize,
    in_dim: usize,
    transform: Option<RowTransform>,
    bias: Option<Tensor>,
    dtype: DType,
    device: Device,
    #[cfg(feature = "cuda")]
    gpu: Option<PackedWeights>,
}

impl Ptq1_0Linear {
    pub fn new(
        archive: Arc<GgufArchive>,
        name: &str,
        transform: Option<RowTransform>,
        bias: Option<Tensor>,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let shape = archive.tensor_info(name)?.shape().to_vec();
        let &[out_dim, in_dim] = shape.as_slice() else {
            candle_core::bail!("PTQ1_0 linear `{name}` must be rank 2, got {shape:?}");
        };
        if !in_dim.is_multiple_of(PTQ1_0_BLOCK_ELEMS) {
            candle_core::bail!("PTQ1_0 linear `{name}` width {in_dim} is not a multiple of 128");
        }
        #[cfg(feature = "cuda")]
        let gpu = if device.is_cuda() {
            let bytes = archive.tensor_data(name)?.bytes();
            Some(PackedWeights::upload(bytes, transform.as_ref(), device)?)
        } else {
            None
        };
        #[cfg(not(feature = "cuda"))]
        if !device.is_cpu() {
            candle_core::bail!(
                "PTQ1_0 packed linear `{name}` needs the cuda feature for {device:?}"
            );
        }
        Ok(Self {
            archive,
            name: name.to_string(),
            out_dim,
            in_dim,
            transform,
            bias,
            dtype,
            device: device.clone(),
            #[cfg(feature = "cuda")]
            gpu,
        })
    }

    fn row_bytes(&self) -> usize {
        self.in_dim / PTQ1_0_BLOCK_ELEMS * PTQ1_0_BLOCK_BYTES
    }

    fn weight_bytes(&self) -> Result<&[u8]> {
        Ok(self.archive.tensor_data(&self.name)?.bytes())
    }

    fn require_cpu(tensor: &Tensor) -> Result<()> {
        if !tensor.device().is_cpu() {
            candle_core::bail!("PTQ1_0 packed linear only runs on CPU activations");
        }
        Ok(())
    }
}

const DOT_LANES: usize = 16;
const ROWS_PER_TASK: usize = 16;
#[cfg(all(test, feature = "cuda"))]
const CUDA_INT8_REL_ERR: f32 = 2e-2; // int8 activations, one scale per 128 columns
const TOKEN_GROUP: usize = 4;
const TOKEN_TILE: usize = 16;
const K_TILE_BLOCKS: usize = 8;
const FUSED_MAX_TOKENS: usize = 2; // above this, decoding each block once beats redoing it per token
const QS_WIDE: usize = 16;
const QS_NARROW: usize = 8;
const QS_NARROW_START: usize = QS_WIDE * 5;
const QH_START: usize = QS_NARROW_START + QS_NARROW * 5;

type Lanes = [f32; DOT_LANES];

/// Per-lane partial sums for `T` tokens against one f32-converted block, so each weight load feeds `T` FMAs.
#[inline(always)]
fn block_dot_tokens<const T: usize>(wf: &[f32; PTQ1_0_BLOCK_ELEMS], xs: [&[f32]; T]) -> [Lanes; T] {
    let mut acc = [[0f32; DOT_LANES]; T];
    for (k, w) in wf.as_chunks::<DOT_LANES>().0.iter().enumerate() {
        let at = k * DOT_LANES;
        for t in 0..T {
            let x = &xs[t][at..at + DOT_LANES];
            for i in 0..DOT_LANES {
                acc[t][i] += w[i] * x[i];
            }
        }
    }
    acc
}

/// Unscaled dot of one packed block with `x`, decoding trits in lanes across the stage bytes.
#[inline(always)]
fn block_dot_fused(block: &[u8; PTQ1_0_BLOCK_BYTES], x: &[f32]) -> f32 {
    let mut acc = [0f32; QS_WIDE];
    for n in 0..5 {
        for m in 0..QS_WIDE {
            acc[m] += (trit(block[m], n) as i32 - 1) as f32 * x[n * QS_WIDE + m];
        }
    }
    for n in 0..5 {
        for m in 0..QS_NARROW {
            let t = trit(block[QS_WIDE + m], n) as i32 - 1;
            acc[m] += t as f32 * x[QS_NARROW_START + n * QS_NARROW + m];
        }
    }
    for n in 0..4 {
        for h in 0..2 {
            let t = trit(block[QS_WIDE + QS_NARROW + h], n) as i32 - 1;
            acc[h] += t as f32 * x[QH_START + n * 2 + h];
        }
    }
    acc.iter().sum()
}

struct RowJob<'a> {
    in_dim: usize,
    tokens: usize,
    xt: &'a [f32],
}

/// Accumulates `acc` (`rows x tokens`) for a chunk of weight rows, tiling so activations stay in L2.
#[inline(always)]
fn matmul_rows(lanes: &mut Vec<Lanes>, acc: &mut [f32], bytes: &[u8], job: &RowJob) {
    let tokens = job.tokens;
    let blocks_per_row = job.in_dim / PTQ1_0_BLOCK_ELEMS;
    let row_bytes = blocks_per_row * PTQ1_0_BLOCK_BYTES;
    let rows = acc.len() / tokens;
    if tokens <= FUSED_MAX_TOKENS {
        for (acc, row) in acc.chunks_mut(tokens).zip(bytes.chunks(row_bytes)) {
            acc.fill(0.0);
            for (b, block) in row.as_chunks::<PTQ1_0_BLOCK_BYTES>().0.iter().enumerate() {
                let d = super::ptq1_0::block_scale(block);
                for (t, a) in acc.iter_mut().enumerate() {
                    let start = t * job.in_dim + b * PTQ1_0_BLOCK_ELEMS;
                    *a += d * block_dot_fused(block, &job.xt[start..start + PTQ1_0_BLOCK_ELEMS]);
                }
            }
        }
        return;
    }
    lanes.clear();
    lanes.resize(rows * tokens, [0f32; DOT_LANES]);
    let mut tile = vec![[0i8; PTQ1_0_BLOCK_ELEMS]; rows * K_TILE_BLOCKS];
    let mut scales = vec![0f32; rows * K_TILE_BLOCKS];
    for k0 in (0..blocks_per_row).step_by(K_TILE_BLOCKS) {
        let kn = K_TILE_BLOCKS.min(blocks_per_row - k0);
        for r in 0..rows {
            let row = &bytes[r * row_bytes..(r + 1) * row_bytes];
            let blocks = &row.as_chunks::<PTQ1_0_BLOCK_BYTES>().0[k0..k0 + kn];
            for (j, block) in blocks.iter().enumerate() {
                unpack_block_signed(block, &mut tile[r * K_TILE_BLOCKS + j]);
                scales[r * K_TILE_BLOCKS + j] = super::ptq1_0::block_scale(block);
            }
        }
        for t0 in (0..tokens).step_by(TOKEN_TILE) {
            let t_end = (t0 + TOKEN_TILE).min(tokens);
            for r in 0..rows {
                for j in 0..kn {
                    let wf = tile[r * K_TILE_BLOCKS + j].map(|s| s as f32);
                    let d = scales[r * K_TILE_BLOCKS + j];
                    let at = (k0 + j) * PTQ1_0_BLOCK_ELEMS;
                    let x_of = |t: usize| {
                        let start = t * job.in_dim + at;
                        &job.xt[start..start + PTQ1_0_BLOCK_ELEMS]
                    };
                    let row_lanes = &mut lanes[r * tokens..(r + 1) * tokens];
                    let mut t = t0;
                    while t + TOKEN_GROUP <= t_end {
                        let xs = std::array::from_fn(|i| x_of(t + i));
                        let part = block_dot_tokens::<TOKEN_GROUP>(&wf, xs);
                        for (l, p) in row_lanes[t..t + TOKEN_GROUP].iter_mut().zip(part) {
                            for (l, p) in l.iter_mut().zip(p) {
                                *l += d * p;
                            }
                        }
                        t += TOKEN_GROUP;
                    }
                    for (i, l) in row_lanes[t..t_end].iter_mut().enumerate() {
                        let [part] = block_dot_tokens::<1>(&wf, [x_of(t + i)]);
                        for (l, p) in l.iter_mut().zip(part) {
                            *l += d * p;
                        }
                    }
                }
            }
        }
    }
    for (a, l) in acc.iter_mut().zip(lanes.iter()) {
        *a = l.iter().sum();
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn matmul_rows_avx2(lanes: &mut Vec<Lanes>, acc: &mut [f32], bytes: &[u8], job: &RowJob) {
    matmul_rows(lanes, acc, bytes, job)
}

/// Returns `[tokens, out_dim]` for already-transformed activations `xt` of shape `[tokens, in_dim]`.
fn packed_matmul(
    bytes: &[u8],
    out_dim: usize,
    in_dim: usize,
    xt: &[f32],
    tokens: usize,
) -> Vec<f32> {
    let row_bytes = in_dim / PTQ1_0_BLOCK_ELEMS * PTQ1_0_BLOCK_BYTES;
    #[cfg(target_arch = "x86_64")]
    let avx2 = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
    let job = RowJob { in_dim, tokens, xt };
    let mut by_row = vec![0f32; out_dim * tokens];
    by_row
        .par_chunks_mut(ROWS_PER_TASK * tokens)
        .zip(bytes.par_chunks(ROWS_PER_TASK * row_bytes))
        .for_each_init(Vec::new, |lanes, (acc, rows)| {
            #[cfg(target_arch = "x86_64")]
            if avx2 {
                // SAFETY: avx2 and fma were detected at runtime above
                return unsafe { matmul_rows_avx2(lanes, acc, rows, &job) };
            }
            matmul_rows(lanes, acc, rows, &job)
        });
    let mut out = vec![0f32; tokens * out_dim];
    for (r, col) in by_row.chunks_exact(tokens).enumerate() {
        for (t, v) in col.iter().enumerate() {
            out[t * out_dim + r] = *v;
        }
    }
    out
}

impl QuantMethod for Ptq1_0Linear {
    fn new(_method: QuantMethodConfig) -> Result<Self>
    where
        Self: Sized,
    {
        candle_core::bail!("PTQ1_0 linears are only built from a GGUF archive")
    }

    fn dequantize_w(&self) -> Result<Tensor> {
        let mut data = vec![0f32; self.out_dim * self.in_dim];
        let row_bytes = self.row_bytes();
        data.par_chunks_mut(self.in_dim)
            .zip(self.weight_bytes()?.par_chunks(row_bytes))
            .for_each(|(dst, src)| dequantize_row(src, dst));
        if let Some(transform) = &self.transform {
            transform.unfold_weight(&mut data);
        }
        Tensor::from_vec(data, (self.out_dim, self.in_dim), &Device::Cpu)?
            .to_dtype(self.dtype)?
            .to_device(&self.device)
    }

    fn forward_raw(&self, a: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if let Some(gpu) = &self.gpu {
            let y = gpu.matmul(a, self.out_dim, self.in_dim)?;
            return match &self.bias {
                Some(bias) => y.broadcast_add(&bias.to_dtype(y.dtype())?),
                None => Ok(y),
            };
        }
        Self::require_cpu(a)?;
        let dims = a.dims().to_vec();
        if dims.last() != Some(&self.in_dim) {
            candle_core::bail!("PTQ1_0 linear `{}` got input shape {dims:?}", self.name);
        }
        let tokens = a.elem_count() / self.in_dim;
        let mut xt = a
            .to_dtype(DType::F32)?
            .contiguous()?
            .flatten_all()?
            .to_vec1::<f32>()?;
        if let Some(transform) = &self.transform {
            xt.par_chunks_mut(self.in_dim)
                .for_each_init(Vec::new, |scratch, row| transform.apply(row, scratch));
        }
        let y = packed_matmul(self.weight_bytes()?, self.out_dim, self.in_dim, &xt, tokens);
        let mut y = Tensor::from_vec(y, (tokens, self.out_dim), &Device::Cpu)?;
        if let Some(bias) = &self.bias {
            y = y.broadcast_add(&bias.to_dtype(DType::F32)?)?;
        }
        let mut out_dims = dims;
        *out_dims.last_mut().expect("checked non-empty above") = self.out_dim;
        y.to_dtype(a.dtype())?.reshape(out_dims)
    }

    fn embedding_forward(&self, ids: &Tensor, output_dtype: DType) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if let Some(gpu) = &self.gpu {
            return gpu.embedding(ids, self.in_dim, output_dtype);
        }
        Self::require_cpu(ids)?;
        let dims = ids.dims().to_vec();
        let flat = ids.to_dtype(DType::U32)?.flatten_all()?.to_vec1::<u32>()?;
        let bytes = self.weight_bytes()?;
        let row_bytes = self.row_bytes();
        let mut out = vec![0f32; flat.len() * self.in_dim];
        out.par_chunks_mut(self.in_dim)
            .zip(flat.par_iter())
            .for_each_init(Vec::new, |scratch, (dst, id)| {
                let start = *id as usize * row_bytes;
                dequantize_row(&bytes[start..start + row_bytes], dst);
                if let Some(transform) = &self.transform {
                    transform.apply(dst, scratch);
                }
            });
        let mut out_dims = dims;
        out_dims.push(self.in_dim);
        Tensor::from_vec(out, out_dims, &Device::Cpu)?.to_dtype(output_dtype)
    }

    fn quantized_act_type(&self) -> Option<DType> {
        None
    }

    fn dtype_and_device(&self) -> (DType, Device) {
        (self.dtype, self.device.clone())
    }

    fn plan_isq(&self, request: &IsqRequest) -> Result<IsqPlanParams> {
        Ok(crate::plan_weight_isq(
            self.dtype,
            self.device.clone(),
            vec![self.out_dim, self.in_dim],
            request,
            false,
        ))
    }

    fn add_delta_w(&self, _delta: &Tensor) -> Result<Arc<dyn QuantMethod>> {
        candle_core::bail!("PTQ1_0 packed linear does not support LoRA deltas")
    }

    fn apply_isq(
        self: Arc<Self>,
        dtype: Option<IsqType>,
        device: Device,
        n_quantized: &AtomicUsize,
        imatrix_weight: Option<Vec<f32>>,
        guard: QuantizeOntoGuard,
    ) -> Result<Arc<dyn QuantMethod>> {
        let dense = UnquantLinear::new(QuantMethodConfig::Unquantized(Linear::new(
            self.dequantize_w()?,
            self.bias.clone(),
        )))?;
        Arc::new(dense).apply_isq(dtype, device, n_quantized, imatrix_weight, guard)
    }

    fn has_bias(&self) -> bool {
        self.bias.is_some()
    }
}

impl QuantizedSerde for Ptq1_0Linear {
    fn name(&self) -> &'static str {
        "ptq1_0"
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::*;
    #[cfg(feature = "cuda")]
    use crate::gguf::hadamard::HadamardRole;
    use crate::gguf::ptq1_0::encode_block;

    const LCG_MUL: u64 = 6364136223846793005;

    fn lcg(state: &mut u64) -> u32 {
        *state = state
            .wrapping_mul(LCG_MUL)
            .wrapping_add(1442695040888963407);
        (*state >> 33) as u32
    }

    fn synthetic(out_dim: usize, in_dim: usize, tokens: usize) -> (Vec<u8>, Vec<f32>) {
        let mut s = 7u64;
        let mut bytes = Vec::new();
        for _ in 0..out_dim * in_dim / PTQ1_0_BLOCK_ELEMS {
            let mut codes = [0u8; PTQ1_0_BLOCK_ELEMS];
            codes.iter_mut().for_each(|c| *c = (lcg(&mut s) % 3) as u8);
            bytes.extend_from_slice(&encode_block(
                &codes,
                0.01 + (lcg(&mut s) % 100) as f32 * 1e-4,
            ));
        }
        let x = (0..tokens * in_dim)
            .map(|_| (lcg(&mut s) % 2001) as f32 / 1000.0 - 1.0)
            .collect();
        (bytes, x)
    }

    #[test]
    fn packed_matmul_matches_dequantized_dense() {
        let wide = (K_TILE_BLOCKS + 3) * PTQ1_0_BLOCK_ELEMS;
        for in_dim in [3 * PTQ1_0_BLOCK_ELEMS, wide] {
            for tokens in [1, FUSED_MAX_TOKENS + 1, TOKEN_TILE + TOKEN_GROUP + 1] {
                check_against_dense(in_dim, tokens);
            }
        }
    }

    fn check_against_dense(in_dim: usize, tokens: usize) {
        let out_dim = ROWS_PER_TASK * 2 + 5;
        let (bytes, x) = synthetic(out_dim, in_dim, tokens);
        let mut w = vec![0f32; out_dim * in_dim];
        w.chunks_mut(in_dim)
            .zip(bytes.chunks(in_dim / PTQ1_0_BLOCK_ELEMS * PTQ1_0_BLOCK_BYTES))
            .for_each(|(d, s)| dequantize_row(s, d));
        let got = packed_matmul(&bytes, out_dim, in_dim, &x, tokens);
        for t in 0..tokens {
            for r in 0..out_dim {
                let want: f32 = (0..in_dim)
                    .map(|c| w[r * in_dim + c] * x[t * in_dim + c])
                    .sum();
                let g = got[t * out_dim + r];
                assert!(
                    (g - want).abs() < 1e-3 * want.abs().max(1.0),
                    "{g} vs {want}"
                );
            }
        }
    }

    #[test]
    #[ignore = "timing"]
    fn packed_matmul_speed() {
        for tokens in [1, 16, 64, 256] {
            let (out_dim, in_dim) = (5120, 17408);
            let (bytes, x) = synthetic(out_dim, in_dim, tokens);
            packed_matmul(&bytes, out_dim, in_dim, &x, tokens);
            let start = Instant::now();
            let reps = 5;
            for _ in 0..reps {
                std::hint::black_box(packed_matmul(&bytes, out_dim, in_dim, &x, tokens));
            }
            let secs = start.elapsed().as_secs_f64() / reps as f64;
            let gbps = bytes.len() as f64 / secs / 1e9;
            eprintln!(
                "tokens {tokens}: {:.1} ms, {gbps:.2} GB/s of weights",
                secs * 1e3
            );
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "needs a CUDA device"]
    fn cuda_matches_cpu_packed() -> Result<()> {
        use crate::gguf::ptq1_0_cuda::PackedWeights;

        let dev = Device::new_cuda(0)?;
        let cases = [
            (384, 1, false),
            (2048, 1, true),
            (2048, 5, false),
            (3072, 300, true),
        ];
        for (in_dim, tokens, folded) in cases {
            let out_dim = ROWS_PER_TASK * 2 + 5;
            let (bytes, x) = synthetic(out_dim, in_dim, tokens);
            let transform = folded
                .then(|| RowTransform::for_test(HadamardRole::Fold, in_dim, 11, in_dim > 2048));
            let gpu = PackedWeights::upload(&bytes, transform.as_ref(), &dev)?;
            for dtype in [DType::F32, DType::F16, DType::BF16] {
                let input = Tensor::from_vec(x.clone(), (tokens, in_dim), &dev)?.to_dtype(dtype)?;
                let rounded = input
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                let mut xt = rounded.clone();
                if let Some(transform) = &transform {
                    xt.chunks_mut(in_dim)
                        .for_each(|row| transform.apply(row, &mut Vec::new()));
                }
                let want = packed_matmul(&bytes, out_dim, in_dim, &xt, tokens);
                let got = gpu
                    .matmul(&input, out_dim, in_dim)?
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                let norm = want.iter().map(|w| w * w).sum::<f32>().sqrt();
                let err = got
                    .iter()
                    .zip(&want)
                    .map(|(g, w)| (g - w).powi(2))
                    .sum::<f32>()
                    .sqrt();
                assert!(
                    err / norm < CUDA_INT8_REL_ERR,
                    "{dtype:?} in {in_dim} tokens {tokens} folded {folded}: {}",
                    err / norm
                );
            }
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "needs a CUDA device"]
    fn cuda_embedding_matches_cpu() -> Result<()> {
        use crate::gguf::ptq1_0_cuda::PackedWeights;

        let dev = Device::new_cuda(0)?;
        let (vocab, width) = (50, 3072);
        let (bytes, _) = synthetic(vocab, width, 1);
        let transform = RowTransform::for_test(HadamardRole::Inverse, width, 5, false);
        let gpu = PackedWeights::upload(&bytes, Some(&transform), &dev)?;
        let ids = vec![3u32, 0, 49, 7, 7, 21];
        let row_bytes = width / PTQ1_0_BLOCK_ELEMS * PTQ1_0_BLOCK_BYTES;
        let mut want = Vec::new();
        for id in &ids {
            let mut row = vec![0f32; width];
            let start = *id as usize * row_bytes;
            dequantize_row(&bytes[start..start + row_bytes], &mut row);
            transform.apply(&mut row, &mut Vec::new());
            want.extend(row);
        }
        let input = Tensor::from_vec(ids, (2, 3), &dev)?;
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            let got = gpu.embedding(&input, width, dtype)?;
            assert_eq!(got.dims(), [2, 3, width]);
            let got = got.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
            let norm = want.iter().map(|w| w * w).sum::<f32>().sqrt();
            let err = got
                .iter()
                .zip(&want)
                .map(|(g, w)| (g - w).powi(2))
                .sum::<f32>()
                .sqrt();
            let tol = if dtype == DType::F32 { 1e-4 } else { 1e-2 };
            assert!(err / norm < tol, "{dtype:?}: {}", err / norm);
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "needs a CUDA device"]
    fn cuda_matmul_speed() -> Result<()> {
        use crate::gguf::ptq1_0_cuda::PackedWeights;

        const REPS: usize = 50;
        const UNSIGNED_FROM: i32 = 3;
        const UNSIGNED_REL_ERR: f32 = 5e-3;
        const VARIANTS: [&str; 8] = [
            "pf2", "pf4", "pf8", "pf2 u", "pf4 u", "pf8 u", "bpl", "bpl u",
        ];
        let dev = Device::new_cuda(0)?;
        let time = |f: &dyn Fn() -> Result<Tensor>| -> Result<f64> {
            f()?;
            dev.synchronize()?;
            let start = Instant::now();
            for _ in 0..REPS {
                f()?;
            }
            dev.synchronize()?;
            Ok(start.elapsed().as_secs_f64() / REPS as f64)
        };

        eprintln!("1 token, GB/s of weights; variants {VARIANTS:?} (u = unsigned digits)");
        let shapes = [
            (17408, 5120, "ffn gate/up"),
            (5120, 17408, "ffn down"),
            (10240, 5120, "gdn qkv"),
            (6144, 5120, "gdn gate"),
            (5120, 6144, "ssm out"),
            (12288, 5120, "attn q"),
            (1024, 5120, "attn k/v"),
            (248320, 5120, "output"),
        ];
        for (out_dim, in_dim, label) in shapes {
            let (bytes, x) = synthetic(out_dim, in_dim, 1);
            let transform = RowTransform::for_test(HadamardRole::Fold, in_dim, 11, false);
            let gpu = PackedWeights::upload(&bytes, Some(&transform), &dev)?;
            let input = Tensor::from_vec(x, (1, in_dim), &dev)?.to_dtype(DType::BF16)?;
            let floats = |t: Tensor| -> Result<Vec<f32>> {
                t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()
            };
            let baseline = floats(gpu.matmul_variant(&input, out_dim, 0)?)?;
            let norm = baseline.iter().map(|v| v * v).sum::<f32>().sqrt();
            let mut row = Vec::new();
            for variant in 0..VARIANTS.len() as i32 {
                let got = floats(gpu.matmul_variant(&input, out_dim, variant)?)?;
                let err = got
                    .iter()
                    .zip(&baseline)
                    .map(|(g, b)| (g - b).powi(2))
                    .sum::<f32>()
                    .sqrt();
                // Unsigned decode reorders float partial sums, so it only matches to rounding.
                let tol = if variant < UNSIGNED_FROM {
                    0.0
                } else {
                    UNSIGNED_REL_ERR
                };
                assert!(
                    err <= tol * norm,
                    "{label} variant {variant}: {}",
                    err / norm
                );
                let secs = time(&|| gpu.matmul_variant(&input, out_dim, variant))?;
                row.push(format!("{:5.0}", bytes.len() as f64 / secs / 1e9));
            }
            eprintln!(
                "{out_dim:>6} x {in_dim:<5} {label:<12} {:>6.1} MB  {}",
                bytes.len() as f64 / 1e6,
                row.join(" ")
            );
        }

        let (out_dim, in_dim) = (5120, 17408);
        let (bytes, _) = synthetic(out_dim, in_dim, 1);
        let transform = RowTransform::for_test(HadamardRole::Fold, in_dim, 11, false);
        let gpu = PackedWeights::upload(&bytes, Some(&transform), &dev)?;
        for tokens in [1, 8, 59, 256] {
            let x = vec![0.5f32; tokens * in_dim];
            let input = Tensor::from_vec(x, (tokens, in_dim), &dev)?.to_dtype(DType::BF16)?;
            let secs = time(&|| gpu.matmul(&input, out_dim, in_dim))?;
            eprintln!(
                "production kernel, tokens {tokens}: {:.3} ms, {:.1} GB/s of weights",
                secs * 1e3,
                bytes.len() as f64 / secs / 1e9
            );
        }
        Ok(())
    }
}
