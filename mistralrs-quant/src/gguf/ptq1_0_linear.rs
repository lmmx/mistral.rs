//! CPU linear over PTQ1_0 blocks read straight from the GGUF mmap, applying the Hadamard fold to activations.

use std::sync::{atomic::AtomicUsize, Arc};

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::Linear;
use rayon::prelude::*;

use super::{
    archive::GgufArchive,
    hadamard::RowTransform,
    ptq1_0::{dequantize_row, unpack_block_signed, PTQ1_0_BLOCK_BYTES, PTQ1_0_BLOCK_ELEMS},
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
}

impl Ptq1_0Linear {
    pub fn new(
        archive: Arc<GgufArchive>,
        name: &str,
        transform: Option<RowTransform>,
        bias: Option<Tensor>,
        dtype: DType,
    ) -> Result<Self> {
        let shape = archive.tensor_info(name)?.shape().to_vec();
        let &[out_dim, in_dim] = shape.as_slice() else {
            candle_core::bail!("PTQ1_0 linear `{name}` must be rank 2, got {shape:?}");
        };
        if !in_dim.is_multiple_of(PTQ1_0_BLOCK_ELEMS) {
            candle_core::bail!("PTQ1_0 linear `{name}` width {in_dim} is not a multiple of 128");
        }
        Ok(Self {
            archive,
            name: name.to_string(),
            out_dim,
            in_dim,
            transform,
            bias,
            dtype,
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

/// Returns `[tokens, out_dim]` for already-transformed activations `xt` of shape `[tokens, in_dim]`.
fn packed_matmul(
    bytes: &[u8],
    out_dim: usize,
    in_dim: usize,
    xt: &[f32],
    tokens: usize,
) -> Vec<f32> {
    let row_bytes = in_dim / PTQ1_0_BLOCK_ELEMS * PTQ1_0_BLOCK_BYTES;
    let mut by_row = vec![0f32; out_dim * tokens];
    by_row
        .par_chunks_mut(tokens)
        .zip(bytes.par_chunks(row_bytes))
        .for_each(|(acc, row)| {
            let mut signed = [0i8; PTQ1_0_BLOCK_ELEMS];
            for (b, block) in row.as_chunks::<PTQ1_0_BLOCK_BYTES>().0.iter().enumerate() {
                unpack_block_signed(block, &mut signed);
                let d = super::ptq1_0::block_scale(block);
                for (t, a) in acc.iter_mut().enumerate() {
                    let start = t * in_dim + b * PTQ1_0_BLOCK_ELEMS;
                    let x = &xt[start..start + PTQ1_0_BLOCK_ELEMS];
                    let dot: f32 = signed.iter().zip(x).map(|(s, x)| *s as f32 * x).sum();
                    *a += d * dot;
                }
            }
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
        Tensor::from_vec(data, (self.out_dim, self.in_dim), &Device::Cpu)?.to_dtype(self.dtype)
    }

    fn forward_raw(&self, a: &Tensor) -> Result<Tensor> {
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
        (self.dtype, Device::Cpu)
    }

    fn plan_isq(&self, request: &IsqRequest) -> Result<IsqPlanParams> {
        Ok(crate::plan_weight_isq(
            self.dtype,
            Device::Cpu,
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
