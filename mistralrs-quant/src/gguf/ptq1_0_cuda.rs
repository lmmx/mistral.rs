//! CUDA path for PTQ1_0 linears: the Hadamard fold on activations plus a packed ternary matmul.

use std::ffi::c_void;

use candle_core::cuda::cudarc::driver::DeviceRepr;
use candle_core::{
    cuda::cudarc::driver::{CudaStream, SyncOnDrop},
    cuda_backend::CudaDType,
    CudaDevice, CudaStorage, DType, Device, Result, Shape, Storage, Tensor,
};

use super::{fast_mmvq::workspace_ensure, ffi, hadamard::RowTransform, ptq1_0::PTQ1_0_BLOCK_ELEMS};
use crate::utils::{slice_ptr_mut_on_stream, slice_ptr_on_stream};

const TOKEN_CHUNK: usize = 256;

type Launcher = unsafe extern "C" fn(
    x: *const c_void,
    w: *const c_void,
    signs: *const c_void,
    gather: *const c_void,
    scratch: *mut c_void,
    dst: *mut c_void,
    ncols_x: i32,
    nrows_x: i32,
    b_size: i32,
    do_fwht: i32,
    stream: *mut c_void,
);

/// Packed blocks and fold constants resident on one CUDA device.
#[derive(Debug)]
pub(crate) struct PackedWeights {
    blocks: Tensor,
    signs: Option<Tensor>,
    gather: Option<Tensor>,
    inverse: bool,
}

type EmbeddingLauncher = unsafe extern "C" fn(
    ids: *const c_void,
    w: *const c_void,
    signs: *const c_void,
    dst: *mut c_void,
    ncols_x: i32,
    n_ids: i32,
    stream: *mut c_void,
);

fn cuda_ptr<'a, T: CudaDType + DeviceRepr + 'a>(
    storage: &'a Storage,
    offset: usize,
    stream: &'a CudaStream,
) -> Result<(u64, SyncOnDrop<'a>)> {
    let Storage::Cuda(cuda) = storage else {
        candle_core::bail!("PTQ1_0 CUDA path: tensor must live on CUDA");
    };
    Ok(slice_ptr_on_stream(
        cuda.as_cuda_slice::<T>()?,
        offset,
        stream,
    ))
}

impl PackedWeights {
    pub(crate) fn upload(
        bytes: &[u8],
        transform: Option<&RowTransform>,
        device: &Device,
    ) -> Result<Self> {
        let blocks = Tensor::from_raw_buffer(bytes, DType::U8, &[bytes.len()], device)?;
        let (signs, gather) = match transform {
            Some(t) => (
                Some(Tensor::from_slice(t.signs(), t.signs().len(), device)?),
                t.gather()
                    .map(|g| Tensor::from_slice(g, g.len(), device))
                    .transpose()?,
            ),
            None => (None, None),
        };
        Ok(Self {
            blocks,
            signs,
            gather,
            inverse: transform.is_some_and(RowTransform::is_inverse),
        })
    }

    /// `xs` is `[..., ncols]`; returns `[..., nrows]` in the input dtype.
    pub(crate) fn matmul(&self, xs: &Tensor, nrows: usize, ncols: usize) -> Result<Tensor> {
        let Device::Cuda(dev) = xs.device() else {
            candle_core::bail!("PTQ1_0 CUDA path: input must live on CUDA");
        };
        if self.inverse {
            candle_core::bail!(
                "PTQ1_0 CUDA path: inverse-role tensors only support embedding lookups"
            );
        }
        let Some((&k, batch_dims)) = xs.dims().split_last() else {
            candle_core::bail!("PTQ1_0 CUDA path: input must have at least one dimension");
        };
        if k != ncols {
            candle_core::bail!("PTQ1_0 CUDA path: weight width {ncols} vs input tail {k}");
        }
        let b_size = batch_dims.iter().product::<usize>();
        let xs2 = xs.contiguous()?;
        let out = match xs.dtype() {
            DType::F32 => {
                self.run::<f32>(dev, &xs2, nrows, k, b_size, ffi::launch_ptq1_0_matmul_f32)
            }
            DType::F16 => {
                self.run::<half::f16>(dev, &xs2, nrows, k, b_size, ffi::launch_ptq1_0_matmul_f16)
            }
            DType::BF16 => {
                self.run::<half::bf16>(dev, &xs2, nrows, k, b_size, ffi::launch_ptq1_0_matmul_bf16)
            }
            other => candle_core::bail!("PTQ1_0 CUDA path: unsupported input dtype {other:?}"),
        }?;
        let mut out_dims = xs.dims().to_vec();
        *out_dims.last_mut().expect("split_last succeeded") = nrows;
        out.reshape(out_dims)
    }

    fn run<T: CudaDType + DeviceRepr>(
        &self,
        dev: &CudaDevice,
        xs: &Tensor,
        nrows: usize,
        k: usize,
        b_size: usize,
        launcher: Launcher,
    ) -> Result<Tensor> {
        let stream = dev.cuda_stream();
        let stream_ptr = stream.cu_stream() as *mut c_void;
        let (xs_storage, xs_layout) = xs.storage_and_layout();
        let (w_storage, w_layout) = self.blocks.storage_and_layout();
        let signs_pair = self.signs.as_ref().map(|t| t.storage_and_layout());
        let gather_pair = self.gather.as_ref().map(|t| t.storage_and_layout());

        let (x_ptr, _x_guard) = cuda_ptr::<T>(&xs_storage, xs_layout.start_offset(), &stream)?;
        let (w_ptr, _w_guard) = cuda_ptr::<u8>(&w_storage, w_layout.start_offset(), &stream)?;
        let signs = signs_pair
            .as_ref()
            .map(|(s, l)| cuda_ptr::<f32>(s, l.start_offset(), &stream))
            .transpose()?;
        let gather = gather_pair
            .as_ref()
            .map(|(s, l)| cuda_ptr::<u32>(s, l.start_offset(), &stream))
            .transpose()?;
        let signs_ptr = signs
            .as_ref()
            .map_or(std::ptr::null(), |(p, _)| *p as *const c_void);
        let gather_ptr = gather
            .as_ref()
            .map_or(std::ptr::null(), |(p, _)| *p as *const c_void);
        let do_fwht = i32::from(self.signs.is_some());

        let scratch_bytes =
            b_size.min(TOKEN_CHUNK) * (k + k / PTQ1_0_BLOCK_ELEMS * size_of::<f32>());
        let mut workspace = workspace_ensure(dev, scratch_bytes, &stream)?;
        let (scratch_ptr, _scratch_guard) = workspace.ptr_mut();

        let mut out = unsafe { dev.alloc::<T>(b_size * nrows)? };
        {
            let (out_ptr, _out_guard) = slice_ptr_mut_on_stream(&mut out, 0, &stream);
            let mut done = 0;
            while done < b_size {
                let m = (b_size - done).min(TOKEN_CHUNK);
                let x_at = x_ptr + (done * k * size_of::<T>()) as u64;
                let out_at = out_ptr + (done * nrows * size_of::<T>()) as u64;
                unsafe {
                    launcher(
                        x_at as *const c_void,
                        w_ptr as *const c_void,
                        signs_ptr,
                        gather_ptr,
                        scratch_ptr as *mut c_void,
                        out_at as *mut c_void,
                        k as i32,
                        nrows as i32,
                        m as i32,
                        do_fwht,
                        stream_ptr,
                    );
                }
                done += m;
            }
        }

        let out_storage = CudaStorage::wrap_cuda_slice(out, dev.clone());
        Ok(Tensor::from((
            Storage::Cuda(out_storage),
            Shape::from((b_size, nrows)),
        )))
    }

    /// Rows of the packed table for `ids`, with the inverse fold applied; returns `[ids..., ncols]`.
    pub(crate) fn embedding(&self, ids: &Tensor, ncols: usize, dtype: DType) -> Result<Tensor> {
        let Device::Cuda(dev) = ids.device() else {
            candle_core::bail!("PTQ1_0 CUDA path: ids must live on CUDA");
        };
        let ids2 = ids.to_dtype(DType::U32)?.contiguous()?;
        let out = match dtype {
            DType::F32 => self.embed::<f32>(dev, &ids2, ncols, ffi::launch_ptq1_0_embedding_f32),
            DType::F16 => {
                self.embed::<half::f16>(dev, &ids2, ncols, ffi::launch_ptq1_0_embedding_f16)
            }
            DType::BF16 => {
                self.embed::<half::bf16>(dev, &ids2, ncols, ffi::launch_ptq1_0_embedding_bf16)
            }
            other => candle_core::bail!("PTQ1_0 CUDA path: unsupported embedding dtype {other:?}"),
        }?;
        let mut out_dims = ids.dims().to_vec();
        out_dims.push(ncols);
        out.reshape(out_dims)
    }

    fn embed<T: CudaDType + DeviceRepr>(
        &self,
        dev: &CudaDevice,
        ids: &Tensor,
        ncols: usize,
        launcher: EmbeddingLauncher,
    ) -> Result<Tensor> {
        let Some(signs) = &self.signs else {
            candle_core::bail!("PTQ1_0 CUDA path: embedding needs the inverse-fold signs");
        };
        let n_ids = ids.elem_count();
        let stream = dev.cuda_stream();
        let stream_ptr = stream.cu_stream() as *mut c_void;
        let (ids_storage, ids_layout) = ids.storage_and_layout();
        let (w_storage, w_layout) = self.blocks.storage_and_layout();
        let (signs_storage, signs_layout) = signs.storage_and_layout();
        let (ids_ptr, _ids_guard) =
            cuda_ptr::<u32>(&ids_storage, ids_layout.start_offset(), &stream)?;
        let (w_ptr, _w_guard) = cuda_ptr::<u8>(&w_storage, w_layout.start_offset(), &stream)?;
        let (signs_ptr, _signs_guard) =
            cuda_ptr::<f32>(&signs_storage, signs_layout.start_offset(), &stream)?;

        let mut out = unsafe { dev.alloc::<T>(n_ids * ncols)? };
        {
            let (out_ptr, _out_guard) = slice_ptr_mut_on_stream(&mut out, 0, &stream);
            unsafe {
                launcher(
                    ids_ptr as *const c_void,
                    w_ptr as *const c_void,
                    signs_ptr as *const c_void,
                    out_ptr as *mut c_void,
                    ncols as i32,
                    n_ids as i32,
                    stream_ptr,
                );
            }
        }
        let out_storage = CudaStorage::wrap_cuda_slice(out, dev.clone());
        Ok(Tensor::from((
            Storage::Cuda(out_storage),
            Shape::from((n_ids, ncols)),
        )))
    }
}
