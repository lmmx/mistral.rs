//! Prism Hadamard weight fold: stored weights are `W' = W * diag(s) * H_blk` along the input axis.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use candle_core::{quantized::gguf_file::Value, Error, Result};
use rayon::prelude::*;

const KEY_VERSION: &str = "prism.hadamard.version";
const KEY_BLOCK_SIZE: &str = "prism.hadamard.block_size";
const KEY_TRANSFORM: &str = "prism.hadamard.transform";
const KEY_AXIS: &str = "prism.hadamard.axis";
const KEY_SIGN_MODE: &str = "prism.hadamard.sign_mode";
const KEY_SIGN_WIDTHS: &str = "prism.hadamard.sign_widths";
const KEY_SIGN_VALUES: &str = "prism.hadamard.sign_values";
const KEY_WEIGHT_NAMES: &str = "prism.hadamard.weight_names";
const KEY_INVERSE_NAMES: &str = "prism.hadamard.inverse_weight_names";
const KEY_GDN_V_GROUPED: &str = "prism.hadamard.gdn_v_grouped";
const SUPPORTED_VERSION: u32 = 1;
const TRANSFORM_NAME: &str = "normalized-sylvester-walsh-hadamard";
const AXIS_NAME: &str = "input-last-dimension";
const SIGN_MODE_IDENTITY: &str = "identity";
const SIGN_MODE_EXPLICIT: &str = "explicit";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HadamardRole {
    /// Weight consumes activations transformed by `H(s * x)`.
    Fold,
    /// Table whose looked-up rows are latent; the graph applies `s * (H z)`.
    Inverse,
}

#[derive(Clone, Debug)]
pub struct HadamardSpec {
    block: usize,
    signs: Option<HashMap<usize, Arc<[f32]>>>,
    weights: HashSet<String>,
    inverse: HashSet<String>,
    gdn_v_grouped: bool,
}

impl HadamardSpec {
    /// Returns `None` when the file carries no fold.
    pub fn from_metadata(metadata: &HashMap<String, Value>) -> Result<Option<Self>> {
        let Some(version) = metadata.get(KEY_VERSION) else {
            return Ok(None);
        };
        let version = version.to_u32()?;
        if version != SUPPORTED_VERSION {
            candle_core::bail!("unsupported {KEY_VERSION} {version}");
        }
        let block = required(metadata, KEY_BLOCK_SIZE)?.to_u32()? as usize;
        if !block.is_power_of_two() {
            candle_core::bail!("{KEY_BLOCK_SIZE} {block} is not a power of two");
        }
        expect_string(metadata, KEY_TRANSFORM, TRANSFORM_NAME)?;
        expect_string(metadata, KEY_AXIS, AXIS_NAME)?;
        let signs = match required(metadata, KEY_SIGN_MODE)?.to_string()?.as_str() {
            SIGN_MODE_IDENTITY => None,
            SIGN_MODE_EXPLICIT => Some(parse_signs(metadata, block)?),
            other => candle_core::bail!("unsupported {KEY_SIGN_MODE} `{other}`"),
        };
        let weights = string_set(required(metadata, KEY_WEIGHT_NAMES)?)?;
        if weights.is_empty() {
            candle_core::bail!("{KEY_WEIGHT_NAMES} is empty");
        }
        let inverse = match metadata.get(KEY_INVERSE_NAMES) {
            Some(value) => string_set(value)?,
            None => HashSet::new(),
        };
        let gdn_v_grouped = match metadata.get(KEY_GDN_V_GROUPED) {
            Some(value) => value.to_bool()?,
            None => false,
        };
        Ok(Some(Self {
            block,
            signs,
            weights,
            inverse,
            gdn_v_grouped,
        }))
    }

    pub fn block_size(&self) -> usize {
        self.block
    }

    pub fn gdn_v_grouped(&self) -> bool {
        self.gdn_v_grouped
    }

    pub fn role(&self, tensor_name: &str) -> Option<HadamardRole> {
        if self.inverse.contains(tensor_name) {
            Some(HadamardRole::Inverse)
        } else if self.weights.contains(tensor_name) {
            Some(HadamardRole::Fold)
        } else {
            None
        }
    }

    pub fn signs_for(&self, width: usize) -> Result<Arc<[f32]>> {
        if !width.is_multiple_of(self.block) {
            candle_core::bail!(
                "width {width} is not a multiple of Hadamard block {}",
                self.block
            );
        }
        match &self.signs {
            None => Ok(vec![1.0; width].into()),
            Some(map) => map
                .get(&width)
                .cloned()
                .ok_or_else(|| Error::msg(format!("no Hadamard sign vector for width {width}"))),
        }
    }

    /// Turns a stored (folded) row-major `[rows, width]` weight or table into its unfolded equivalent.
    pub fn unfold(&self, data: &mut [f32], width: usize) -> Result<()> {
        let signs = self.signs_for(width)?;
        unfold_rows(data, width, self.block, &signs);
        Ok(())
    }
}

fn required<'a>(metadata: &'a HashMap<String, Value>, key: &str) -> Result<&'a Value> {
    metadata
        .get(key)
        .ok_or_else(|| Error::msg(format!("Hadamard GGUF is missing `{key}`")))
}

fn expect_string(metadata: &HashMap<String, Value>, key: &str, expected: &str) -> Result<()> {
    let got = required(metadata, key)?.to_string()?;
    if got != expected {
        candle_core::bail!("unsupported {key} `{got}`, expected `{expected}`");
    }
    Ok(())
}

fn string_set(value: &Value) -> Result<HashSet<String>> {
    value
        .to_vec()?
        .iter()
        .map(|v| v.to_string().cloned())
        .collect()
}

fn int_of(value: &Value) -> Result<i64> {
    Ok(match value {
        Value::U8(v) => *v as i64,
        Value::I8(v) => *v as i64,
        Value::U16(v) => *v as i64,
        Value::I16(v) => *v as i64,
        Value::U32(v) => *v as i64,
        Value::I32(v) => *v as i64,
        Value::U64(v) => *v as i64,
        Value::I64(v) => *v,
        other => candle_core::bail!("expected an integer GGUF value, got {other:?}"),
    })
}

fn parse_signs(
    metadata: &HashMap<String, Value>,
    block: usize,
) -> Result<HashMap<usize, Arc<[f32]>>> {
    let widths = required(metadata, KEY_SIGN_WIDTHS)?
        .to_vec()?
        .iter()
        .map(int_of)
        .collect::<Result<Vec<_>>>()?;
    let values = required(metadata, KEY_SIGN_VALUES)?
        .to_vec()?
        .iter()
        .map(int_of)
        .collect::<Result<Vec<_>>>()?;
    if widths.is_empty() {
        candle_core::bail!("explicit sign mode with empty {KEY_SIGN_WIDTHS}");
    }
    let total: i64 = widths.iter().sum();
    if total != values.len() as i64 {
        candle_core::bail!(
            "{KEY_SIGN_VALUES} has {} entries, {KEY_SIGN_WIDTHS} sums to {total}",
            values.len()
        );
    }
    let mut map = HashMap::new();
    let mut offset = 0;
    for width in widths {
        let width = usize::try_from(width)
            .ok()
            .filter(|w| *w > 0 && w.is_multiple_of(block))
            .ok_or_else(|| Error::msg(format!("bad sign width {width} for block {block}")))?;
        let signs = values[offset..offset + width]
            .iter()
            .map(|v| match v {
                1 => Ok(1.0f32),
                -1 => Ok(-1.0f32),
                other => Err(Error::msg(format!("sign value {other} is not +/-1"))),
            })
            .collect::<Result<Vec<_>>>()?;
        map.insert(width, signs.into());
        offset += width;
    }
    Ok(map)
}

/// In-place normalized Walsh-Hadamard transform in natural order; `x.len()` must be a power of two.
pub fn fwht_normalized(x: &mut [f32]) {
    let n = x.len();
    debug_assert!(n.is_power_of_two());
    let mut len = 1;
    while len < n {
        for chunk in x.chunks_exact_mut(2 * len) {
            let (lo, hi) = chunk.split_at_mut(len);
            for (u, v) in lo.iter_mut().zip(hi.iter_mut()) {
                let (a, b) = (*u, *v);
                *u = a + b;
                *v = a - b;
            }
        }
        len *= 2;
    }
    let scale = 1.0 / (n as f32).sqrt();
    x.iter_mut().for_each(|v| *v *= scale);
}

/// Rewrites each row of `[rows, width]` as `row * H_blk * diag(signs)`.
pub fn unfold_rows(data: &mut [f32], width: usize, block: usize, signs: &[f32]) {
    assert_eq!(signs.len(), width);
    assert!(width.is_multiple_of(block) && data.len().is_multiple_of(width));
    data.par_chunks_exact_mut(width).for_each(|row| {
        for blk in row.chunks_exact_mut(block) {
            fwht_normalized(blk);
        }
        row.iter_mut().zip(signs).for_each(|(v, s)| *v *= s);
    });
}

/// Column map for `ssm_out` stored in grouped V-head order: `dst[h + hd*(k + nk*r)] = src[h + hd*(r + rep*k)]`.
pub fn grouped_to_tiled_cols(data: &mut [f32], width: usize, hd: usize, nk: usize, rep: usize) {
    assert_eq!(width, hd * nk * rep);
    assert!(data.len().is_multiple_of(width));
    data.par_chunks_exact_mut(width).for_each(|row| {
        let src = row.to_vec();
        for k in 0..nk {
            for r in 0..rep {
                let from = hd * (r + rep * k);
                let to = hd * (k + nk * r);
                row[to..to + hd].copy_from_slice(&src[from..from + hd]);
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 40) as f32 / (1u64 << 24) as f32) - 0.5
    }

    fn random(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..n).map(|_| lcg(&mut s)).collect()
    }

    fn random_signs(n: usize, seed: u64) -> Vec<f32> {
        random(n, seed)
            .iter()
            .map(|v| if *v < 0.0 { -1.0 } else { 1.0 })
            .collect()
    }

    fn dense_h(n: usize) -> Vec<f32> {
        let scale = 1.0 / (n as f32).sqrt();
        (0..n * n)
            .map(|i| {
                let (r, c) = (i / n, i % n);
                if (r & c).count_ones() % 2 == 0 {
                    scale
                } else {
                    -scale
                }
            })
            .collect()
    }

    fn close(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert!((x - y).abs() <= tol, "index {i}: {x} vs {y}");
        }
    }

    #[test]
    fn fwht_matches_dense_matrix() {
        for n in [2usize, 8, 64, 1024] {
            let x = random(n, n as u64);
            let h = dense_h(n);
            let want: Vec<f32> = (0..n)
                .map(|r| (0..n).map(|c| h[r * n + c] * x[c]).sum())
                .collect();
            let mut got = x.clone();
            fwht_normalized(&mut got);
            close(&got, &want, 1e-4);
        }
    }

    #[test]
    fn fwht_is_an_involution() {
        let x = random(1024, 3);
        let mut y = x.clone();
        fwht_normalized(&mut y);
        fwht_normalized(&mut y);
        close(&y, &x, 1e-5);
    }

    #[test]
    fn unfolded_weight_equals_runtime_transform() {
        const BLOCK: usize = 1024;
        for width in [BLOCK, 5 * BLOCK, 6 * BLOCK] {
            let rows = 4;
            let stored = random(rows * width, 11);
            let signs = random_signs(width, 12);
            let x = random(width, 13);

            let mut xt: Vec<f32> = x.iter().zip(&signs).map(|(a, s)| a * s).collect();
            xt.chunks_exact_mut(BLOCK).for_each(fwht_normalized);
            let want: Vec<f32> = stored
                .chunks_exact(width)
                .map(|row| row.iter().zip(&xt).map(|(w, a)| w * a).sum())
                .collect();

            let mut unfolded = stored.clone();
            unfold_rows(&mut unfolded, width, BLOCK, &signs);
            let got: Vec<f32> = unfolded
                .chunks_exact(width)
                .map(|row| row.iter().zip(&x).map(|(w, a)| w * a).sum())
                .collect();
            close(&got, &want, 2e-3);
        }
    }

    #[test]
    fn inverse_role_row_is_h_then_signs() {
        const BLOCK: usize = 64;
        let z = random(2 * BLOCK, 5);
        let signs = random_signs(2 * BLOCK, 6);
        let mut want = z.clone();
        want.chunks_exact_mut(BLOCK).for_each(fwht_normalized);
        want.iter_mut().zip(&signs).for_each(|(v, s)| *v *= s);
        let mut got = z;
        unfold_rows(&mut got, 2 * BLOCK, BLOCK, &signs);
        close(&got, &want, 1e-6);
    }

    #[test]
    fn grouped_to_tiled_moves_heads_and_is_invertible_by_index() {
        let (hd, nk, rep) = (2usize, 4usize, 3usize);
        let width = hd * nk * rep;
        let mut row: Vec<f32> = (0..width).map(|i| i as f32).collect();
        grouped_to_tiled_cols(&mut row, width, hd, nk, rep);
        for k in 0..nk {
            for r in 0..rep {
                for h in 0..hd {
                    let grouped = h + hd * (r + rep * k);
                    let tiled = h + hd * (k + nk * r);
                    assert_eq!(row[tiled], grouped as f32);
                }
            }
        }
    }

    fn meta(pairs: Vec<(&str, Value)>) -> HashMap<String, Value> {
        pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect()
    }

    fn strings(names: &[&str]) -> Value {
        Value::Array(names.iter().map(|n| Value::String(n.to_string())).collect())
    }

    fn base_metadata(sign_mode: &str) -> Vec<(&'static str, Value)> {
        vec![
            (KEY_VERSION, Value::U32(1)),
            (KEY_BLOCK_SIZE, Value::U32(4)),
            (KEY_TRANSFORM, Value::String(TRANSFORM_NAME.into())),
            (KEY_AXIS, Value::String(AXIS_NAME.into())),
            (KEY_SIGN_MODE, Value::String(sign_mode.into())),
            (KEY_WEIGHT_NAMES, strings(&["blk.0.ffn_down.weight"])),
            (KEY_INVERSE_NAMES, strings(&["token_embd.weight"])),
            (KEY_GDN_V_GROUPED, Value::Bool(true)),
        ]
    }

    #[test]
    fn absent_version_means_no_fold() {
        assert!(HadamardSpec::from_metadata(&HashMap::new())
            .unwrap()
            .is_none());
    }

    #[test]
    fn parses_explicit_signs_and_roles() {
        let mut pairs = base_metadata(SIGN_MODE_EXPLICIT);
        pairs.push((
            KEY_SIGN_WIDTHS,
            Value::Array(vec![Value::I32(4), Value::I32(8)]),
        ));
        let values: Vec<Value> = [1, -1, 1, 1, -1, -1, 1, 1, 1, -1, 1, -1]
            .into_iter()
            .map(Value::I8)
            .collect();
        pairs.push((KEY_SIGN_VALUES, Value::Array(values)));
        let spec = HadamardSpec::from_metadata(&meta(pairs)).unwrap().unwrap();
        assert_eq!(spec.block_size(), 4);
        assert!(spec.gdn_v_grouped());
        assert_eq!(spec.role("blk.0.ffn_down.weight"), Some(HadamardRole::Fold));
        assert_eq!(spec.role("token_embd.weight"), Some(HadamardRole::Inverse));
        assert_eq!(spec.role("blk.0.ssm_alpha.weight"), None);
        assert_eq!(&*spec.signs_for(4).unwrap(), &[1.0, -1.0, 1.0, 1.0]);
        assert_eq!(spec.signs_for(8).unwrap().len(), 8);
        assert!(spec.signs_for(16).is_err());
        assert!(spec.signs_for(6).is_err());
    }

    #[test]
    fn identity_mode_yields_all_ones() {
        let spec = HadamardSpec::from_metadata(&meta(base_metadata(SIGN_MODE_IDENTITY)))
            .unwrap()
            .unwrap();
        assert_eq!(&*spec.signs_for(8).unwrap(), &[1.0; 8]);
    }

    #[test]
    fn rejects_malformed_metadata() {
        let mut bad_len = base_metadata(SIGN_MODE_EXPLICIT);
        bad_len.push((KEY_SIGN_WIDTHS, Value::Array(vec![Value::I32(4)])));
        bad_len.push((KEY_SIGN_VALUES, Value::Array(vec![Value::I8(1); 3])));
        assert!(HadamardSpec::from_metadata(&meta(bad_len)).is_err());

        let mut bad_sign = base_metadata(SIGN_MODE_EXPLICIT);
        bad_sign.push((KEY_SIGN_WIDTHS, Value::Array(vec![Value::I32(4)])));
        bad_sign.push((KEY_SIGN_VALUES, Value::Array(vec![Value::I8(2); 4])));
        assert!(HadamardSpec::from_metadata(&meta(bad_sign)).is_err());

        let mut bad_transform = base_metadata(SIGN_MODE_IDENTITY);
        bad_transform.retain(|(k, _)| *k != KEY_TRANSFORM);
        bad_transform.push((KEY_TRANSFORM, Value::String("other".into())));
        assert!(HadamardSpec::from_metadata(&meta(bad_transform)).is_err());
    }
}
