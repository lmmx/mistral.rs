//! Observability for the grammar fast-forward staging path.
//!
//! `mistralrs_grammar_ff_support_total{supported,reason}` is recorded once per pipeline load,
//! saying whether that pipeline resolved fast-forward support and, if not, which conjunct denied
//! it, so configuration is distinguishable from runtime eligibility. It is accompanied by an
//! `info` log line, which is the only form this reaches in-process callers (`mistralrs-pyo3`
//! installs no Prometheus recorder).
//!
//! No label carries prompt, schema or token content; `reason` is one of five `&'static str`s.

const SUPPORT_METRIC: &str = "mistralrs_grammar_ff_support_total";

/// Whether a pipeline stages grammar fast-forward splices, and why not when it doesn't.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum FfSupport {
    Enabled,
    FlagDisabled,
    NoKvCache,
    XLora,
    /// The pipeline overrides support off regardless of the flag (AnyMoE).
    PipelineUnsupported,
}

impl FfSupport {
    pub(crate) fn supported(self) -> bool {
        matches!(self, FfSupport::Enabled)
    }

    pub(crate) fn as_str(self) -> &'static str {
        match self {
            FfSupport::Enabled => "enabled",
            FfSupport::FlagDisabled => "flag_disabled",
            FfSupport::NoKvCache => "no_kv_cache",
            FfSupport::XLora => "xlora",
            FfSupport::PipelineUnsupported => "pipeline_unsupported",
        }
    }
}

/// The conjunction the text pipelines each spelled out inline before this function existed:
/// the env flag, a KV cache, and not X-LoRA. Returns the same boolean they stored in
/// `GeneralMetadata::supports_grammar_fast_forward`, and records why.
pub(crate) fn resolve_support(no_kv_cache: bool, is_xlora: bool) -> bool {
    let resolved = if !crate::perf_flags::grammar_fast_forward_enabled() {
        FfSupport::FlagDisabled
    } else if no_kv_cache {
        FfSupport::NoKvCache
    } else if is_xlora {
        FfSupport::XLora
    } else {
        FfSupport::Enabled
    };
    record_support(resolved);
    resolved.supported()
}

/// Records a pipeline's resolved fast-forward capability.
pub(crate) fn record_support(resolved: FfSupport) {
    tracing::info!(
        supported = resolved.supported(),
        reason = resolved.as_str(),
        "grammar fast-forward support resolved for this pipeline"
    );
    metrics::counter!(
        SUPPORT_METRIC,
        "supported" => if resolved.supported() { "true" } else { "false" },
        "reason" => resolved.as_str()
    )
    .increment(1);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn support_reasons_are_a_bounded_distinct_set() {
        let supports = [
            FfSupport::Enabled,
            FfSupport::FlagDisabled,
            FfSupport::NoKvCache,
            FfSupport::XLora,
            FfSupport::PipelineUnsupported,
        ];
        let reasons: HashSet<&str> = supports.iter().map(|s| s.as_str()).collect();
        assert_eq!(reasons.len(), 5);
        assert_eq!(supports.iter().filter(|s| s.supported()).count(), 1);
    }
}
