//! Observability for the grammar fast-forward staging path.
//!
//! - `mistralrs_grammar_ff_support_total{supported,reason}`: recorded once per pipeline load.
//! - `mistralrs_grammar_ff_attempts_total{outcome}`: recorded once per grammar-constrained decode
//!   step, at the `Matcher::consume_ff_tokens` call site in `sampling.rs`. Not recorded for
//!   unconstrained sequences.
//!
//! The support record pre-registers every attempt outcome series at zero in Prometheus.

const SUPPORT_METRIC: &str = "mistralrs_grammar_ff_support_total";
const ATTEMPT_METRIC: &str = "mistralrs_grammar_ff_attempts_total";

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

/// What one grammar-constrained decode step did about a fast-forward splice.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum FfAttempt {
    /// `GeneralMetadata::supports_grammar_fast_forward` is false, so the matcher was never asked.
    Unsupported,
    /// The matcher had stopped, or the sampled token ended the turn, so no splice was possible.
    GrammarStopped,
    /// `consume_ff_tokens` returned no tokens and the matcher is not in error: nothing was forced.
    EmptySplice,
    /// The matcher reported an error while computing the splice; nothing was staged.
    MatcherError,
    /// A non-empty splice was staged on the sequence.
    Staged,
}

impl FfAttempt {
    pub(crate) const ALL: [FfAttempt; 5] = [
        FfAttempt::Unsupported,
        FfAttempt::GrammarStopped,
        FfAttempt::EmptySplice,
        FfAttempt::MatcherError,
        FfAttempt::Staged,
    ];

    pub(crate) fn as_str(self) -> &'static str {
        match self {
            FfAttempt::Unsupported => "unsupported",
            FfAttempt::GrammarStopped => "grammar_stopped",
            FfAttempt::EmptySplice => "empty_splice",
            FfAttempt::MatcherError => "matcher_error",
            FfAttempt::Staged => "staged",
        }
    }
}

/// Resolves `GeneralMetadata::supports_grammar_fast_forward` and records why.
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

/// Records a pipeline's resolved fast-forward capability and pre-registers every attempt
/// outcome series at zero.
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
    for outcome in FfAttempt::ALL {
        metrics::counter!(ATTEMPT_METRIC, "outcome" => outcome.as_str()).increment(0);
    }
}

/// Classifies one grammar-constrained decode step.
///
/// `grammar_active` means the matcher is still running after consuming the sampled token and
/// that token did not end the turn. `matcher_error` and `splice_len` describe the result of
/// `consume_ff_tokens`, and are only meaningful when the first two arguments are both true.
pub(crate) fn classify_attempt(
    supports_fast_forward: bool,
    grammar_active: bool,
    matcher_error: bool,
    splice_len: usize,
) -> FfAttempt {
    if !supports_fast_forward {
        FfAttempt::Unsupported
    } else if !grammar_active {
        FfAttempt::GrammarStopped
    } else if matcher_error {
        FfAttempt::MatcherError
    } else if splice_len == 0 {
        FfAttempt::EmptySplice
    } else {
        FfAttempt::Staged
    }
}

pub(crate) fn record_attempt(outcome: FfAttempt) {
    metrics::counter!(ATTEMPT_METRIC, "outcome" => outcome.as_str()).increment(1);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn unsupported_wins_over_every_runtime_state() {
        for &grammar_active in &[true, false] {
            for &matcher_error in &[true, false] {
                for &splice_len in &[0usize, 3] {
                    assert_eq!(
                        classify_attempt(false, grammar_active, matcher_error, splice_len),
                        FfAttempt::Unsupported
                    );
                }
            }
        }
    }

    #[test]
    fn a_stopped_grammar_is_not_an_empty_splice() {
        assert_eq!(
            classify_attempt(true, false, false, 0),
            FfAttempt::GrammarStopped
        );
    }

    #[test]
    fn matcher_error_wins_over_splice_length() {
        assert_eq!(
            classify_attempt(true, true, true, 0),
            FfAttempt::MatcherError
        );
        assert_eq!(
            classify_attempt(true, true, true, 7),
            FfAttempt::MatcherError
        );
    }

    #[test]
    fn an_empty_splice_without_an_error_is_its_own_outcome() {
        assert_eq!(
            classify_attempt(true, true, false, 0),
            FfAttempt::EmptySplice
        );
    }

    #[test]
    fn a_non_empty_splice_from_a_healthy_matcher_is_staged() {
        assert_eq!(classify_attempt(true, true, false, 1), FfAttempt::Staged);
        assert_eq!(classify_attempt(true, true, false, 12), FfAttempt::Staged);
    }

    #[test]
    fn every_outcome_is_reachable_from_classify_attempt() {
        let reached: HashSet<FfAttempt> = [
            classify_attempt(false, true, false, 0),
            classify_attempt(true, false, false, 0),
            classify_attempt(true, true, false, 0),
            classify_attempt(true, true, true, 0),
            classify_attempt(true, true, false, 4),
        ]
        .into_iter()
        .collect();
        assert_eq!(reached.len(), FfAttempt::ALL.len());
        for outcome in FfAttempt::ALL {
            assert!(
                reached.contains(&outcome),
                "{} unreachable",
                outcome.as_str()
            );
        }
    }

    #[test]
    fn labels_are_a_bounded_distinct_set() {
        let outcomes: HashSet<&str> = FfAttempt::ALL.iter().map(|o| o.as_str()).collect();
        assert_eq!(outcomes.len(), 5);
        let supports = [
            FfSupport::Enabled,
            FfSupport::FlagDisabled,
            FfSupport::NoKvCache,
            FfSupport::XLora,
            FfSupport::PipelineUnsupported,
        ];
        let reasons: HashSet<&str> = supports.iter().map(|s| s.as_str()).collect();
        assert_eq!(reasons.len(), 5);
        assert!(supports.iter().filter(|s| s.supported()).count() == 1);
    }
}
