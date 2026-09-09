use crate::sequence::Sequence;

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum StagedBatchState {
    None,
    Homogeneous(usize),
    Mixed,
}

pub(crate) fn staged_batch_state(seqs: &[&mut Sequence]) -> StagedBatchState {
    staged_batch_state_from_widths(seqs.iter().map(|seq| seq.active_staged_speculative_len()))
}

// Scans per-sequence widths for one of three outcomes: no sequence carries any (`None`), every
// carrying sequence agrees on a width (`Homogeneous`), or widths disagree, or some sequences carry
// none while others do (`Mixed`). Used both for staged speculative proposals and for pending
// grammar fast-forward splices below -- the same "all-or-none across the batch" rule over two
// different `Sequence` fields.
pub(crate) fn staged_batch_state_from_widths(
    widths: impl IntoIterator<Item = usize>,
) -> StagedBatchState {
    let mut width = None;
    let mut saw_empty = false;
    for len in widths {
        if len == 0 {
            if width.is_some() {
                return StagedBatchState::Mixed;
            }
            saw_empty = true;
            continue;
        }
        if saw_empty {
            return StagedBatchState::Mixed;
        }
        match width {
            Some(existing) if existing != len => return StagedBatchState::Mixed,
            Some(_) => {}
            None => width = Some(len),
        }
    }
    width.map_or(StagedBatchState::None, StagedBatchState::Homogeneous)
}

pub(crate) fn staged_batch_width(seqs: &[&mut Sequence]) -> Option<usize> {
    match staged_batch_state(seqs) {
        StagedBatchState::Homogeneous(width) => Some(width),
        StagedBatchState::None | StagedBatchState::Mixed => None,
    }
}

/// Homogeneous-width check for `Sequence::pending_ff_tokens`, mirroring `staged_batch_width`
/// above: a batch mixing forced and non-forced sequences, or forced sequences of differing splice
/// length, falls back to `None` (no widening for this step) rather than a ragged decode window.
pub(crate) fn pending_ff_batch_width(seqs: &[&mut Sequence]) -> Option<usize> {
    match staged_batch_state_from_widths(
        seqs.iter().map(|seq| seq.active_pending_ff_tokens().len()),
    ) {
        StagedBatchState::Homogeneous(width) => Some(width),
        _ => None,
    }
}

/// Decides, once per completion step, which sequences' staged fast-forward splices actually
/// reach this step's decode window: a homogeneous-width batch keeps all of them, anything else
/// discards every splice in the batch via `Sequence::discard_pending_ff_tokens`, rolling each
/// back to a consistent matcher state. Must run before the decode window is built and before
/// `scheduled_token_counts` is computed, so that neither reads a splice width about to be
/// discarded.
pub(crate) fn resolve_pending_ff_batch(seqs: &mut [&mut Sequence]) {
    if let StagedBatchState::Homogeneous(_) = staged_batch_state_from_widths(
        seqs.iter().map(|seq| seq.active_pending_ff_tokens().len()),
    ) {
        return;
    }
    for seq in seqs.iter_mut() {
        seq.discard_pending_ff_tokens("batch_shape");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mixed_staged_widths_disable_batched_verification_input() {
        assert_eq!(
            staged_batch_state_from_widths([15, 0]),
            StagedBatchState::Mixed
        );
        assert_eq!(
            staged_batch_state_from_widths([15, 7]),
            StagedBatchState::Mixed
        );
        assert_eq!(
            staged_batch_state_from_widths([15, 15]),
            StagedBatchState::Homogeneous(15)
        );
    }

    fn ff_test_sequence(id: usize) -> Sequence {
        use std::{collections::HashMap, sync::Arc};

        use crate::sampler::Sampler;
        use crate::sequence::{SeqStepType, SequenceGroup, SequenceRecognizer};
        use tokio::sync::{mpsc::channel, Mutex as TokioMutex};

        let (tx, _rx) = channel(1);
        let sampler = Sampler::new(
            None,
            0,
            None,
            None,
            None,
            None,
            None,
            32,
            1.0,
            0.0,
            HashMap::new(),
            vec![],
        )
        .unwrap();
        let group = Arc::new(TokioMutex::new(SequenceGroup::new(1, false, true, None)));
        Sequence::new_waiting(
            vec![1; 4],
            "prompt".to_string(),
            id,
            id as u128,
            1,
            tx,
            sampler,
            vec![],
            vec![],
            None,
            false,
            false,
            group,
            0,
            0,
            SequenceRecognizer::None,
            None,
            None,
            None,
            None,
            None,
            Some(8),
            None,
            None,
            SeqStepType::PromptAndDecode,
            None,
            None,
            None,
            false,
            false,
            vec![],
            None,
        )
    }

    #[test]
    fn resolve_pending_ff_batch_discards_mismatched_splice_widths() {
        let mut seqs: Vec<Sequence> = (0..2).map(ff_test_sequence).collect();
        seqs[0].set_pending_ff_tokens(vec![10, 11, 12]);
        seqs[1].set_pending_ff_tokens(vec![20, 21]);

        let mut refs: Vec<&mut Sequence> = seqs.iter_mut().collect();
        resolve_pending_ff_batch(&mut refs);

        assert!(refs[0].active_pending_ff_tokens().is_empty());
        assert!(refs[1].active_pending_ff_tokens().is_empty());
    }

    #[test]
    fn resolve_pending_ff_batch_keeps_equal_width_splices() {
        let mut seqs: Vec<Sequence> = (0..2).map(ff_test_sequence).collect();
        seqs[0].set_pending_ff_tokens(vec![10, 11]);
        seqs[1].set_pending_ff_tokens(vec![20, 21]);

        let mut refs: Vec<&mut Sequence> = seqs.iter_mut().collect();
        resolve_pending_ff_batch(&mut refs);

        assert_eq!(refs[0].active_pending_ff_tokens(), &[10, 11]);
        assert_eq!(refs[1].active_pending_ff_tokens(), &[20, 21]);
    }
}
