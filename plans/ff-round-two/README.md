# Fast-forward round two — implementation plans

These nine plans discharge `docs/journal/2026-09-09-fast-forward-second-round-research.md`.
They are written to be executed by a coding agent in order, with two human checkpoints.

## Where the code is

The code lives on branch **`grammar-fast-forward`** (tip `8850efa93`, on top of v0.9.3
`d5ae0f18f`). The branch these plans are stored on, `ff-demo-artifacts`, is an **artifacts-only**
branch: it holds `docs/`, `plans/`, `ff_bench.py`, `RESULTS.md` and a built wheel, and **no
`mistralrs-core/` source tree at all**. Check out `grammar-fast-forward` before touching code, and
land code commits there. Reports produced by these plans go back to `ff-demo-artifacts` under
`plans/ff-round-two/reports/`.

## Three classes of work — do not blur them

The research surfaced three different kinds of item. Treating all of them as "issues to fix" is the
main way this goes wrong.

| Class | Meaning | Plans |
|---|---|---|
| **Prove / disprove** | The research establishes a mechanism, not an outcome. Gather evidence; change nothing on the strength of a suspicion. | 03 (evidence half), 04, 06 |
| **Fix** | An established defect, or documentation that contradicts the code. Fix it. | 02, 03 (gating half), 08 |
| **Decide** | An architectural option whose value is unmeasured. Do not build it until the measurement exists and a human has chosen. | 07 |

## Order and dependencies

```
01 build gate ── everything below depends on it
   ├── 02 splice accounting (fix)          ── prerequisite for 06
   ├── 03 AnyMoE gating (fix) ─┐
   ├── 04 recurrent audit ─────┼── need 05 for their experiment halves
   ├── 05 harness ─────────────┘
   ├── 08 tests + docs (fix, independent)
   └── 06 batch-shape measurement (needs 02 + 05 + a paged build)
          └── ⟨HUMAN CHECKPOINT⟩ → 07 choose B / C / D1
09 runtime-toggle parity — verification only, owned elsewhere
```

Human checkpoints: after **01** if the toolchain is missing or the baseline fails, and after **06**
before any of 07's options is built.

## Rules of engagement

1. **Do not reopen W1–W4** in `docs/journal/2026-09-09-fast-forward-development-plan.md`. They are
   withdrawn with reasons. In particular W3 stands: **do not add a `RecurrentBatchKind::FastForward`
   variant.**
2. **Do not redesign D1, D2, D3, B1 or R1.** They are owned by the development plan, unchanged and
   unstarted. Plans here may size them; they may not restart their design.
3. **One testable change per commit.** Say plainly which claims are confirmed by a run and which
   remain hypotheses.
4. **Report, don't decide.** Where a plan says "hand back", produce the numbers and stop.
5. The second-round entry is one of five overlapping journal entries on this branch. Before
   proposing anything not in these plans, check whether `-splice-review.md`, `-remaining-work.md`,
   `-comprehensiveness-research.md` or `-development-plan.md` already settled it.

## Plans

| # | File | Class | Blocked by |
|---|---|---|---|
| 01 | `01-build-gate.md` | prerequisite | — |
| 02 | `02-splice-accounting.md` | fix | 01 |
| 03 | `03-anymoe.md` | fix + evidence | 01 (fix), 05 (evidence) |
| 04 | `04-recurrent-site-audit.md` | prove/disprove | 01, 05 |
| 05 | `05-harness.md` | supporting | 01 |
| 06 | `06-batch-shape-measurement.md` | prove/disprove | 01, 02, 05, paged build |
| 07 | `07-batch-shape-options.md` | decide | 06 + human |
| 08 | `08-tests-and-docs.md` | fix | 01 |
| 09 | `09-runtime-toggle-dependency.md` | verification | other repo |
