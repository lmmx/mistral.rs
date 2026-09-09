These are **two versions of the same plan**.

We have:

1. **`declarative-forging-flask.md`**
2. **`declarative-forging-flask-v1-specdec-reuse.md`**

The second one is an earlier/alternative design that Claude subsequently revised.

### The key difference

The two plans are trying to implement the same feature: **grammar fast-forward tokens** in `mistral.rs`.

But they make a different architectural choice about how to get multi-token fast-forward windows through GDN.

|                                       | `-v1-specdec-reuse`                                               | `declarative-forging-flask`            |
| ------------------------------------- | ----------------------------------------------------------------- | -------------------------------------- |
| GDN approach                          | Reuse `RecurrentBatchKind::SpeculativeDecode`                     | Relax `RecurrentBatchKind::Decode`     |
| Modify `gdn/backend.rs`?              | **No**                                                            | **Yes**                                |
| Reuse speculative-decoding machinery? | **Yes**                                                           | **No**                                 |
| Main concern                          | CUDA speculative checkpoint machinery may have unintended effects | Need to prove widened `Decode` is safe |
| Design maturity                       | Earlier/alternative design                                        | **Later, more conservative design**    |

You can see the older plan explicitly saying:

> “Reuse `RecurrentBatchKind::SpeculativeDecode` for FF-widened decode” 

while the newer plan deliberately reverses that decision:

> “Relax `causal_conv1d`'s bail in `gdn/backend.rs` directly ... rather than reusing `RecurrentBatchKind::SpeculativeDecode`.” 

### Why the change of direction?

The **`-v1-specdec-reuse` plan was the first version of the plan**. It initially reasoned:

> Existing `SpeculativeDecode` already supports multi-token GDN windows, so why not use it?

Then the later investigation identified a subtle problem: **speculative decoding and grammar fast-forwarding have different semantics**.

Speculative decoding means:

```text
propose tokens
      ↓
run them through model
      ↓
verify
      ↓
accept some / reject some
      ↓
possibly roll state back
```

Grammar fast-forward means:

```text
grammar says these tokens MUST occur
      ↓
there is nothing to verify
      ↓
all tokens are guaranteed to be accepted
```

The later plan therefore decided it was cleaner **not to pretend grammar fast-forward is speculative decoding**, particularly because `SpeculativeDecode` activates checkpoint/rollback machinery on CUDA. The later plan explains that reasoning at length. 

So the progression is roughly:

**Plan A**

> “We can probably reuse speculative decode.”

↓

**More investigation**

> “Actually, that couples two semantically different mechanisms and potentially activates CUDA rollback machinery.”

↓

**Plan B**

> “Keep FF as ordinary `Decode`, but teach the GDN decode path to accept a multi-token window.”

That's a meaningful architectural refinement, not just a rewritten copy.

### Which one should you care about?

**`declarative-forging-flask.md` is the newer/final plan.**

The strongest evidence is its section:

> **Current implementation state (uncommitted, on local branch `grammar-fast-forward`)**

and it records what had actually been implemented versus what remained. 

The `-v1-specdec-reuse` version is essentially the **superseded design iteration**.

Also interesting: the later plan says the GDN change is *not* just theoretical—it explicitly records that the existing `SpeculativeDecode` machinery was investigated and intentionally rejected rather than forgotten. 
