# Consensus Fact Spine Design

## Objective

Raise Track 2 caption accuracy without sacrificing style by keeping the proven
v19 production profile and changing only the shared writer contract. The writer
must select a small, confidence-gated set of facts once, then express that same
set in all four requested voices.

## Evidence behind the change

- v19 scored 0.8942 with dense captions and the three-observer frontier stack.
- v30 scored 0.76 after simultaneously reducing frames, observers, resolution,
  exemplars, output length, and factual coverage.
- The zero-credit three-clip A/B improved mean cross-style content overlap from
  0.110 to 0.151 and reduced length CV from 0.181 to 0.129, but its vague parity
  rule propagated weak single-observer specifics.
- The challenge says humorous non-technical captions must avoid technical
  jargon. It does not say that visible computers, keyboards, phones, machinery,
  or other technology may be omitted.
- Public competitors repeatedly use a shared factual description before style;
  concise output is common, so raw length is not the objective.

## Production design

Keep the v19 production settings unchanged: 10 frames at 896 px, three distinct
frontier observers, Claude Opus writer, tone exemplars, 1600-character safety
cap, and one grouped JSON writer call.

The writer performs these silent steps:

1. Resolve conflicts between observer lists.
2. Build an ordered spine with at least five core factual clauses covering the
   main subject, central action or state, setting, useful context, and temporal
   change when the clip shows one. Five to nine is typical, not a hard maximum;
   complex clips retain additional salient facts when consensus supports them.
3. Treat exact text, brands, locations, counts, fine-grained types, colors,
   species or breeds, and spatial placement as high-risk. Such details require
   agreement from at least two independent observation lists. A single-observer
   fact may survive only after being generalized into a safe claim.
4. Draft the formal caption from every spine fact.
5. Rewrite every spine fact, in the same order, into sarcastic,
   humorous-technical, and humorous-non-technical voices.
6. Run a one-for-one coverage audit before returning JSON so every spine clause
   has an equivalent factual clause in every caption.
7. Permit at most one clearly non-literal humorous clause per creative caption.
   The clause may not imply that an off-screen entity is literally present, and
   absent or not-visible entities must never be mentioned even as disclaimers.
8. Prefer roughly 55-90 words and two to four sentences when the facts support
   that range. Never pad with weak claims or cut facts merely to hit a quota.

## Style semantics

- `formal`: professional, objective, factual, no jokes or direct address.
- `sarcastic`: dry, ironic, lightly mocking. Avoid technical jokes and jargon,
  but retain plainly worded technology that is visibly part of the scene. If a
  model emits hard jargon, normalize it to plain language instead of discarding
  the caption.
- `humorous_tech`: use a clear technology or programming analogy tied to a
  visible subject, state, object, or action while preserving all spine facts.
- `humorous_non_tech`: everyday humor without technical jargon or technical
  metaphors. Visible technology remains literal factual content; hard jargon is
  rewritten in plain language before any fallback is considered.

Off-scene social-platform metaphors such as an Instagram filter or TikTok dance
are rewritten into ordinary visual humor. Identity-linked hairstyle labels such
as `afro` are normalized to neutral visible wording such as `curly hair`, so a
single sensitive word cannot discard an otherwise accurate caption.

## Acceptance gates

- Existing contract, style-filter, retry, hardening, and preflight tests pass.
- A regression test proves that the prompt contains the confidence-gated spine,
  preserves literal visible technology, and no longer says `ZERO technology
  words`.
- Paired local generation reuses identical frames and observations.
- Candidate produces every requested style with zero static fallback and no
  timeout.
- Mean cross-style length CV is at most 0.20 and is not worse than control.
- Content overlap is not worse than control; manual review finds no propagation
  of single-observer high-risk claims, no omitted visible technology, no
  off-scene entities, and no identity-linked hairstyle label.
- Docker linux/amd64 build, mounted I/O contract, JSON validation, image size,
  and runtime checks pass before registry push.
- Submission is allowed only after the image digest and tag are recorded.

## Deliberate exclusions

This version does not add an extra ledger call, per-style candidate generation,
a multimodal selector, a fourth observer, or audio transcription. Those changes
would confound the causal test and increase runtime or cost without current
evidence that they beat the v19 base.
