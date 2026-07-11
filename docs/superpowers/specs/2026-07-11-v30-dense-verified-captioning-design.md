# V30 Dense Verified Captioning Design

## Objective

Raise hidden-set caption accuracy and detail density without weakening the V30
anti-hallucination guarantees or the Track 2 runtime and output contract.

## Selected Approach

Use the V30 Verified Scene Gate with three independent frontier observers:
GPT-5.5, Gemini 3.1 Pro, and Claude Opus 4.8. Observations remain candidates,
not truth. GPT-5.5 verifies a closed fact registry against chronological frames,
and only accepted fact text can reach the style writers.

The verifier must preserve the central subject, action or state, and temporal
progression first. It should then prefer non-redundant, clearly visible details
that distinguish the scene: subject appearance or markings, clothing and
accessories, objects being handled or used, setting, background, and lighting.
Risky colors, counts, species, brands, and text still require independent
corroboration and pixel confirmation.

## Caption Generation

Each style writer receives only the ordered verified-fact ledger. The writer
must pack as many useful, non-redundant verified details as naturally fit while
retaining the requested voice. It must never fill a quota by inventing facts.

Formal captions prioritize subject, action, distinctive appearance, interaction
objects, setting, and one compact background or lighting detail. Creative
captions use a literal first sentence with several verified details and a second
scene-specific figurative sentence. The existing auditor compares every literal
claim with both frames and the ledger; failed styles alone are repaired.

Every final caption is limited to 300 characters. The model word ranges remain
bounded so captions finish naturally instead of relying on truncation.

## Runtime And Failure Handling

The three observers run concurrently behind the existing process-wide
OpenRouter semaphore. The verifier, four parallel writers, auditor, and
selective repairs retain their current stage deadlines. The existing global
budget and pre-seeded result file remain unchanged so a provider outage cannot
turn the whole run into a zero.

## Verification

1. Add failing tests for the dense-fact prompt, three-observer profile, and
   300-character hard cap.
2. Run the full V30 contract, style-filter, and 40-test scene-gate suite.
3. Build a clean `linux/amd64` image and verify embedded key encoding without
   exposing key values.
4. Run the three public clips and require zero fallbacks, structural success,
   detail success, and grounding success.
5. Run the broader official-distribution set under the 10-minute budget and
   reject the profile if fallback rate, grounding, or runtime regresses.

## Non-Goals

Do not hardcode facts from the public clips, weaken risky-detail corroboration,
remove the visual auditor, or return to long unbounded captions. Do not add a
new provider path unless the three-observer profile fails its measured gates.
