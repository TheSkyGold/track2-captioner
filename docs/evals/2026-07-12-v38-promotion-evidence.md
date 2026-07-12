# v38 promotion evidence — 2026-07-12

## Official trigger

- The exact `submission-v36` rollback was resubmitted at 09:10 GMT+2.
- Its fresh official result was `0.8917`, rank 10, evaluated at 11:01 GMT+2.
- That fresh result replaced the stale `0.8733` row and triggered promotion of the already-published v38 candidate.

## Promoted artifact

- Image: `ghcr.io/theskygold/track2-captioner:submission-v38`
- Immutable digest: `sha256:817ef98b53298ddba4172ea6ba98d5bc4c994d1cb936b63d8105776dbfd305b5`
- Platform: one `linux/amd64` Docker manifest
- Active-profile delta from v36: add `qwen/qwen3-vl-235b-a22b-instruct` as the fourth independent visual observer.
- Engine, writer, writer system prompt, temperature, frame geometry, concurrency, task timeout, global budget, style exemplars and grounding mode remain unchanged in the active profile.
- Twelve-clip runtime validation: 48 captions, zero fallbacks, approximately 325 seconds, valid output contract.

## Rejected causal experiments

All experiments below remained opt-in and were excluded from the submitted image.

| Experiment | Corpus and judge | Accuracy delta | Style delta | Final delta | Decision |
| --- | --- | ---: | ---: | ---: | --- |
| Timestamp overlay | public three, Mistral Medium 3.1, 12 valid pairs | +0.0250 | -0.0292 | -0.0021 | Reject |
| Textual temporal manifest | public three, Mistral Medium 3.1, 12 valid pairs | +0.0167 | -0.0333 | -0.0083 | Reject |
| Endpoint-aware 10-frame sampling | public three, Mistral Medium 3.1, 12 valid pairs | +0.0125 | 0.0000 | +0.0063 | Continue screening |
| Endpoint-aware 10-frame sampling | public three, Llama 4 Maverick, 12 valid pairs | +0.0083 | +0.0042 | +0.0062 | Continue screening |
| Endpoint-aware 10-frame sampling | five out-of-example clips, Mistral Medium 3.1, 20 valid pairs | 0.0000 | -0.0300 | -0.0150 | Reject |

The endpoint-aware arm improved the three public examples but regressed on the broader five-clip screen. It was rejected to avoid public-example overfitting.

## Delivery verification

- Local full preflight: pass.
- GitHub Actions Track 2 CI run `29185941649`: offline preflight and linux/amd64 Docker build passed.
- Publish run `29185741736`: passed.
- Anonymous manifest inspection and pull: passed.
- Lablab form confirmation: `Submission Updated` and `data has been shipped to DB`.

The public image necessarily contains dedicated contest credentials because the evaluation injects no keys. Treat every credential embedded in a public image as exposed, apply strict limits, and revoke it after evaluation.
