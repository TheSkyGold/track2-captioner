# Submit Status

Last checked locally: 2026-07-08 18:20 Europe/Paris.

## 2026-07-08 — Critical style-filter fix + Gemma-bonus decision

- **Fixed a score-killing bug:** the pronoun `"it"` was in `TECH_KEYWORDS`
  (`app/models.py`), so any sarcastic / humorous_non_tech caption containing
  "it" (most of them) was misclassified as tech jargon, rejected, and replaced
  by a hardcoded fallback. On unseen clips this both tanks style-match and can
  leak an off-clip hardcoded caption (observed: a kitten caption emitted on the
  city-traffic clip). Root cause found by isolating the v1 sarcastic call:
  Gemma produced a good caption ("...to avoid actually experiencing it") that
  the filter wrongly killed. Guard added: `scripts/test_style_filter.py`.
- **Gemma bonus unlocked without quality loss.** Head-to-head on the 3-clip
  sample with the same visual judge:
  - Qwen3-VL describes + Qwen writes: acc 0.992 / style 0.900 / final **0.946**
  - Qwen3-VL describes + Gemma-3-27B writes: acc 0.950 / style **0.942** / final **0.946**
  Identical final; Gemma wins on style-match and makes the run eligible for the
  separate Gemma prize. Adopted config: **"Qwen sees, Gemma writes"**
  (`OPENROUTER_VLM_MODEL=qwen/qwen3-vl-8b-instruct`,
  `OPENROUTER_STYLE_MODEL=google/gemma-3-27b-it`).
- Remaining Gemma weak spots are the two hardcoded fallbacks that still fire
  (humorous_tech missing a required tech reference; one sarcastic). Beating Qwen
  outright means reducing those fallbacks, not changing models.
- **Validated at scale (checklist item #1).** Ran the Gemma config on the 12
  `eval/stress_clips.json` clips (48 captions), visual judge: acc 0.919 /
  style **0.958** / final **0.939** — holds vs the 3-clip 0.946, style-match
  even higher. Self-check + quality audit pass, no empty captions. 12 clips ran
  in ~6 min at `MAX_CONCURRENCY=2` (under the 8-min budget). Weakest clip:
  `stress_repair_workshop` (acc 0.70, generic). Scores in
  `eval/scores_stress_gemma.json`.
- Offline `scripts/preflight.py` still fully green after the fix.

## Proven Locally

- Python modules compile.
- `scripts/contract_test.py` passes.
- Competitive audit is documented in `AUDIT.md`; Ponytail was used as an
  anti-overengineering method, with no new runtime dependency added.
- Deep research and archive sweep are documented in `RESEARCH.md`.
- Video downloads stream to disk and retry transient HTTP errors.
- Fireworks describe/style calls support comma-separated fallback model chains.
- Groq describe/style fallback is implemented and was validated with real API
  calls on `data/sample_tasks.json`.
- OpenRouter fallback is implemented and currently powers the best measured
  quality-profile run through `qwen/qwen3-vl-8b-instruct`.
- Fireworks credentials are configured locally. `accounts/fireworks/models/qwen3p7-plus`
  responds, and `accounts/fireworks/models/gpt-oss-120b` is available as a
  text fallback, but Fireworks-first judging still hit rate limits and occasional
  non-JSON outputs during local tests.
- Optional direct video/audio preprocessing works locally against the synthetic
  test clip; actual direct model inference still requires `DIRECT_VIDEO_MODEL`
  and `FIREWORKS_API_KEY`.
- Runtime caption normalization is aligned with `eval/self_check.py`: captions
  are capped at 300 characters, formal captions reject exclamation and direct
  first/second-person phrasing, sarcastic captions reject exclamation and tech
  jargon, `humorous_non_tech` rejects tech jargon, and all styles reject
  captions that do not look English/ASCII-safe.
- Offline mock pipeline writes `out/mock_results.json`.
- `eval/self_check.py` passes on mock results.
- `finetune/train_gemma_lora.py --dry-run` loads `finetune/dataset_v2.jsonl`.
- Degraded app run without API keys writes valid non-empty captions.
- `eval/self_check.py` passes on degraded app output.
- Real Groq run wrote `out/groq_results_final.json` for 3 sample clips and
  passed `eval/self_check.py`: 3 rows, 12 captions, all style bans respected.
- Quality-mode run wrote `out/demo_quality_results.json`; it passed
  `eval/self_check.py` and `eval/quality_audit.py`.
- OpenRouter-backed local judge proxy scored the grounded rich-detail
  `out/demo_quality_results.json` at mean accuracy `1.000`, mean style match
  `1.000`, mean final `1.000` over 12 sample captions after grounding fixes
  and fine-grained visible detail enrichment.
- `scripts/quality_gate.py` now bundles self-check, quality audit, and judge
  score summary. It also runs `eval/detail_audit.py` so variants that score well
  but lose visible detail are rejected.
- `eval/grounding_audit.py` is now part of the quality gate for the public
  sample clips; it blocks known unsupported/distracting terms such as doubtful
  eye colors, distant mountains used as a punchline, or code-specific claims
  not visible on screen.
- The describe prompt now asks for conservative `fine_grained_observations`:
  approximate quantities, animal type without breed guesses, leaf/sunlight
  details, jewelry, hand position, nail color, cables, peripherals, and nearby
  objects.
- Evidence-Locked Captioning v1 is implemented behind `EVIDENCE_LOCK_ENABLED=1`:
  it can generate alternate candidates, score visual evidence, repair weak
  captions, and run a post-normalization cross-style repair. It is kept
  experimental until an A/B run beats the current rich-detail canonical output.
- Fireworks-first local judge fallback was hardened: malformed Fireworks score
  JSON is now rejected and the judge continues to OpenRouter/Groq instead of
  accepting a parse error as a real score.
- The live demo cockpit displays videos, captions, per-caption proxy scores,
  weakest captions, validation gates, and copyable top-1 iteration commands.
- Secret literal scan finds no obvious committed credentials.
- Git repository is initialized; generated outputs, Docker test outputs, archives,
  extracted comparison bundles, and flattened duplicates are ignored.
- Added kit files and duplicate/evolution exports were audited; the canonical
  implementation remains `track2_starter/`.
- Docker daemon is available.
- `docker buildx build --platform linux/amd64 --tag track2-captioner:dev --load .` passes.
- Built image inspects as `architecture=amd64 os=linux`.
- Local image inspects at 252,351,797 bytes and reports about 978 MB in
  `docker image ls`, well below the 10 GB limit.
- Docker degraded run against mounted `/input/tasks.json` writes valid
  `/output/results.json` and passed `eval/self_check.py`; latest measured local
  wall time was about 3.16 s with `PER_TASK_TIMEOUT_S=1`.
- `scripts/publish_image.sh` and `scripts/verify_public_image.sh` are present
  and fail safely unless `PUBLIC_IMAGE` is set.
- GitHub Actions workflow `.github/workflows/track2-ci.yml` is present for
  offline preflight and Docker build checks after publishing the repo.

Canonical command:

```bash
python scripts/preflight.py --docker-build --docker-run
```

## Working Temporary Live Mode

For a simple low-rate live mode, use Groq/OpenRouter first and keep Fireworks
available as a fallback:

```bash
PROVIDER_ORDER=groq,fireworks,openrouter \
DESCRIBE_PROVIDER_ORDER=groq,openrouter,fireworks \
STYLE_PROVIDER_ORDER=openrouter,groq,fireworks \
MAX_CONCURRENCY=1 \
NUM_FRAMES=5 \
FRAME_MAX_EDGE=512 \
INPUT_PATH=data/sample_tasks.json \
OUTPUT_PATH=out/groq_results_final.json \
python -m app.main

python eval/self_check.py --results out/groq_results_final.json
python eval/quality_audit.py --results out/groq_results_final.json
```

On the local 3-clip sample run this completed successfully and produced four
validated captions per clip. `MAX_CONCURRENCY=1` is the safe setting for the
current Groq rate limit; concurrency can be raised later after plan limits are
known.

For the strongest measured quality profile on the public 3-clip sample, use
OpenRouter first for describe/style and keep Groq/Fireworks as fallback:

```bash
PROVIDER_ORDER=openrouter,groq,fireworks \
DESCRIBE_PROVIDER_ORDER=openrouter,groq,fireworks \
STYLE_PROVIDER_ORDER=openrouter,groq,fireworks \
MAX_CONCURRENCY=1 \
NUM_FRAMES=8 \
FRAME_MAX_EDGE=640 \
DESCRIBE_MAX_TOKENS=900 \
STYLE_MAX_TOKENS=180 \
INPUT_PATH=data/sample_tasks.json \
OUTPUT_PATH=out/demo_quality_results.json \
python -m app.main

python eval/self_check.py --results out/demo_quality_results.json
python eval/quality_audit.py --results out/demo_quality_results.json --strict
python eval/detail_audit.py --results out/demo_quality_results.json --strict
python eval/local_judge.py --results out/demo_quality_results.json --clips eval/clips.json --out eval/scores_quality_openrouter.json
python scripts/quality_gate.py --results out/demo_quality_results.json --scores eval/scores_quality_openrouter.json
```

## Not Yet Proven

- Fireworks-first describe/style has not beaten the canonical OpenRouter-first
  sample output. Keep it as fallback/A-B infrastructure until a run passes all
  gates and improves the measured score.
- `DIRECT_VIDEO_MODEL` is not set; optional direct video/audio Fireworks
  inference has not been completed.
- Public image push/pull, compressed registry size, and 12-clip real-inference
  runtime must be verified after credentials and registry target are available.

## Top-1 Optimization Backlog

- Re-test Fireworks model availability and rate limits before submission; promote
  it only if `quality_gate.py` improves over the current `1.000` proxy profile.
- Run prompt A/B variants and keep only variants that improve `quality_gate.py`.
- Expand validation to 12+ clips to catch style drift beyond the public samples.
- Publish and anonymously pull the final `linux/amd64` image before submission.

## Next Commands

```bash
python scripts/preflight.py
python scripts/preflight.py --strict --docker-build --docker-run
make submit-check
PUBLIC_IMAGE=ghcr.io/<user>/track2-captioner:final make publish
PUBLIC_IMAGE=ghcr.io/<user>/track2-captioner:final make verify-public
```
