# Consensus Fact Spine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the vague v32 parity prompt with a confidence-gated fact spine, prove the behavior locally, build the linux/amd64 image, and submit the verified image.

**Architecture:** Preserve the full v19 inference profile and make one causal writer-prompt change. The single writer call first selects five to seven consensus facts, then renders the same ordered facts in all four styles while keeping literal visible technology in every voice. The existing paired A/B harness reuses frames and observations so local evidence isolates the prompt.

**Tech Stack:** Python 3.11, httpx, FFmpeg, Docker Buildx, OpenRouter-compatible APIs, Qwen3-VL on ROCm for zero-credit development, Git/GitHub, public container registry.

---

### Task 1: Lock the approved design

**Files:**
- Create: `docs/superpowers/specs/2026-07-11-consensus-fact-spine-design.md`
- Create: `docs/superpowers/plans/2026-07-11-consensus-fact-spine.md`

- [ ] **Step 1: Verify the design covers the discovered failures**

Run:

```powershell
rg -n "single-observer|visible technology|five to seven|55-90|same order" docs/superpowers/specs/2026-07-11-consensus-fact-spine-design.md
```

Expected: every failure class and the bounded-length guidance appear.

- [ ] **Step 2: Commit the design and plan**

```powershell
git add docs/superpowers/specs/2026-07-11-consensus-fact-spine-design.md docs/superpowers/plans/2026-07-11-consensus-fact-spine.md
git commit -m "docs: specify consensus fact spine"
```

### Task 2: Write the prompt regression tests first

**Files:**
- Modify: `scripts/test_fact_parity.py`
- Test: `scripts/test_fact_parity.py`

- [ ] **Step 1: Replace the old vague-contract assertions with the approved behavior**

Add assertions equivalent to:

```python
required = (
    "CONSENSUS FACT SPINE",
    "five to seven",
    "at least two independent observation lists",
    "same ordered spine facts",
    "literal visible technology remains factual scene content",
    "one clearly non-literal",
    "55-90 words",
)
for phrase in required:
    assert phrase.lower() in prompt.lower(), phrase
assert "ZERO technology words" not in prompt
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```powershell
python scripts/test_fact_parity.py
```

Expected: FAIL on `CONSENSUS FACT SPINE` because production still contains the
old `FACT PARITY CONTRACT`.

### Task 3: Implement the minimal consensus fact-spine prompt

**Files:**
- Modify: `app/ensemble.py:91-139`
- Test: `scripts/test_fact_parity.py`

- [ ] **Step 1: Replace `_FACT_PARITY_RULE` with the approved silent workflow**

The rule must state all of the following directly: five to seven ordered facts,
two-observer support for high-risk specifics, safe generalization of a generic
single-observer detail, all spine facts in every style, one non-literal clause,
and non-binding 55-90-word guidance.

- [ ] **Step 2: Correct the style definitions**

Replace `ZERO technology words` with:

```python
"sarcastic = dry ironic wit, with no technical joke or jargon; "
"humorous_non_tech = warm everyday humor with no technical jargon or tech metaphor. "
"Literal visible technology remains factual scene content in every style and must be named plainly. "
```

- [ ] **Step 3: Run the prompt test and verify GREEN**

```powershell
python scripts/test_fact_parity.py
```

Expected: `FACT-SPINE TEST OK` and exit code 0.

- [ ] **Step 4: Run adjacent regression tests**

```powershell
python scripts/test_style_filter.py
python scripts/test_hardening.py
python scripts/contract_test.py
python scripts/test_429_retry.py
```

Expected: all commands exit 0.

- [ ] **Step 5: Commit the prompt change**

```powershell
git add app/ensemble.py scripts/test_fact_parity.py
git commit -m "feat: gate captions on a consensus fact spine"
```

### Task 4: Run the paired zero-credit notebook gate

**Files:**
- Use: `scripts/fact_parity_ab.py`
- Produce remotely: `/workspace/ab_fact_spine_v33/ab_report.json`
- Produce remotely: `/workspace/ab_fact_spine_v33/control_results.json`
- Produce remotely: `/workspace/ab_fact_spine_v33/candidate_results.json`

- [ ] **Step 1: Pull the committed branch on the AMD notebook**

```bash
cd /workspace/track2-captioner
git fetch pub codex/v32-v19-fact-parity
git reset --hard pub/codex/v32-v19-fact-parity
```

- [ ] **Step 2: Run the exact paired three-clip profile against the local endpoint**

```bash
PYTHONPATH=. OPENROUTER_BASE_URL=http://127.0.0.1:8000/v1 \
OPENROUTER_API_KEY=local-zero-credit MAX_CAPTION_CHARS=1600 NUM_FRAMES=10 \
FRAME_MAX_EDGE=896 SCENE_DETECT_ENABLED=0 TIMESTAMP_FRAMES=0 STYLE_EXEMPLARS=1 \
STRICT_GROUNDING=0 ENSEMBLE_CONCISE=0 WRITER_TEMP=0 \
python -u scripts/fact_parity_ab.py --tasks data/sample_tasks.json \
  --output-dir /workspace/ab_fact_spine_v33 --observers 2 \
  --observer-max-tokens 320 --writer-max-tokens 900 --http-timeout 600
```

Expected: three checkpoints, 12 captions per arm, zero static fallbacks.

- [ ] **Step 3: Apply the acceptance gates**

Reject the candidate if any style is missing, a fallback fires, length CV exceeds
0.20, content overlap regresses, visible technology disappears, or a high-risk
single-observer detail is repeated across styles.

### Task 5: Verify the full container contract

**Files:**
- Verify: `Dockerfile`
- Verify: `scripts/preflight.py`
- Produce: local linux/amd64 Docker image

- [ ] **Step 1: Run the full local test suite and static checks**

```powershell
python scripts/test_fact_parity.py
python scripts/test_qwen3vl_local_server.py
python scripts/test_style_filter.py
python scripts/test_hardening.py
python scripts/contract_test.py
python scripts/test_429_retry.py
python scripts/preflight.py
git diff --check
```

Expected: every command exits 0 and `git diff --check` prints nothing.

- [ ] **Step 2: Build linux/amd64**

```powershell
docker buildx build --platform linux/amd64 --load -t track2-captioner:v33 .
```

Expected: exit code 0 and compressed image under 10 GB.

- [ ] **Step 3: Run the mounted input/output contract**

Use a temporary input directory containing `tasks.json`, mount a writable output
directory, run the image, then parse `results.json` and verify all task/style keys.

### Task 6: Publish and submit the verified image

**Files:**
- No source changes expected
- Record: immutable registry digest and submission timestamp

- [ ] **Step 1: Tag and push the verified image**

```powershell
docker tag track2-captioner:v33 ghcr.io/theskygold/track2-captioner:v33
docker push ghcr.io/theskygold/track2-captioner:v33
docker buildx imagetools inspect ghcr.io/theskygold/track2-captioner:v33
```

Expected: public manifest includes linux/amd64 and an immutable `sha256:` digest.

- [ ] **Step 2: Submit through the authenticated hackathon page**

Open the existing submission form, replace the image reference with the verified
`v33` tag or digest, confirm the Track 2 title, submit once, and capture the new
`last resubmitted` timestamp.

- [ ] **Step 3: Monitor the result**

Record the score and rank when evaluation completes. Do not resubmit another
variant without new evidence.
