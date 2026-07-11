# V30 Dense Verified Captioning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce richer, strictly grounded Track 2 captions from the V30 closed fact ledger while enforcing a 300-character output cap.

**Architecture:** Keep the existing observe, verify, parallel-write, audit, and selective-repair pipeline. Add one independent observer, improve fact and writer priorities generically, and make both the verified gate and Docker profile enforce the starter cap.

**Tech Stack:** Python 3.11, asyncio, httpx, unittest, FFmpeg, Docker, OpenRouter.

---

### Task 1: Lock The Dense Grounded Prompt Contract

**Files:**
- Modify: `scripts/test_verified_scene_gate.py:339`
- Modify: `app/verified_scene.py:172`
- Modify: `app/verified_scene.py:188`
- Modify: `app/verified_scene.py:383`

- [ ] **Step 1: Write the failing prompt tests**

Add these assertions to `StylePromptTests`:

```python
def test_observer_and_verifier_prioritize_distinctive_nonredundant_details(self) -> None:
    assert verified_observer_system is not None
    observer = verified_observer_system.lower()
    verifier = verified_scene.VERIFIER_SYSTEM.lower()
    for phrase in (
        "distinctive appearance or markings",
        "clothing and accessories",
        "objects being handled or used",
    ):
        self.assertIn(phrase, observer)
    self.assertIn("non-redundant distinctive details", verifier)
    self.assertNotIn("nails", observer)
    self.assertNotIn("jewelry", observer)

def test_style_prompt_packs_verified_detail_without_forcing_a_quota(self) -> None:
    assert build_style_prompt is not None
    facts = [
        "A person types at a desk.",
        "A large monitor stands in front of the person.",
        "A pendant hangs from a necklace.",
        "A leafy plant stands behind the desk.",
    ]
    for style in style_limits:
        _, user = build_style_prompt(style, facts)
        low = user.lower()
        self.assertIn("as many useful, non-redundant verified details", low)
        self.assertIn("never invent a detail to fill a quota", low)
        self.assertIn("distinctive appearance", low)
        self.assertIn("objects being handled or used", low)
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```powershell
$env:PYTHONPATH='.'
python -m unittest scripts.test_verified_scene_gate.StylePromptTests -v
```

Expected: the two new tests fail because the dense-detail instructions are absent.

- [ ] **Step 3: Implement the minimal generic prompt changes**

In `VERIFIED_OBSERVER_SYSTEM`, request non-redundant facts in this order after the core action:

```python
"After the core subject and action, preserve non-redundant distinctive details: "
"distinctive appearance or markings, clothing and accessories, objects being "
"handled or used, setting, background, and lighting. "
```

In `VERIFIER_SYSTEM`, add:

```python
"After the mandatory core facts, prefer non-redundant distinctive details over "
"repeated generic descriptions of the same setting. "
```

Replace the single-background-detail instruction in `build_style_prompt` with:

```python
"Pack as many useful, non-redundant verified details as naturally fit, prioritizing "
"the main subject and action, distinctive appearance or markings, clothing or "
"accessories, objects being handled or used, setting, background, and lighting. "
"Never invent a detail to fill a quota. "
```

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run the same `unittest` command. Expected: all `StylePromptTests` pass.

- [ ] **Step 5: Commit the prompt contract**

```powershell
git add app/verified_scene.py scripts/test_verified_scene_gate.py
git commit -m "feat: preserve dense verified scene details"
```

### Task 2: Enforce The 300-Character Contract

**Files:**
- Modify: `scripts/test_verified_scene_gate.py:447`
- Modify: `app/verified_scene.py:19`
- Modify: `Dockerfile:60`

- [ ] **Step 1: Write the failing cap and profile test**

Change the character-cap fixture and add a Docker profile assertion:

```python
def test_character_cap_is_independent_from_word_count(self) -> None:
    assert caption_quality_issues is not None
    long_tokens = " ".join("abcdefgh" for _ in range(38))
    issues = caption_quality_issues("formal", long_tokens)
    self.assertNotIn("too_few_words", issues)
    self.assertNotIn("too_many_words", issues)
    self.assertIn("too_many_chars", issues)

def test_submission_profile_uses_three_observers_and_300_char_cap(self) -> None:
    dockerfile = (Path(__file__).resolve().parents[1] / "Dockerfile").read_text(
        encoding="utf-8"
    )
    self.assertIn(
        "ENSEMBLE_OBSERVERS=openai/gpt-5.5,google/gemini-3.1-pro-preview,"
        "anthropic/claude-opus-4.8",
        dockerfile,
    )
    self.assertIn("MAX_CAPTION_CHARS=300", dockerfile)
```

- [ ] **Step 2: Run and verify RED**

Run:

```powershell
$env:PYTHONPATH='.'
python -m unittest scripts.test_verified_scene_gate.LengthValidationTests -v
```

Expected: the 341-character fixture is accepted by the current 420-character gate.

- [ ] **Step 3: Implement the cap and third-observer profile**

Set:

```python
MAX_VERIFIED_CAPTION_CHARS = 300
```

Set these Docker values:

```dockerfile
ENSEMBLE_OBSERVERS=openai/gpt-5.5,google/gemini-3.1-pro-preview,anthropic/claude-opus-4.8
MAX_CAPTION_CHARS=300
```

Update the Docker comment from two to three independent observers.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run:

```powershell
$env:PYTHONPATH='.'
python -m unittest scripts.test_verified_scene_gate.LengthValidationTests scripts.test_verified_scene_gate.StylePromptTests -v
```

Expected: all focused tests pass.

- [ ] **Step 5: Commit the bounded submission profile**

```powershell
git add Dockerfile app/verified_scene.py scripts/test_verified_scene_gate.py
git commit -m "feat: bound dense verified submission captions"
```

### Task 3: Run Local Regression Gates

**Files:**
- Verify: `app/verified_scene.py`
- Verify: `app/ensemble.py`
- Verify: `app/main.py`

- [ ] **Step 1: Run the complete deterministic suite**

```powershell
$env:PYTHONPATH='.'
python scripts/contract_test.py
python scripts/test_style_filter.py
python scripts/test_verified_scene_gate.py
```

Expected: contract and style checks pass; all 43 scene-gate tests pass.

- [ ] **Step 2: Run syntax and diff checks**

```powershell
python -m compileall -q app scripts
git diff --check HEAD~2..HEAD
```

Expected: both commands exit zero.

### Task 4: Build And Compare Real Inference

**Files:**
- Generate: `out/bench_v30_dense/output/results.json`
- Compare: `out/bench_v30_clean/output/results.json`

- [ ] **Step 1: Build the clean linux/amd64 image**

Run from the worktree. The command reads the ignored parent `.env` without
printing the key:

```powershell
$envMap = @{}
Get-Content -LiteralPath '..\..\.env' | ForEach-Object {
    if ($_ -match '^([^#=]+)=(.*)$') {
        $name = $Matches[1].Trim([char]0xFEFF)
        $value = $Matches[2].Trim([char]34).Trim([char]39).Trim([char]0xFEFF)
        $envMap[$name] = $value
    }
}
if (-not $envMap['OPENROUTER_API_KEY']) { throw 'Missing OpenRouter key' }
docker build --platform linux/amd64 `
    --build-arg "OPENROUTER_API_KEY=$($envMap['OPENROUTER_API_KEY'])" `
    -t track2-captioner:v30-dense .
```

Expected: Docker exits zero and creates `track2-captioner:v30-dense`.

- [ ] **Step 2: Verify image architecture and key encoding**

```powershell
$tag = 'track2-captioner:v30-dense'
$envs = docker inspect $tag --format '{{json .Config.Env}}' | ConvertFrom-Json
$line = $envs | Where-Object { $_.StartsWith('OPENROUTER_API_KEY=') } | Select-Object -First 1
$value = if ($line) { $line.Substring(19) } else { '' }
$first = if ($value.Length) { [int][char]$value[0] } else { -1 }
docker inspect $tag --format 'architecture={{.Architecture}} os={{.Os}} size={{.Size}}'
Write-Output "key_length=$($value.Length) first_codepoint=$first"
```

Expected: `architecture=amd64`, `os=linux`, non-zero key length, and
`first_codepoint=115`.

- [ ] **Step 3: Run the three public clips through the Docker contract**

```powershell
$root = (git rev-parse --show-toplevel).Trim()
$verify = Join-Path $root 'out\bench_v30_dense'
$inputDir = Join-Path $verify 'input'
$outputDir = Join-Path $verify 'output'
New-Item -ItemType Directory -Force -Path $inputDir,$outputDir | Out-Null
Copy-Item data\sample_tasks.json (Join-Path $inputDir 'tasks.json') -Force
Remove-Item (Join-Path $outputDir 'results.json') -Force -ErrorAction SilentlyContinue
$sw = [Diagnostics.Stopwatch]::StartNew()
docker run --rm -v "${inputDir}:/input:ro" -v "${outputDir}:/output" `
    track2-captioner:v30-dense
$sw.Stop()
Write-Output "elapsed_seconds=$($sw.Elapsed.TotalSeconds)"
```

Expected: exit zero and a fresh `out/bench_v30_dense/output/results.json`.

- [ ] **Step 4: Run all output gates**

```powershell
python eval/self_check.py --results out/bench_v30_dense/output/results.json
python scripts/fallback_scan.py --results out/bench_v30_dense/output/results.json --strict
python eval/quality_audit.py --results out/bench_v30_dense/output/results.json --strict
python eval/detail_audit.py --results out/bench_v30_dense/output/results.json --strict
python eval/grounding_audit.py --results out/bench_v30_dense/output/results.json --strict
```

Acceptance: 12 valid captions, zero fallbacks, no grounding warnings, no garbled text, and total time below 180 seconds.

- [ ] **Step 5: Inspect the factual detail delta**

```powershell
$rows = Get-Content -Raw out\bench_v30_dense\output\results.json | ConvertFrom-Json
$formal = ($rows | Where-Object task_id -eq 'v3').captions.formal
if ($formal -notmatch '(desktop|large) monitor') { throw 'v3 lost the desktop monitor' }
if ($formal -notmatch '(cross pendant|pendant)') { throw 'v3 lost the pendant' }
if ($formal -notmatch '(bun|pink nail|mouse|ceiling light|orange|beige)') {
    throw 'v3 gained no additional verified distinctive detail'
}
```

Expected: all three checks pass. These checks validate the public example only;
they must not appear in runtime caption code or prompts.

### Task 5: Generalization, Publish, And Submit

**Files:**
- Generate: `out/bench_v30_dense_official/output/results.json`
- Modify: `README.md`
- Modify: `SUBMISSION.md`

- [ ] **Step 1: Run the official-distribution task set**

```powershell
$root = (git rev-parse --show-toplevel).Trim()
$verify = Join-Path $root 'out\bench_v30_dense_official'
$inputDir = Join-Path $verify 'input'
$outputDir = Join-Path $verify 'output'
New-Item -ItemType Directory -Force -Path $inputDir,$outputDir | Out-Null
Copy-Item data\official_new12.json (Join-Path $inputDir 'tasks.json') -Force
Remove-Item (Join-Path $outputDir 'results.json') -Force -ErrorAction SilentlyContinue
$sw = [Diagnostics.Stopwatch]::StartNew()
docker run --rm -v "${inputDir}:/input:ro" -v "${outputDir}:/output" `
    track2-captioner:v30-dense
$sw.Stop()
$rows = Get-Content -Raw (Join-Path $outputDir 'results.json') | ConvertFrom-Json
if ($rows.Count -ne 12) { throw "Expected 12 rows, got $($rows.Count)" }
if ($sw.Elapsed.TotalSeconds -ge 535) { throw 'Official-set run exceeded budget' }
python scripts\fallback_scan.py --results `
    out\bench_v30_dense_official\output\results.json --strict
```

Expected: exit zero, 12 rows, 48 captions, runtime below 535 seconds, and zero
detected fallbacks.

- [ ] **Step 2: Run structural, detail, and grounding gates**

Reject the candidate if it violates the contract, exceeds 300 characters, regresses the deterministic audits, or emits provider fallbacks.

- [ ] **Step 3: Update submission documentation with measured facts only**

Document the three-observer Verified Scene Gate, actual demo and official-set runtime, fallback count, and 300-character cap. Do not claim an official score before evaluation.

- [ ] **Step 4: Commit, push, and rebuild GHCR**

Push the reviewed V30 dense commits to `TheSkyGold/track2-captioner` main, dispatch `publish.yml`, and wait for success.

- [ ] **Step 5: Pull anonymously and rerun the contract**

Require public anonymous pull, `linux/amd64`, clean key codepoint, valid results, and zero fallbacks before submission.

- [ ] **Step 6: Submit and poll the live leaderboard**

Submit only the verified GHCR image, then use `python scripts/lb_poll.py` until the new `evaluatedAt` timestamp appears. Record the live score separately from proxy measurements.
