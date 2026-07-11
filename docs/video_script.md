# Demo video script — Track 2 Captioner (2:30)

Format: screen recording + voiceover. Unlisted YouTube is fine. Keep it tight.

---

## 0:00–0:20 — Hook + problem
**Visual:** the three sample clips playing (traffic / kitten / office), four caption chips fading in per clip.
**VO:** "Every short video needs captions — for accessibility, for marketing, for reach. Track 2 asks for four *very different* tones of the same clip, and it scores you on two things at once: are you *accurate*, and does the *tone* land. Those pull against each other. Here's how we win both."

## 0:20–0:50 — The core idea: a verified scene gate
**Visual:** animated diagram — one clip → keyframes → two observer boxes (GPT‑5.5 and Gemini 3.1 Pro), a GPT‑5.5 verification gate, then four Claude Opus 4.8 style writers.
**VO:** "Two frontier vision models independently propose short atomic facts. GPT‑5.5 then re-reads the pixels and can keep only existing fact IDs, never invent or rewrite one. Four Claude Opus 4.8 writers receive that same closed ledger, and a final judge repairs only the style that fails. Accuracy is locked before creativity begins."

## 0:50–1:20 — Proof beyond the public examples
**Visual:** the release dashboard highlighting 40 passing tests, 3/3 public clips in 63.2 seconds, and 48/48 captions on a twelve-clip stress set.
**VO:** "We do not present a local proxy as an official score. Instead, we test the failure modes the hidden judge can punish: invented facts, mixed timelines, missing styles, malformed JSON, provider stalls, and generic humor. Forty targeted tests pass. The final container completes every public example in just over a minute, and a separate twelve-clip stress run produces all forty-eight captions inside the ten-minute limit."

## 1:20–1:45 — It's not a demo, it's a product
**Visual:** the upload web app — paste a URL / drop a file → four captions appear. Then the side‑by‑side model comparison page.
**VO:** "It runs on *any* video, not just the samples — paste a URL or upload a file and get all four styles. Everything is a reproducible Docker image: read /input, write /output, no keys baked into the repo."

## 1:45–2:10 — Use of AMD
**Visual:** the JupyterLab terminal on the AMD Radeon/ROCm box — `rocm-smi`, then Qwen2.5‑VL loading weights 824/824, then the MI300X notebook.
**VO:** "And it's built for AMD. We stood Qwen‑VL up on AMD ROCm to caption real video the hosted APIs won't ingest, with a turnkey notebook to scale it to a single MI300X — 192 gigs, no sharding."

## 2:10–2:30 — Close
**Visual:** the four rich captions on screen, then the title card + repo URL.
**VO:** "Accurate, detailed, four distinct voices, verified on the real clips, deployable anywhere. That's Track 2 Captioner."

---

### Shot list / assets to capture
1. Cockpit `docs/demo-results.html` (the four styled captions, scored).
2. `docs/comparison.html` (six models side by side).
3. `docs/official.html` (captions on the official jury clips).
4. `eval/BENCHMARK_LOG.md` table.
5. The upload app at `:8799` — one live URL run.
6. AMD JupyterLab terminal: `rocm-smi`, Qwen weights loading.
7. `docker pull ghcr.io/theskygold/track2-captioner:latest` + a run writing results.json.
