# Official-rubric proxy judge: scores captions against REAL frames.
#   PYTHONPATH=. python eval/frame_judge.py --results out/x.json --tasks data/sample_tasks.json
# Guide wording: accuracy = "how faithfully the caption reflects the video
# content"; style = "how well the caption matches the requested tone".
import argparse, asyncio, base64, json, os, subprocess, sys, tempfile
from pathlib import Path
import httpx

sys.stdout.reconfigure(encoding="utf-8")
for line in open(".env"):
    if line.startswith("OPENROUTER_API_KEY="):
        os.environ.setdefault("OPENROUTER_API_KEY", line.partition("=")[2].strip())

JUDGE = os.environ.get("JUDGE_MODEL", "google/gemini-3.1-pro-preview")
STYLE_DEFS = {
    "formal": "Professional, objective, factual tone",
    "sarcastic": "Dry, ironic, lightly mocking",
    "humorous_tech": "Funny, with technology or programming references",
    "humorous_non_tech": "Funny, everyday humour with no technical jargon",
}
RUBRIC = (
    "You are the LLM-Judge of a video-captioning contest. You see frames sampled "
    "from ONE video clip, a requested style, and a caption. Score two dimensions:\n"
    "1. accuracy (0-1): how faithfully the caption reflects the video content - "
    "penalize any claim not supported by the frames; reward correct, specific coverage.\n"
    "2. style (0-1): how well the caption matches the requested tone.\n"
    'Return STRICT JSON: {"accuracy": <float>, "style": <float>, "wrong_claims": ["..."]}'
)


def frames_b64(url: str, n: int = 6) -> list[str]:
    with tempfile.TemporaryDirectory() as td:
        vp = Path(td) / "c.mp4"
        subprocess.run(["curl", "-sL", url, "-o", str(vp)], check=True)
        out: list[str] = []
        dur = float(subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(vp)]).strip() or 30)
        for i in range(1, n + 1):
            fp = Path(td) / f"f{i}.jpg"
            subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                            "-ss", str(round(dur * i / (n + 1), 2)), "-i", str(vp),
                            "-frames:v", "1", "-vf", "scale=640:-2", str(fp)], check=True)
            out.append(base64.b64encode(fp.read_bytes()).decode())
        return out


async def judge_one(client, frames, style, caption):
    content = [{"type": "text", "text": f"Requested style: {style} = {STYLE_DEFS[style]}\n\nCaption:\n{caption}\n\nFrames:"}]
    for b in frames:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}"}})
    for attempt in range(3):
        r = await client.post("https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
            json={"model": JUDGE, "messages": [{"role": "system", "content": RUBRIC},
                  {"role": "user", "content": content}], "max_tokens": 3000, "temperature": 0.0})
        try:
            t = r.json()["choices"][0]["message"]["content"]
            return json.loads(t[t.find("{"):t.rfind("}") + 1])
        except Exception:  # noqa: BLE001 - empty/reasoning-truncated reply: retry
            if attempt == 2:
                print(f"  judge unparseable ({r.status_code}): {r.text[:120]}", file=sys.stderr)
                return {"accuracy": float("nan"), "style": float("nan")}
            await asyncio.sleep(2)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--tasks", required=True)
    a = ap.parse_args()
    tasks = {t["task_id"]: t["video_url"] for t in json.load(open(a.tasks))}
    results = json.load(open(a.results, encoding="utf-8"))
    rows, tot_a, tot_s, n = [], 0.0, 0.0, 0
    async with httpx.AsyncClient(timeout=120) as client:
        for r in results:
            fb = frames_b64(tasks[r["task_id"]])
            js = await asyncio.gather(*[judge_one(client, fb, s, c) for s, c in r["captions"].items()])
            for (s, _), j in zip(r["captions"].items(), js):
                rows.append((r["task_id"], s, j))
                tot_a += j["accuracy"]; tot_s += j["style"]; n += 1
                wc = ("  WRONG: " + "; ".join(j.get("wrong_claims", [])[:2])) if j.get("wrong_claims") else ""
                print(f"{r['task_id']:10s} {s:18s} acc={j['accuracy']:.2f} style={j['style']:.2f}{wc}")
    print(f"\nMEAN accuracy={tot_a/n:.3f} style={tot_s/n:.3f} FINAL={(tot_a+tot_s)/(2*n):.3f}")

asyncio.run(main())
