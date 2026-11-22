#!/usr/bin/env python3
"""Image-to-video worker: one still image + one prompt -> one short clip.

Designed to run as a single self-contained container job on a rented GPU node
(Nosana, or any Docker host with an NVIDIA GPU). No orchestration framework and
no cloud SDK: everything arrives through env vars, so the same image runs
unchanged on a laptop, a CI runner or a decentralised GPU market.

That last target shapes the I/O. A market node has no host directory to mount
and no artifact store to write to, so the worker supports both modes:

  local  - job file and stills in a mounted directory, clips written next to them
  market - the job JSON arrives inline in an env var, stills are fetched from
           URLs, and results leave through stdout (always) plus an optional
           presigned PUT per clip. Presigned URLs expire and are scoped to one
           object, so nothing long-lived is ever handed to a rented machine.

Inputs (env)
  I2V_JOB_JSON  job JSON inline; takes precedence over the job file      (market)
  I2V_IN        directory holding the job file and source images (default /work/in)
  I2V_OUT       directory the clip is written to                  (default /work/out)
  I2V_JOB       job file name inside I2V_IN                        (default job.json)
  I2V_MODEL     diffusers repo id, or a local path staged by a Nosana HF
                resource                   (default Wan-AI/Wan2.2-TI2V-5B-Diffusers)
  I2V_STEPS     denoising steps                                    (default 30)
  I2V_FRAMES    frame count, 16 fps                                (default 81 = ~5 s)
  I2V_WIDTH     output width                                       (default 1280)
  I2V_HEIGHT    output height                                      (default 704)
  I2V_GUIDANCE  classifier-free guidance scale                     (default 3.5)
  I2V_TIMEOUT   hard wall-clock budget in seconds, 0 = off        (default 2700)

job entry
  {"name": "hook_01", "image": "hook_01.png", "prompt": "...",
   "negative": "...", "put_url": "<optional presigned PUT for the mp4>"}
  "image" may also be an http(s) URL. One entry or a list, rendered in order.

Output
  $I2V_OUT/<name>.mp4 plus $I2V_OUT/<name>.json (timings, so cost per clip can
  be computed from the node's hourly rate without trusting our own numbers), and
  one `[i2v-result] {...}` line per clip on stdout - the only channel guaranteed
  to survive a market job, since `nosana job get` returns the logs.

Why the 5B model is the default: it fits in 24 GB with VAE tiling, which is what
the cheap end of the GPU market actually offers. Set I2V_MODEL to
Wan-AI/Wan2.2-I2V-A14B-Diffusers on an 80 GB node for the higher-fidelity path.
"""
from __future__ import annotations

import json
import os
import signal
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

NEG_DEFAULT = (
    "blurry, distorted, deformed hands, extra fingers, morphing, scene change, "
    "low quality, watermark, text, letters"
)


def env_int(key: str, default: int) -> int:
    raw = os.environ.get(key, "").strip()
    try:
        return int(raw) if raw else default
    except ValueError:
        print(f"[i2v] {key}={raw!r} is not an integer, using {default}", flush=True)
        return default


def env_float(key: str, default: float) -> float:
    raw = os.environ.get(key, "").strip()
    try:
        return float(raw) if raw else default
    except ValueError:
        print(f"[i2v] {key}={raw!r} is not a number, using {default}", flush=True)
        return default


def arm_timeout(seconds: int) -> None:
    """A rented GPU bills by the second. Never let a hung job run to the credit floor."""
    if seconds <= 0:
        return

    def _die(signum, frame):  # noqa: ARG001
        print(f"[i2v] wall-clock budget {seconds}s exhausted, aborting", flush=True)
        sys.exit(124)

    signal.signal(signal.SIGALRM, _die)
    signal.alarm(seconds)


def load_jobs(in_dir: Path, job_name: str) -> list[dict]:
    """Inline env JSON first, mounted file second - a market node has no mount."""
    inline = os.environ.get("I2V_JOB_JSON", "").strip()
    if inline:
        try:
            data = json.loads(inline)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"[i2v] I2V_JOB_JSON is not valid JSON: {exc}") from exc
        source = "I2V_JOB_JSON"
    else:
        path = in_dir / job_name
        if not path.exists():
            raise SystemExit(
                f"[i2v] no job: {path} missing and I2V_JOB_JSON unset"
            )
        data = json.loads(path.read_text(encoding="utf-8"))
        source = str(path)
    jobs = data if isinstance(data, list) else [data]
    for i, job in enumerate(jobs):
        for key in ("name", "image", "prompt"):
            if not job.get(key):
                raise SystemExit(f"[i2v] job[{i}] from {source} is missing field {key!r}")
    print(f"[i2v] {len(jobs)} job(s) from {source}", flush=True)
    return jobs


def resolve_image(spec: str, in_dir: Path, work: Path) -> Path:
    """Local path when mounted, downloaded file when the job hands us a URL."""
    if not spec.startswith(("http://", "https://")):
        return in_dir / spec
    dest = work / Path(spec.split("?", 1)[0]).name
    if dest.exists():
        return dest
    work.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(spec, timeout=120) as resp, dest.open("wb") as fh:
        fh.write(resp.read())
    print(f"[i2v] fetched {dest.name} ({dest.stat().st_size // 1024} KiB)", flush=True)
    return dest


def put_file(path: Path, url: str) -> bool:
    """Upload one clip to a presigned PUT URL.

    Presigned beats credentials here: the URL expires and grants write to exactly
    one object, so a rented node that turns out to be hostile gets nothing useful.
    Failure is reported, never fatal - the timing record already went to stdout.
    """
    body = path.read_bytes()
    req = urllib.request.Request(
        url, data=body, method="PUT",
        headers={"Content-Type": "video/mp4", "Content-Length": str(len(body))},
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            ok = 200 <= resp.status < 300
        print(f"[i2v] {path.name}: upload {'ok' if ok else 'rejected'}", flush=True)
        return ok
    except (urllib.error.URLError, OSError) as exc:
        print(f"[i2v] {path.name}: upload failed: {exc}", flush=True)
        return False


def build_pipeline(model: str):
    import torch
    from diffusers import WanImageToVideoPipeline

    t0 = time.time()
    # Both tiers load through WanImageToVideoPipeline even though the 5B repo
    # declares WanPipeline in its model_index: its image encoder is an optional
    # component and its `expand_timesteps` flag is honoured by the i2v pipeline,
    # which is what makes one container able to serve both tiers.
    pipe = WanImageToVideoPipeline.from_pretrained(model, torch_dtype=torch.bfloat16)
    # Offload + tiling is not an optimisation here, it is the reason this runs at
    # all on a 24 GB card: without VAE tiling the decode step OOMs at 1280x704.
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()
    return pipe, round(time.time() - t0, 1)


def gpu_name() -> str:
    try:
        import torch

        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001 - diagnostics only, never fail the job for this
        return "unknown"


def main() -> int:
    in_dir = Path(os.environ.get("I2V_IN", "/work/in"))
    out_dir = Path(os.environ.get("I2V_OUT", "/work/out"))
    model = os.environ.get("I2V_MODEL", "Wan-AI/Wan2.2-TI2V-5B-Diffusers")
    steps = env_int("I2V_STEPS", 30)
    frames = env_int("I2V_FRAMES", 81)
    width = env_int("I2V_WIDTH", 1280)
    height = env_int("I2V_HEIGHT", 704)
    # 3.5 is what the production pipeline uses on the 14B tier; the 5B tier
    # prefers a little more. Tunable because it is a quality knob, not a constant.
    guidance = env_float("I2V_GUIDANCE", 3.5)
    arm_timeout(env_int("I2V_TIMEOUT", 2700))

    jobs = load_jobs(in_dir, os.environ.get("I2V_JOB", "job.json"))
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[i2v] gpu={gpu_name()} model={model} jobs={len(jobs)} "
          f"{width}x{height} frames={frames} steps={steps}", flush=True)

    from diffusers.utils import export_to_video
    from PIL import Image

    pipe, load_s = build_pipeline(model)
    print(f"[i2v] pipeline ready in {load_s}s", flush=True)

    failed = 0
    for job in jobs:
        name = job["name"]
        try:
            src = resolve_image(job["image"], in_dir, out_dir / "_src")
        except (urllib.error.URLError, OSError) as exc:
            print(f"[i2v] {name}: could not fetch source image: {exc}", flush=True)
            failed += 1
            continue
        if not src.exists():
            print(f"[i2v] {name}: source image missing ({src}), skipped", flush=True)
            failed += 1
            continue
        image = Image.open(src).convert("RGB").resize((width, height))
        t0 = time.time()
        try:
            frames_out = pipe(
                image=image,
                prompt=job["prompt"],
                negative_prompt=job.get("negative", NEG_DEFAULT),
                height=height,
                width=width,
                num_frames=frames,
                num_inference_steps=steps,
                guidance_scale=guidance,
            ).frames[0]
        except Exception as exc:  # noqa: BLE001 - one bad clip must not lose the batch
            print(f"[i2v] {name}: generation failed: {exc}", flush=True)
            failed += 1
            continue
        gen_s = round(time.time() - t0, 1)
        clip = out_dir / f"{name}.mp4"
        export_to_video(frames_out, str(clip), fps=16)
        meta = {
            "name": name,
            "gpu": gpu_name(),
            "model": model,
            "load_s": load_s,
            "gen_s": gen_s,
            "frames": frames,
            "steps": steps,
            "resolution": f"{width}x{height}",
            "guidance": guidance,
            "bytes": clip.stat().st_size,
        }
        if job.get("put_url"):
            meta["uploaded"] = put_file(clip, job["put_url"])
        (out_dir / f"{name}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"[i2v] {name}: {gen_s}s -> {clip.name} ({meta['bytes'] // 1024} KiB)", flush=True)
        # One line, one clip, machine-readable: on a market this log line is the
        # result. `nosana job get <id>` returns logs, not files.
        print(f"[i2v-result] {json.dumps(meta, sort_keys=True)}", flush=True)

    done = len(jobs) - failed
    print(f"[i2v] finished {done}/{len(jobs)} clips", flush=True)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
