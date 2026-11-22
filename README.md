# nosana-i2v — image-to-video b-roll on rented GPUs

A single-purpose container that turns one still image plus one prompt into one
short video clip, and the scripts to fan that container out across a
decentralised GPU market instead of a single machine.

It exists because of a very ordinary bottleneck: the pipeline that needs these
clips runs on a laptop with an **RTX 2050, 4 GB**. Wan2.2 image-to-video needs
roughly 24 GB with VAE tiling and offloading, and 80 GB for the 14B variant. So
the model is not slow locally — it does not run locally at all. The choice is a
hyperscaler at $2.50–$3.95 per GPU-hour, or a market where the same class of
card is listed at a fraction of that.

## What it does

```
local   in/job.json + in/*.png ──▶ [ container, 1 GPU ] ──▶ out/<name>.mp4
                                                            out/<name>.json
market  I2V_JOB_JSON + image URLs ─▶ [ container, 1 GPU ] ─▶ [i2v-result] log lines
                                                            (+ optional presigned PUT)
```

A market node has no directory to mount and no artifact store, so the worker
runs both ways. Locally it reads a mounted directory. On a market the job JSON
arrives inline in `I2V_JOB_JSON`, stills are fetched from URLs, and every clip's
timing record is printed as one `[i2v-result] {...}` line — logs are what
`nosana job get` gives back. The MP4 itself leaves via an optional presigned
`put_url` per clip: scoped to one object, expiring, so no standing credential is
ever handed to a rented machine.

- `worker/i2v.py` — the whole worker. Env vars in, clips and timing records out.
  No cloud SDK and nothing stored: the only secrets it can see are presigned
  URLs the job itself passes in.
- `Dockerfile` — python 3.12, torch 2.9.1 (cu126), diffusers 0.42.1, all pinned.
  `diffusers >= 0.42` is a floor: Wan2.2 needs the MoE `boundary_ratio` path and
  the 5B TI2V `expand_timesteps` path, and older releases have neither.
- `nosana/job-i2v.json` — Nosana job definition. Stages the model with an `HF`
  resource so the node, not the container, pulls the weights; if that path
  misbehaves, point `I2V_MODEL` back at the repo id and the worker downloads
  them itself.
- `scripts/run_local.sh` — the control run on a local GPU.
- `scripts/submit_nosana.sh` — post N independent jobs to a market.
- `bench/bench.py` — timing files ➜ clips/GPU-hour and $/clip.

Default model is **Wan2.2-TI2V-5B** (Apache-2.0) because it fits the cheap end
of the market. Point `I2V_MODEL` at `Wan-AI/Wan2.2-I2V-A14B-Diffusers` on an
80 GB node for the higher-fidelity path.

## Quick start

```bash
# local control run (needs docker + nvidia-container-toolkit)
scripts/run_local.sh examples/ out/
bench/bench.py out/ --rate <your-hourly-rate>

# on a market
docker build -t <registry>/nosana-i2v:latest . && docker push <registry>/nosana-i2v:latest
# job-i2v.json ships pointing at ghcr.io/bisstown3000/nosana-i2v:latest and examples/stills/hook_01.png — change both for your own run
$EDITOR nosana/job-i2v.json
scripts/submit_nosana.sh nvidia-4090 8
nosana job get <job-id> > out/node-a.log
bench/bench.py out/node-a.log --rate 0.40
```

The market slug (`nvidia-4090`, `nvidia-3090`, …) is what picks the GPU class —
there is no GPU-model field in the job definition, only `"gpu": true`.

## Why one clip per job

Each clip is independent: no inter-node communication, no shared state, no
gradient sync. N nodes therefore give N× throughput minus a one-time model
download per node, and a node that dies costs one clip instead of the batch.
`bench/bench.py` reports that download separately (`load_s`) rather than
amortising it into the per-clip average, so the scaling claim can be checked
rather than taken on trust.

## Cost accounting

`bench.py` never hardcodes a price. It takes `--rate` and divides, so the
numbers it prints are whatever the node actually charged. It accepts either a
local output directory or a saved job log per node, and de-duplicates records
that appear in both. Measure, then compare.

## Guard rails

`I2V_TIMEOUT` (default 2700 s) arms a `SIGALRM` that kills the job. A rented GPU
bills by the second, and an unattended generation loop that hangs will happily
run until the credits are gone. One bad clip also never aborts the batch — the
failure is logged, the remaining jobs continue, and the exit code reports it.

## Status

The generation path is the one already in production use for a YouTube pipeline
(Wan2.2 via diffusers, VAE tiling, 1280×704, 81 frames at 16 fps). What is new
here is the packaging: the same code as a portable container plus a measurement
harness. The job definition follows the documented `0.1` container schema
(`global.env`, `ops[].args.resources`), but re-check it and the CLI flags
against the current Nosana docs before a paid run — an invalid spec still costs
the posting fee.

## Licence

Apache-2.0 (see `LICENSE`). The Wan2.2 model weights are Apache-2.0 upstream;
check the model card before redistributing them.
