#!/usr/bin/env python3
"""Turn a directory of per-clip timing files into throughput and cost numbers.

Reads every *.json written by worker/i2v.py and reports, for one node:
  clips, median/mean seconds per clip, clips per GPU-hour,
  and - if an hourly rate is given - the cost per clip.

Scaling across N nodes is reported as the sum of per-node throughput, which is
honest for this workload: each clip is an independent container job, so there is
no inter-node communication to erode the speedup. The only shared cost is the
one-time model download per node, which is why `load_s` is reported separately
instead of being folded into the per-clip average.

A market node returns logs rather than files, so a saved job log is accepted in
place of a directory: the worker prints one `[i2v-result] {...}` line per clip,
and those lines carry the same record the json files do.

Usage
  python3 bench/bench.py out/                      # one node (local run)
  python3 bench/bench.py out/ --rate 1.50          # $/GPU-hour -> cost per clip
  python3 bench/bench.py node-a.log node-b.log --rate 0.40   # aggregate N nodes
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


RESULT_TAG = "[i2v-result] "


def from_log(p: Path) -> list[dict]:
    """Pull timing records out of a job log (`nosana job get <id> > node.log`)."""
    out = []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        i = line.find(RESULT_TAG)
        if i < 0:
            continue
        try:
            meta = json.loads(line[i + len(RESULT_TAG):])
        except json.JSONDecodeError:
            continue
        if "gen_s" in meta:
            out.append(meta)
    return out


def load_node(src: Path) -> list[dict]:
    if src.is_file():
        return from_log(src)
    out = []
    for p in sorted(src.glob("*.json")):
        try:
            meta = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"  ! {p.name}: not valid json, skipped")
            continue
        if "gen_s" in meta:
            out.append(meta)
    for p in sorted(src.glob("*.log")):
        out.extend(from_log(p))
    # Same clip may appear as both a json file and a log line on a local run.
    seen, uniq = set(), []
    for meta in out:
        key = (meta.get("name"), meta.get("gen_s"))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(meta)
    return uniq


def summarise(name: str, clips: list[dict], rate: float | None) -> dict:
    gen = [c["gen_s"] for c in clips]
    load = max((c.get("load_s", 0) for c in clips), default=0)
    per_hour = 3600 / statistics.mean(gen) if gen else 0
    row = {
        "node": name,
        "gpu": clips[0].get("gpu", "?") if clips else "?",
        "clips": len(clips),
        "median_s": round(statistics.median(gen), 1) if gen else 0,
        "mean_s": round(statistics.mean(gen), 1) if gen else 0,
        "load_s": load,
        "clips_per_gpu_hour": round(per_hour, 1),
    }
    if rate and per_hour:
        row["usd_per_clip"] = round(rate / per_hour, 4)
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", type=Path,
                    help="one output directory or job log per node")
    ap.add_argument("--rate", type=float, default=None, help="GPU price in USD per hour")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args()

    rows, total_clips, total_gen = [], 0, 0.0
    for d in args.dirs:
        if not d.exists():
            print(f"! {d} not found, skipped")
            continue
        clips = load_node(d)
        if not clips:
            print(f"! {d} has no timing records, skipped")
            continue
        rows.append(summarise(d.name, clips, args.rate))
        total_clips += len(clips)
        total_gen += sum(c["gen_s"] for c in clips)

    if not rows:
        print("no results found")
        return 1

    fleet = {
        "nodes": len(rows),
        "clips": total_clips,
        "mean_s": round(total_gen / total_clips, 1),
        "clips_per_hour": round(sum(r["clips_per_gpu_hour"] for r in rows), 1),
    }
    if args.rate:
        fleet["usd_per_clip"] = round(args.rate / (3600 / fleet["mean_s"]), 4)
        fleet["usd_per_hour"] = round(args.rate * len(rows), 2)

    if args.json:
        print(json.dumps({"nodes": rows, "fleet": fleet}, indent=2))
        return 0

    head = f"{'node':<16}{'gpu':<22}{'clips':>6}{'median s':>10}{'mean s':>9}{'clips/h':>9}"
    if args.rate:
        head += f"{'$/clip':>9}"
    print(head)
    print("-" * len(head))
    for r in rows:
        line = (f"{r['node']:<16}{r['gpu'][:21]:<22}{r['clips']:>6}"
                f"{r['median_s']:>10}{r['mean_s']:>9}{r['clips_per_gpu_hour']:>9}")
        if args.rate:
            line += f"{r.get('usd_per_clip', 0):>9}"
        print(line)
    print("-" * len(head))
    print(f"fleet: {fleet['nodes']} node(s), {fleet['clips']} clips, "
          f"{fleet['clips_per_hour']} clips/hour"
          + (f", ${fleet['usd_per_clip']}/clip at ${args.rate}/GPU-h" if args.rate else ""))
    print("note: first clip on each node includes a one-time model download "
          f"(max load_s seen: {max(r['load_s'] for r in rows)}s), excluded from gen_s.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
