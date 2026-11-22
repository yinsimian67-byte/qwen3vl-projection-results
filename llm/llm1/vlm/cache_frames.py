"""CPU preprocessing: validate video coverage and populate the frame cache."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import join_inputs, load_jsonl
from .modeling import read_config
from .video import VideoStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--sft-data", help="Prepared SFT JSONL")
    inputs.add_argument("--annotations", help="Evaluation/train query JSONL, used with --candidates")
    parser.add_argument("--candidates")
    parser.add_argument("--config", default="configs/vlm_qlora_pointwise.json")
    parser.add_argument("--video-root", default="data/videos")
    parser.add_argument("--cache-only", action="store_true", help="Audit a transported cache without raw videos or decoding")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--strict", action="store_true", help="Return a failing exit code when any clip cannot be decoded")
    args = parser.parse_args()
    if args.annotations and not args.candidates:
        parser.error("--annotations requires --candidates")
    config = read_config(args.config)
    store = VideoStore(args.video_root, args.cache_dir, config["num_frames"], config["cache_max_side"], args.cache_only)
    requests = {}
    if args.sft_data:
        for row in load_jsonl(args.sft_data):
            for start, end in row["clips"]:
                requests.setdefault((row["vid"], float(start), float(end)), set()).add(str(row["qid"]))
    else:
        for annotation, _, windows in join_inputs(load_jsonl(args.annotations), load_jsonl(args.candidates)):
            for start, end, _ in windows:
                requests.setdefault((annotation["vid"], start, end), set()).add(str(annotation["qid"]))
    failures, processed = [], 0
    for (vid, start, end), qids in requests.items():
        try:
            store.get(vid, start, end)
            processed += 1
        except Exception as error:
            failures.append({"vid": vid, "start": start, "end": end, "qids": sorted(qids),
                             "error_type": type(error).__name__, "error": str(error)})
        if (processed + len(failures)) % 100 == 0:
            print(f"Cached {processed}; failed {len(failures)}; requested {len(requests)}", flush=True)
    report = {"unique_intervals": len(requests), "cached_intervals": processed,
              "failed_intervals": len(failures), "failures": failures,
              "complete": not failures, "config": config,
              "sampling_adjustments": store.drain_sampling_events()}
    path = Path(args.report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key not in {"failures", "config"}}, indent=2))
    if failures and args.strict:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
