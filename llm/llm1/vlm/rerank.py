"""Rerank a fixed candidate pool. Keep every qid; audit all decode/model failures."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from .common import (file_sha256, join_inputs, load_jsonl, pairwise_order,
                     pointwise_order, ranked_prediction)
from .modeling import label_probability, load_model, lora_parameter_digest, prepare_inputs, read_config
from .video import VideoStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True, help="Query-only JSONL is sufficient; GT is never read")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--config", default="configs/vlm_qlora_pointwise.json")
    parser.add_argument("--video-root", default="data/videos")
    parser.add_argument("--cache-only", action="store_true", help="Use portable frames without opening original video files")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--adapter", help="Omit for zero-shot inference")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-queries", type=int, help="Smoke/debug only; unprocessed qids retain baseline order and are logged")
    args = parser.parse_args()
    if args.top_k != 3 or (args.max_queries is not None and args.max_queries < 0):
        parser.error("The course protocol requires top-k=3; max-queries must be nonnegative")
    config = read_config(args.config)
    joined = join_inputs(load_jsonl(args.annotations), load_jsonl(args.candidates))
    model, processor, _ = load_model(config, adapter=args.adapter)
    store = VideoStore(args.video_root, args.cache_dir, config["num_frames"], config["cache_max_side"], args.cache_only)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    audit_path = Path(str(output) + ".audit.jsonl")
    temporary_audit = Path(str(audit_path) + ".partial")
    temporary = Path(str(output) + ".partial")
    import torch
    torch.cuda.reset_peak_memory_stats()
    start_time = time.monotonic()
    failed_queries = 0
    skipped_queries = 0
    calls = 0
    with temporary.open("w", encoding="utf-8") as predictions, temporary_audit.open("w", encoding="utf-8") as audit:
        for query_index, (annotation, _, windows) in enumerate(joined):
            record = {"qid": annotation["qid"], "vid": annotation["vid"], "task": config["task"], "comparisons": []}
            count = min(args.top_k, len(windows))
            fallback = False
            if args.max_queries is not None and query_index >= args.max_queries:
                order = list(range(len(windows)))
                record["status"] = "debug_unprocessed_baseline_order"
                skipped_queries += 1
            else:
                try:
                    if config["task"] == "pointwise":
                        scores = []
                        for candidate_index in range(count):
                            batch = prepare_inputs(
                                processor, annotation["query"], [windows[candidate_index][:2]],
                                "pointwise", store, annotation["vid"], max_tokens=config["max_tokens"],
                            )
                            score = label_probability(model, processor, batch, "pointwise")
                            calls += 1
                            scores.append(score)
                        record["yes_probabilities"] = scores
                        order, fallback = pointwise_order(scores, len(windows))
                    else:
                        def compare(left, right):
                            nonlocal calls
                            # Score both orders to reduce the A/B position bias.
                            probabilities = []
                            for first, second in [(left, right), (right, left)]:
                                batch = prepare_inputs(
                                    processor, annotation["query"], [windows[first][:2], windows[second][:2]],
                                    "pairwise", store, annotation["vid"], max_tokens=config["max_tokens"],
                                )
                                probabilities.append(label_probability(model, processor, batch, "pairwise"))
                                calls += 1
                            preference = (probabilities[0] + 1.0 - probabilities[1]) / 2.0
                            record["comparisons"].append({"left": left, "right": right, "p_left": preference,
                                                          "forward_p_A": probabilities[0], "reverse_p_A": probabilities[1]})
                            return preference
                        order = pairwise_order(len(windows), count, compare)
                    record["status"] = "ok"
                except Exception as error:
                    # Preserve the entire original order if even one candidate
                    # fails. Partial reranking could otherwise reward failure.
                    order = list(range(len(windows)))
                    fallback = True
                    record["status"] = "failed_baseline_order"
                    record["error_type"] = type(error).__name__
                    record["error"] = str(error)
                    torch.cuda.empty_cache()
                if fallback:
                    failed_queries += 1
            record["order"] = order
            record["sampling_adjustments"] = store.drain_sampling_events()
            prediction = ranked_prediction(annotation, windows, order)
            predictions.write(json.dumps(prediction, ensure_ascii=False, allow_nan=False) + "\n")
            audit.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            predictions.flush()
            audit.flush()
            if (query_index + 1) % 25 == 0:
                print(f"Processed {query_index + 1}/{len(joined)} queries; fallback={failed_queries}", flush=True)
    temporary.replace(output)
    temporary_audit.replace(audit_path)
    metrics = {
        "device": "cuda", "gpu_name": torch.cuda.get_device_name(0), "torch_version": torch.__version__,
        "queries": len(joined), "failed_queries": failed_queries, "debug_unprocessed_queries": skipped_queries,
        "failure_rate": failed_queries / len(joined) if joined else 0,
        "model_forward_calls": calls, "elapsed_seconds": time.monotonic() - start_time,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "candidate_sha256": file_sha256(args.candidates), "annotation_sha256": file_sha256(args.annotations),
        "prediction_sha256": file_sha256(output),
        "adapter": args.adapter, "config": config, "top_k": args.top_k,
        "loaded_adapter_parameters": lora_parameter_digest(model) if args.adapter else None,
        "resolved_model_revision": getattr(model.config, "_commit_hash", None),
        "complete_vlm_evaluation": failed_queries == 0 and skipped_queries == 0,
    }
    Path(str(output) + ".metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
