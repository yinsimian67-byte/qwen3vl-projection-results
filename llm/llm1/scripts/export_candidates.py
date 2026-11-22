#!/usr/bin/env python3
"""Export exactly three label-free Moment-DETR proposals for train, dev, or final."""
import argparse
import hashlib
from pathlib import Path
from types import SimpleNamespace

from locator_common import audit_features, discover_features, dump_json, make_dataset, predict
from prepare_course_data import read_jsonl, source_id, write_jsonl
from evaluate import validate_predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--annotations", required=True, help="Labeled JSONL or *_queries.jsonl; labels never loaded")
    parser.add_argument("--split", required=True, choices=["train", "dev", "final"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()
    import torch
    from torch.utils.data import DataLoader
    from moment_detr.model import build_model
    from moment_detr.start_end_dataset import start_end_collate
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    rows = read_jsonl(args.annotations)
    if any(row.get("split", args.split) != args.split for row in rows):
        raise ValueError("--split disagrees with annotation split labels")
    layout = discover_features(args.feature_root)
    audit_features(layout, rows)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("schema_version") != 1:
        raise ValueError("Use a course train_locator.py checkpoint, not the full-training official pretrained model")
    train_qids = set(checkpoint["train_qids"])
    current_qids = {row["qid"] for row in rows}
    if args.split != "train" and current_qids & train_qids:
        raise ValueError("Evaluation data overlap locator training qids")
    if args.split != "train" and {source_id(row["vid"]) for row in rows} & set(checkpoint["train_source_ids"]):
        raise ValueError("Evaluation data overlap locator training source videos")
    if args.split == "train" and not current_qids <= train_qids:
        raise ValueError("Train candidate export must be a subset of the locator's recorded train qids")
    config = dict(checkpoint["model_config"], device=args.device)
    model, _ = build_model(SimpleNamespace(**config))
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    dataset = make_dataset(args.annotations, layout, config, load_labels=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers,
                        shuffle=False, collate_fn=start_end_collate, pin_memory=device.type == "cuda")
    predictions = predict(model, loader, device, args.split)
    validate_predictions(predictions, rows)
    write_jsonl(args.output, predictions)
    dump_json(str(args.output) + ".meta.json", {"split": args.split, "queries": len(predictions), "top_k": 3,
              "uses_labels": False, "epoch": checkpoint["epoch"], "postprocessing": "2-second rounding; clamp to each clip duration; minimum 2s",
              "checkpoint_sha256": hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
              "annotation_sha256": hashlib.sha256(Path(args.annotations).read_bytes()).hexdigest()})
    print(f"Wrote {len(predictions)} {args.split} queries to {args.output}")


if __name__ == "__main__":
    main()
