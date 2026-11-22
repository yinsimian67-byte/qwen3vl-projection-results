#!/usr/bin/env python3
"""Train official Moment-DETR on fixed pretrained features; select by course dev mAP@3."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time
from types import SimpleNamespace

from locator_common import audit_features, discover_features, dump_json, make_dataset, model_config, predict
from prepare_course_data import assert_disjoint, read_jsonl, source_id, write_jsonl
from evaluate import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--data-dir", default="data/course")
    parser.add_argument("--output-dir", default="outputs/locator")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--check-only", action="store_true", help="Check layout, dimensions, coverage and split isolation without training")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.eval_every) < 1:
        parser.error("epochs/batch-size/eval-every must be positive")
    out, data = Path(args.output_dir), Path(args.data_dir)
    if not args.check_only and (out / "model_latest.pt").exists():
        raise FileExistsError(f"Existing run at {out}; choose a fresh --output-dir")
    rows = {split: read_jsonl(data / f"{split}.jsonl") for split in ("train", "dev")}
    assert_disjoint(rows)
    for split, records in rows.items():
        if any(record.get("split", split) != split for record in records):
            raise ValueError(f"Wrong split label in {split}.jsonl")
    layout = discover_features(args.feature_root)
    audit = audit_features(layout, rows["train"] + rows["dev"])
    out.mkdir(parents=True, exist_ok=True)
    dump_json(out / "feature_audit.json", audit)
    if args.check_only:
        print(json.dumps(audit, indent=2)); return

    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from moment_detr.model import build_model
    from moment_detr.start_end_dataset import start_end_collate, prepare_batch_inputs
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Use an NVIDIA training machine; --device cpu is for a small smoke test only.")
    device = torch.device(args.device)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.cuda.reset_peak_memory_stats()
    config = model_config(args.device)
    config["max_windows"] = max(5, max(len(row["relevant_windows"]) for row in rows["train"]))
    dump_json(out / "config.json", {"course_args": vars(args), "model": config})
    datasets = {split: make_dataset(data / f"{split}.jsonl", layout, config, load_labels=split == "train") for split in rows}
    loaders = {split: DataLoader(dataset, batch_size=args.batch_size, shuffle=split == "train",
                                num_workers=args.workers, collate_fn=start_end_collate,
                                pin_memory=device.type == "cuda") for split, dataset in datasets.items()}
    model, criterion = build_model(SimpleNamespace(**config))
    model.to(device); criterion.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, int(args.epochs * 0.75)), gamma=0.1)
    best_score, start_time = -1.0, time.time()
    peak_allocated, peak_reserved = 0, 0
    training_qids = [row["qid"] for row in rows["train"]]
    print(f"Training {sum(p.numel() for p in model.parameters()):,} parameters, {len(training_qids)} queries, {args.epochs} epochs", flush=True)
    for epoch in range(1, args.epochs + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        model.train(); criterion.train()
        losses, epoch_started = [], time.time()
        for _, batch in loaders["train"]:
            inputs, targets = prepare_batch_inputs(batch, device, non_blocking=device.type == "cuda")
            loss_dict = criterion(model(**inputs), targets)
            loss = sum(value * criterion.weight_dict[key] for key, value in loss_dict.items() if key in criterion.weight_dict)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss in epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        train_seconds = time.time()-epoch_started
        log = {"epoch": epoch, "loss": sum(losses)/len(losses), "seconds": train_seconds,
               "train_seconds": train_seconds, "dev_seconds": 0.0}
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            dev_started = time.time()
            predictions = predict(model, loaders["dev"], device, "dev")
            metrics = evaluate(predictions, rows["dev"])
            log["dev_seconds"] = time.time() - dev_started
            log["dev"] = metrics["brief"]
            checkpoint = {"schema_version": 1, "model": model.state_dict(), "model_config": config,
                          "epoch": epoch, "seed": args.seed, "train_qids": training_qids,
                          "train_source_ids": sorted({source_id(row["vid"]) for row in rows["train"]}),
                          "train_source_sha256": hashlib.sha256((data / "train.jsonl").read_bytes()).hexdigest(),
                          "selection_metric": "dev MR-full-mAP with exactly 3 predictions/query"}
            torch.save(checkpoint, out / "model_latest.pt")
            if metrics["brief"]["MR-full-mAP"] > best_score:
                best_score = metrics["brief"]["MR-full-mAP"]
                torch.save(checkpoint, out / "model_best.pt")
                write_jsonl(out / "dev_best_candidates.jsonl", predictions)
                dump_json(out / "dev_best_metrics.json", metrics)
        log["epoch_wall_seconds"] = time.time() - epoch_started
        if device.type == "cuda":
            allocated, reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
            peak_allocated, peak_reserved = max(peak_allocated, allocated), max(peak_reserved, reserved)
            log["gpu_peak_allocated_GiB"], log["gpu_peak_reserved_GiB"] = allocated/2**30, reserved/2**30
        with (out / "train_log.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(log) + "\n")
        print(json.dumps(log), flush=True)
    dump_json(out / "run_summary.json", {"elapsed_seconds": time.time()-start_time, "best_dev_mAP": best_score,
              "gpu_peak_allocated_GiB": peak_allocated/2**30 if device.type == "cuda" else None,
              "gpu_peak_reserved_GiB": peak_reserved/2**30 if device.type == "cuda" else None,
              "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
              "torch_version": torch.__version__, "device": str(device), "epochs": args.epochs})


if __name__ == "__main__":
    main()
