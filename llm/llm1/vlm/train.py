"""Single-CUDA-GPU, batch-one LoRA/QLoRA SFT with gradient accumulation."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import time

from .common import file_sha256, load_jsonl
from .modeling import load_model, prepare_inputs, read_config, to_cuda
from .video import VideoStore


def snapshot_trainable(model):
    """Copy only trainable adapters to CPU RAM, never the frozen 3B backbone."""
    return {name: parameter.detach().float().cpu().clone()
            for name, parameter in model.named_parameters() if parameter.requires_grad}


def update_evidence(model, initial, optimizer, run_start_step, last_step):
    """Measure actual parameter changes, independently of the reported loss."""
    import torch

    current_names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    if current_names != set(initial):
        raise RuntimeError("Trainable parameter set changed during the run")
    initial_digest, final_digest = hashlib.sha256(), hashlib.sha256()
    squared_delta, max_delta, changed_elements, total_elements, changed_tensors = 0.0, 0.0, 0, 0, 0
    samples = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        before = initial[name]
        after = parameter.detach().float().cpu()
        if not torch.isfinite(after).all():
            raise FloatingPointError(f"Nonfinite adapter parameters: {name}")
        delta = after.double() - before.double()
        changed = int(torch.count_nonzero(delta))
        changed_elements += changed
        total_elements += delta.numel()
        changed_tensors += int(changed > 0)
        squared_delta += float(torch.sum(delta.square()))
        max_delta = max(max_delta, float(delta.abs().max()))
        for digest, value in [(initial_digest, before), (final_digest, after)]:
            digest.update(name.encode("utf-8"))
            digest.update(str(tuple(value.shape)).encode("ascii"))
            digest.update(value.contiguous().numpy().tobytes())
        if len(samples) < 4:
            samples.append({"name": name, "shape": list(after.shape),
                            "before_first8": before.flatten()[:8].tolist(),
                            "after_first8": after.flatten()[:8].tolist(),
                            "delta_l2": float(torch.linalg.vector_norm(delta)),
                            "changed_elements": changed})
    state_steps = [float(state["step"]) for state in optimizer.state.values() if "step" in state]
    return {
        "run_start_step": run_start_step, "last_step": last_step,
        "optimizer_state_step_min": min(state_steps) if state_steps else None,
        "optimizer_state_step_max": max(state_steps) if state_steps else None,
        "trainable_tensor_count": len(initial), "trainable_elements": total_elements,
        "changed_tensor_count": changed_tensors, "changed_elements": changed_elements,
        "total_delta_l2": math.sqrt(squared_delta), "max_abs_delta": max_delta,
        "initial_trainable_sha256": initial_digest.hexdigest(),
        "final_trainable_sha256": final_digest.hexdigest(), "parameter_samples": samples,
        "evidence_of_parameter_update": changed_elements > 0 and last_step > run_start_step
                                        and bool(state_steps) and max(state_steps) > 0,
        "scope": "All trainable LoRA parameters; deltas are relative to this invocation, after resume loading if used.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--video-root", default="data/videos")
    parser.add_argument("--cache-only", action="store_true", help="Use portable frames without opening original video files")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", help="A checkpoint directory produced by this script")
    parser.add_argument("--max-steps", type=int, help="Override config; use 1 for a CUDA smoke check")
    args = parser.parse_args()

    import torch
    from peft import set_peft_model_state_dict
    from peft.utils.save_and_load import load_peft_weights

    config = read_config(args.config)
    if args.max_steps is not None:
        config["max_steps"] = args.max_steps
    if config["max_steps"] < 1 or config["gradient_accumulation_steps"] < 1:
        raise ValueError("max_steps and gradient_accumulation_steps must be positive")
    rows = load_jsonl(args.train_data)
    if not rows:
        raise ValueError("No training examples")
    for row in rows:
        if row.get("split") != "train" or row["task"] != config["task"]:
            raise ValueError("Every example must match config.task and be marked split=train")
        allowed = {"Yes", "No"} if row["task"] == "pointwise" else {"A", "B"}
        if row["answer"] not in allowed:
            raise ValueError("Unexpected training answer")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "adapter_config.json").exists() and not args.resume:
        raise FileExistsError("Output already contains an adapter; use a new directory or --resume")
    seed = config["seed"]
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model, processor, dtype = load_model(config, training=True)
    store = VideoStore(args.video_root, args.cache_dir, config["num_frames"], config["cache_max_side"], args.cache_only)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config["learning_rate"], weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    step, offset, epoch = 0, 0, 0
    order = list(range(len(rows)))
    random.shuffle(order)
    train_sha = file_sha256(args.train_data)
    if args.resume:
        checkpoint = Path(args.resume)
        previous_config = json.loads((checkpoint / "training_config.json").read_text())
        for key in ("model_name", "task", "load_in_4bit", "lora_rank", "lora_targets", "num_frames", "max_pixels"):
            if config[key] != previous_config[key]:
                raise ValueError(f"Resume config mismatch: {key}")
        set_peft_model_state_dict(model, load_peft_weights(str(checkpoint), device="cuda"))
        state = torch.load(checkpoint / "trainer_state.pt", map_location="cuda", weights_only=True)
        if state["train_data_sha256"] != train_sha:
            raise ValueError("Training data changed since checkpoint")
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        step, offset, epoch, order = state["step"], state["offset"], state["epoch"], state["order"]
        random.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"].cpu())
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda_rng"]])

    run_start_step = step
    initial_trainable = snapshot_trainable(model)

    def save_checkpoint(destination):
        destination.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(destination, safe_serialization=True)
        processor.save_pretrained(destination)
        (destination / "training_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        torch.save({
            "step": step, "offset": offset, "epoch": epoch, "order": order,
            "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
            "python_rng": random.getstate(), "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(), "train_data_sha256": train_sha,
        }, destination / "trainer_state.pt")
        evidence = update_evidence(model, initial_trainable, optimizer, run_start_step, step)
        (destination / "parameter_update_evidence.json").write_text(
            json.dumps(evidence, indent=2, allow_nan=False), encoding="utf-8",
        )

    metadata = {
        "train_data_sha256": train_sha, "training_examples": len(rows),
        "trainable_parameters": sum(parameter.numel() for parameter in parameters),
        "gpu": torch.cuda.get_device_name(0), "torch_version": torch.__version__,
        "resolved_model_revision": getattr(model.config, "_commit_hash", None),
        "run_start_step": run_start_step,
        "initial_trainable_snapshot_cpu_bytes": sum(value.numel() * value.element_size() for value in initial_trainable.values()),
        "config": config,
    }
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    torch.cuda.reset_peak_memory_stats()
    start_time = time.monotonic()
    accum = config["gradient_accumulation_steps"]
    log_path = output / "training_log.jsonl"
    optimizer.zero_grad(set_to_none=True)
    while step < config["max_steps"]:
        losses = []
        sampling_adjustments = []
        for _ in range(accum):
            if offset == len(order):
                offset = 0
                epoch += 1
                random.shuffle(order)
            row = rows[order[offset]]
            # Fail visibly on unreadable training clips; never silently drop a
            # class or subset while reporting a successful training run.
            batch = prepare_inputs(
                processor, row["query"], row["clips"], config["task"], store,
                row["vid"], answer=row["answer"], max_tokens=config["max_tokens"],
            )
            sampling_adjustments.extend(store.drain_sampling_events())
            with torch.autocast(device_type="cuda", dtype=dtype):
                loss = model(**to_cuda(batch), use_cache=False).loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at step {step}, qid {row['qid']}")
            losses.append(float(loss.detach()))
            scaler.scale(loss / accum).backward()
            offset += 1
        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        warmup = max(1, config["warmup_steps"])
        multiplier = min(1.0, (step + 1) / warmup)
        for group in optimizer.param_groups:
            group["lr"] = config["learning_rate"] * multiplier
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        record = {
            "step": step, "epoch": epoch, "loss": sum(losses) / len(losses),
            "lr": optimizer.param_groups[0]["lr"],
            "gradient_norm_before_clip": float(gradient_norm.detach()),
            "sampling_adjustments": sampling_adjustments,
            "elapsed_seconds": time.monotonic() - start_time,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        if step % config["save_every"] == 0:
            save_checkpoint(output / f"checkpoint-{step}")
    save_checkpoint(output)
    print(f"Adapter update evidence: {output / 'parameter_update_evidence.json'}")
    print(f"Saved final adapter and resumable trainer state to {output}")


if __name__ == "__main__":
    main()
