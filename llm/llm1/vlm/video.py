"""Decode exact QVHighlights clip files and cache uniformly sampled RGB frames."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile


class VideoStore:
    def __init__(self, root, cache_dir, num_frames=4, max_side=448, cache_only=False):
        self.root = Path(root).resolve()
        self.cache = Path(cache_dir)
        self.cache_only = cache_only
        self.num_frames = int(num_frames)
        self.max_side = int(max_side)
        if self.num_frames < 2 or self.num_frames % 2:
            raise ValueError("Qwen temporal patches require a positive even frame count >= 2")
        self.cache.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.cache / "index.jsonl"
        self.manifest = {}
        if self.manifest_path.exists():
            with self.manifest_path.open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        entry = json.loads(line)
                        self.manifest[entry["cache_key"]] = entry
        self.source_hashes = {}
        self.sampling_events = []
        self.index = {}
        for path in ([] if cache_only else self.root.rglob("*")):
            if path.suffix.lower() in {".mp4", ".webm", ".mkv", ".avi", ".mov"}:
                if path.stem in self.index:
                    raise ValueError(f"Duplicate video stem {path.stem}; supply one canonical clip per vid")
                self.index[path.stem] = path

    def get(self, vid, start, end):
        import numpy as np
        fingerprint = {
            "version": 3, "vid": vid, "start": float(start), "end": float(end),
            "num_frames": self.num_frames, "max_side": self.max_side,
        }
        key = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()
        destination = self.cache / (key + ".npz")
        metadata = self.manifest.get(key)
        path = self.index.get(vid)
        stat = path.stat() if path else None
        valid = destination.exists() and metadata is not None
        if valid and stat and not self.cache_only:
            if metadata["source_size"] != stat.st_size or metadata["source_mtime_ns"] != stat.st_mtime_ns:
                valid = self._source_hash(path, stat) == metadata["source_sha256"]
        # Cached frames and their portable manifest are sufficient on an AutoDL
        # host with no raw videos. Absolute source paths are never cache keys.
        if valid:
            with np.load(destination, allow_pickle=False) as cached:
                self._record_sampling(metadata)
                return cached["frames"], float(cached["fps"].item())
        if self.cache_only:
            raise FileNotFoundError(f"Cache-only miss for vid={vid}, [{start}, {end}], frames={self.num_frames}, max_side={self.max_side}")
        if path is None:
            raise FileNotFoundError(f"No portable cache or exact video stem {vid} under {self.root}")
        frames, times, fps, sampling = decode_frames(path, start, end, self.num_frames, self.max_side)
        descriptor, temporary = tempfile.mkstemp(dir=self.cache, suffix=".npz")
        os.close(descriptor)
        try:
            np.savez_compressed(temporary, frames=frames, actual_times=times, fps=fps)
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        entry = {**fingerprint, "cache_key": key, "cache_file": destination.name,
                 "source_path": str(path), "source_size": stat.st_size,
                 "source_mtime_ns": stat.st_mtime_ns, "source_sha256": self._source_hash(path, stat),
                 "sampling": sampling}
        with self.manifest_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self.manifest[key] = entry
        self._record_sampling(entry)
        return frames, fps

    def _record_sampling(self, metadata):
        sampling = metadata.get("sampling", {})
        if sampling.get("end_trim_seconds", 0) > 0 or sampling.get("seek_fallback"):
            self.sampling_events.append({"vid": metadata["vid"], "cache_key": metadata["cache_key"], **sampling})

    def drain_sampling_events(self):
        events, self.sampling_events = self.sampling_events, []
        # A pairwise comparison can view the same clip several times.
        unique = {event["cache_key"]: event for event in events}
        return list(unique.values())

    def _source_hash(self, path, stat):
        fingerprint = (str(path), stat.st_size, stat.st_mtime_ns)
        if fingerprint not in self.source_hashes:
            from .common import file_sha256
            self.source_hashes[fingerprint] = file_sha256(path)
        return self.source_hashes[fingerprint]


def decode_frames(path, start, end, count, max_side):
    import av
    import numpy as np
    from PIL import Image

    start, end = float(start), float(end)
    if not 0 <= start < end:
        raise ValueError(f"Invalid sampling interval {start}, {end}")
    # Midpoint sampling gives count equally spaced observations and an effective
    # sampling fps=count/effective_duration. Passing that fps is essential for
    # Qwen's MRoPE; any bounded annotation/media overrun is audited below.
    chosen, timestamps = [], []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        origin = float((stream.start_time or 0) * stream.time_base)
        media_duration = (float(stream.duration * stream.time_base) if stream.duration is not None
                          else float(container.duration / av.time_base) if container.duration is not None else None)
        effective_end = end
        tolerance = 2.1  # The public annotations have a two-second clip grid.
        if media_duration is not None and end > media_duration:
            if end - media_duration > tolerance:
                raise ValueError(f"Requested end exceeds media duration by more than {tolerance}s: end={end}, media={media_duration}")
            if start >= media_duration:
                raise ValueError(f"Candidate starts beyond the available media: start={start}, media={media_duration}")
            effective_end = media_duration
        targets = start + (np.arange(count) + 0.5) * (effective_end - start) / count
        sampling = {"requested_start": start, "requested_end": end,
                    "effective_start": start, "effective_end": effective_end,
                    "media_duration": media_duration, "end_trim_seconds": end - effective_end,
                    "annotation_tolerance_seconds": tolerance,
                    "effective_fps": count / (effective_end - start)}
        decoded_frames = 0

        def select_nearest(source, source_stream, sample_targets, require_bracket=False):
            nonlocal decoded_frames
            result, result_times = [], []
            previous, previous_time, last_time = None, None, None
            target_index = 0
            source_origin = float((source_stream.start_time or 0) * source_stream.time_base)
            for frame in source.decode(source_stream):
                decoded_frames += 1
                if frame.pts is None:
                    continue
                timestamp = float(frame.pts * source_stream.time_base) - source_origin
                last_time = timestamp
                while target_index < len(sample_targets) and timestamp >= sample_targets[target_index]:
                    # Some containers seek using DTS. If the returned keyframe's
                    # PTS is already after the target, the previous GOP may hold
                    # a closer frame. Fall back explicitly rather than guess.
                    if require_bracket and previous is None and timestamp > sample_targets[target_index] + 1e-7:
                        raise RuntimeError("Keyframe seek did not return a frame at or before the target")
                    selected, selected_time = frame, timestamp
                    if previous is not None and abs(previous_time - sample_targets[target_index]) < abs(timestamp - sample_targets[target_index]):
                        selected, selected_time = previous, previous_time
                    result.append(selected)
                    result_times.append(selected_time)
                    target_index += 1
                if target_index == len(sample_targets):
                    return result, result_times
                previous, previous_time = frame, timestamp
            # Same final-frame rounding rule as the original sequential decoder.
            frame_tolerance = max(0.15, 2 / float(source_stream.average_rate or 25))
            if previous is None or last_time is None or effective_end - last_time > frame_tolerance:
                raise ValueError(f"Clip ended before effective interval: end={effective_end}, decoded_until={last_time}")
            while target_index < len(sample_targets):
                result.append(previous)
                result_times.append(previous_time)
                target_index += 1
            return result, result_times

        selected_frames = []
        sampling["decode_strategy"] = "keyframe_seek"
        sampling["seek_fallback"] = None
        try:
            for target in targets:
                container.seek(int((origin + float(target)) / float(stream.time_base)),
                               stream=stream, backward=True, any_frame=False)
                frames, times = select_nearest(container, stream, [target], require_bracket=True)
                selected_frames.extend(frames)
                timestamps.extend(times)
        except Exception as error:
            # Reopen and decode from the beginning if seeking is unsupported or
            # cannot bracket a target. This keeps exact nearest-frame semantics;
            # the explicit audit event exposes slower fallback cases.
            sampling["decode_strategy"] = "sequential_fallback"
            sampling["seek_fallback"] = f"{type(error).__name__}: {error}"
            with av.open(str(path)) as fallback:
                selected_frames, timestamps = select_nearest(fallback, fallback.streams.video[0], targets)
        sampling["decoded_frames"] = decoded_frames
        for selected in selected_frames:
            picture = selected.to_image().convert("RGB")
            picture.thumbnail((max_side, max_side), Image.Resampling.BICUBIC)
            chosen.append(np.asarray(picture))
    if len({picture.shape for picture in chosen}) != 1:
        raise ValueError("Video changes frame dimensions inside the candidate")
    return np.stack(chosen), np.asarray(timestamps), count / (effective_end - start), sampling
