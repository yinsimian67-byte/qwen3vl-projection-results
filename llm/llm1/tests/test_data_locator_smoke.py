"""Real CPU forward/backward + checkpoint + unlabeled export; requires locator deps."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_course_data import read_jsonl, write_jsonl


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("sklearn"), "install requirements-locator.txt for real model CPU smoke")
class LocatorIntegrationSmoke(unittest.TestCase):
    def test_train_checkpoint_and_export_three_splits_without_labels(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            features, data = temporary / "features", temporary / "course"
            for name in ("slowfast", "clip", "clip_text"):
                (features / name).mkdir(parents=True)
            rng = np.random.default_rng(42)
            for offset, split in enumerate(("train", "dev", "final")):
                rows = []
                for index in range(2):
                    qid = offset * 100 + index
                    vid = f"{split}_source_{index}_0.0_150.0"
                    row = {"qid": qid, "vid": vid, "query": "A person walks outside.", "duration": 150, "split": split,
                           "relevant_windows": [[10, 20]], "relevant_clip_ids": list(range(5, 10)), "saliency_scores": [[3, 4, 3]] * 5}
                    rows.append(row)
                    np.save(features / "slowfast" / f"{vid}.npy", rng.standard_normal((75, 2304)).astype(np.float32))
                    np.savez(features / "clip" / f"{vid}.npz", features=rng.standard_normal((75, 512)).astype(np.float32))
                    np.savez(features / "clip_text" / f"qid{qid}.npz", last_hidden_state=rng.standard_normal((9, 512)).astype(np.float32))
                write_jsonl(data / f"{split}.jsonl", rows)
                write_jsonl(data / f"{split}_queries.jsonl", [{key: row[key] for key in ("qid", "vid", "query", "duration", "split")} for row in rows])
            environment = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")

            def run(script, *arguments):
                result = subprocess.run([sys.executable, str(ROOT / "scripts" / script), *map(str, arguments)],
                                        cwd=ROOT, env=environment, capture_output=True, text=True, timeout=180)
                self.assertEqual(result.returncode, 0, result.stdout[-5000:] + result.stderr[-5000:])

            run("train_locator.py", "--feature-root", features, "--data-dir", data, "--output-dir", temporary / "run",
                "--epochs", 1, "--batch-size", 2, "--workers", 0, "--device", "cpu")
            self.assertTrue((temporary / "run" / "model_best.pt").is_file())
            for split in ("train", "dev", "final"):
                output = temporary / f"{split}_candidates.jsonl"
                run("export_candidates.py", "--checkpoint", temporary / "run" / "model_best.pt", "--feature-root", features,
                    "--annotations", data / f"{split}_queries.jsonl", "--split", split, "--output", output,
                    "--device", "cpu", "--workers", 0)
                predictions = read_jsonl(output)
                self.assertEqual(len(predictions), 2)
                self.assertEqual(len(predictions[0]["pred_relevant_windows"]), 3)
                self.assertNotIn("relevant_windows", predictions[0])
                run("evaluate.py", "--ground-truth", data / f"{split}.jsonl", "--predictions", output,
                    "--reference-candidates", output, "--output", temporary / f"{split}_metrics.json")
            metrics = json.loads((temporary / "final_metrics.json").read_text())
            self.assertTrue(metrics["protocol"]["same_candidate_pool_checked"])


if __name__ == "__main__":
    unittest.main()
