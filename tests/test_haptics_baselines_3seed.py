"""Tests for the Haptics 3-seed baseline study runner (fast, no training)."""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from experiments.haptics_baselines_3seed import runner as R  # noqa: E402


def test_reuse_seed42_reads_stored_results(tmp_path):
    d, _ = R.load_haptics()
    res, preds = R.reuse_seed42("InceptionTime", d, str(tmp_path / "s42"))
    assert len(preds) == 308
    assert res["macro_f1"] == 0.0636          # frozen canonical value
    assert (tmp_path / "s42" / "results.json").exists()
    assert (tmp_path / "s42" / "predictions.csv").exists()
    # predictions agree with the stored reference
    ref = np.load(os.path.join(R.EXT_STACK, "predictions",
                               "Haptics_InceptionTime.npy"))
    assert (preds == ref).all()


def test_fresh_run_writes_all_artifacts(tmp_path):
    import torch
    if not torch.cuda.is_available():
        pass  # CPU is fine for FCN on 132 samples
    d, dd = R.load_haptics()
    res, preds = R.run_fresh("FCN", d, dd, 43, torch.device("cpu"),
                             str(tmp_path / "s43"))
    assert len(preds) == 308
    run_dir = tmp_path / "s43"
    for f in ("results.json", "metrics.json", "training_history.json",
              "predictions.npy", "predictions.csv", "config.json"):
        assert (run_dir / f).exists(), f
    hist = json.load(open(run_dir / "training_history.json"))
    assert hist["best_epoch"] == res["best_epoch"]
    cfg = json.load(open(run_dir / "config.json"))
    assert cfg["test_evaluations"] == 1 and cfg["seed"] == 43


def test_summary_statistics():
    vals = [0.1, 0.2, 0.3]
    s = R.stats(vals)
    assert abs(s["mean"] - 0.2) < 1e-9
    assert abs(s["std"] - 0.1) < 1e-9      # ddof=1
    assert s["median"] == 0.2 and s["min"] == 0.1 and s["max"] == 0.3
    single = R.stats([0.42])
    assert single["std"] == 0.0            # no manufactured SD


def test_final_comparison_rows_shape():
    # the frozen references must be present and correctly labelled
    assert R.M0_SEED42 == 0.4974
    assert set(R.R2_SEEDS) == {42, 43, 44}
    assert set(R.R5_SEEDS) == {42, 43, 44}
    assert all(abs(R.R2_SEEDS[s] - v) < 1e-9
               for s, v in ((42, 0.55), (43, 0.5213), (44, 0.5387)))
