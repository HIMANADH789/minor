"""
PHASE 16-25 analyses: complementarity, statistical inference (reusing the
project's src/diagnostics/statistics.py utilities), and dataset
characteristics. No new statistical tests are invented.
"""
import numpy as np

from src.diagnostics.statistics import (  # noqa: E402  (canonical project utils)
    mcnemar,
    benjamini_hochberg,
    paired_permutation_test,
    cohens_d_paired,
)


def correct_vector(preds, y):
    return np.asarray(preds) == np.asarray(y)


def mcnemar_between(preds_a, preds_b, y):
    """Paired McNemar between two models on the same test set."""
    chi2, p = mcnemar(correct_vector(preds_a, y),
                      correct_vector(preds_b, y))
    return {"chi2": round(float(chi2), 4), "p_raw": round(float(p), 6)}


def cross_model_complementarity(y, preds_mr, branch_preds, stack_preds):
    """PHASE 18: error overlap / agreement / prob-correlation between
    MiniROCKET, each Stack branch, and the combined Stack prediction.

    error_overlap(a, b) = #{both wrong} / #{either wrong}  (canonical def).
    """
    y = np.asarray(y)
    models = {"MiniROCKET": preds_mr}
    for k, v in branch_preds.items():
        models[f"branch_{k}"] = v
    models["Stack"] = stack_preds

    out = {"agreement": {}, "error_overlap": {}, "mr_corrected_by": {}}
    names = list(models)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            pa, pb = np.asarray(models[a]), np.asarray(models[b])
            out["agreement"][f"{a}|{b}"] = round(float((pa == pb).mean()), 4)
            ea, eb = pa != y, pb != y
            both, either = ea & eb, ea | eb
            out["error_overlap"][f"{a}|{b}"] = round(
                float(both.sum() / max(either.sum(), 1)), 4)

    # does Stack fix MiniROCKET's mistakes? (and vice versa)
    e_mr = np.asarray(preds_mr) != y
    e_st = np.asarray(stack_preds) != y
    out["mr_wrong_stack_right"] = int((e_mr & ~e_st).sum())
    out["mr_right_stack_wrong"] = int((~e_mr & e_st).sum())
    out["both_wrong"] = int((e_mr & e_st).sum())
    out["mr_error_rate"] = round(float(e_mr.mean()), 4)
    out["stack_error_rate"] = round(float(e_st.mean()), 4)
    return out


def seed_robustness(delta_mf1_by_seed):
    """PHASE 22 helper: paired stats over seeds (Stack vs MiniROCKET)."""
    stack = [s["stack_mf1"] for s in delta_mf1_by_seed]
    mr = [s["mr_mf1"] for s in delta_mf1_by_seed]
    deltas = [s - m for s, m in zip(stack, mr)]
    _, p = paired_permutation_test(stack, mr)
    return {
        "n_seeds": len(deltas),
        "deltas": [round(d, 4) for d in deltas],
        "mean_delta": round(float(np.mean(deltas)), 4) if deltas else None,
        "std_delta": round(float(np.std(deltas, ddof=1)), 4)
        if len(deltas) > 1 else 0.0,
        "n_positive": int(sum(d > 0 for d in deltas)),
        "n_negative": int(sum(d < 0 for d in deltas)),
        "paired_permutation_p": round(float(p), 6),
        "effect_size_d": round(float(cohens_d_paired(stack, mr)), 4)
        if len(deltas) > 1 else None,
    }


def dataset_characteristics(d, mr_class_recall):
    """PHASE 25: difficulty characteristics (exploratory only)."""
    ytr = d["ytr"]
    _, cnt = np.unique(ytr, return_counts=True)
    props = cnt / cnt.sum()
    minority_recall = None
    minority_classes = np.where(props <= props.max() / 2)[0]
    if len(minority_classes) and mr_class_recall is not None:
        vals = [mr_class_recall[c] for c in minority_classes
                if c < len(mr_class_recall)]
        minority_recall = round(float(np.mean(vals)), 4) if vals else None
    return {
        "n_classes": d["n_classes"],
        "sequence_length": d["L"],
        "n_train": int(len(ytr)),
        "n_val": int(len(d["yva"])),
        "n_test": int(len(d["yte"])),
        "class_entropy_bits": round(float(
            -(props * np.log2(props)).sum()), 4),
        "imbalance_ratio_max_min": round(
            float(props.max() / max(props.min(), 1e-12)), 2),
        "minority_class_proportion": round(float(props.min()), 4),
        "mr_minority_class_recall": minority_recall,
    }
