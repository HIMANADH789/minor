"""CAPACITY-CONTROLLED NESTED/HIERARCHICAL REGIME-CONDITIONED HETEROGENEITY.

Companion module to models/nested_regimes: keeps the verified ANOVA-chain,
null, and stopping machinery untouched and adds the pieces the UWaveY
capacity experiment needs (spec section 32: reuse-or-extend, do not disturb
the Haptics artifacts):

  * level_budgets: energy-share allocation of the PREDECLARED fixed budget
    B_H across retained levels via deterministic largest-remainder rounding
    (floor + largest fractional remainder, tie-break lower K first), so the
    sum is EXACTLY B_H for every input.

  * edge_energies: per-candidate-feature structural energies
    e_{m,e,l} = mean_i [ pi_{i,c} Delta_{i,m,c}^2 ]
    over TRAIN samples, from the chain's per-sample per-kernel energies and
    detail banks.  Per level l with K = 2^l, E_{i,l}(m) =
    sum_c pi_c Delta^2, and the per-edge quantity is defined by REDUCING the
    (n, K, F) delta arrays to parent/child groupings exactly like the chain
    itself does; for K in {4, 8, 16} this reduces (numerically EXACTLY) to
    e = mean_i [ n_c/|S16| * Delta_{i,m,c}^2 ] per (child c, kernel m).

  * select_carriers: keep the top-B_l candidates per level by e, ties broken
    by (level, kernel, child/parent id) -- LABEL-FREE: no labels, no
    validation, no test, no classifier coefficients enter.

  * gather_features: stack the selected Delta columns into (N, B_H) banks
    for train / val / test (frozen transform everywhere).

All quantities are float64; selection is deterministic (argpartition with
explicit tie ordering -> stable).  No Gram-Schmidt, no CCA.
"""
import numpy as np

from models.nested_regimes.model import LEVELS  # shared constants

BUDGET_H = 4998                   # predeclared capacity (== flat H)
TIE_TOL = 0.0                     # exact float equality


def level_budgets(real_E, retained_K, budget=BUDGET_H):
    """Largest-remainder allocation of `budget` across retained levels.

    real_E: (4,) level totals ordered K=2,4,8,16 (chain "E_total").
    retained_K: list of retained level K values (the stopping rule's kept
    levels, coarse->fine).  Returns (list of K, array of ints summing to
    exactly `budget`).

    w_l = E_l / sum_{j in retained} E_j  (intrinsic energies only; if all
    retained energies are 0 -- never observed -- equal shares).
    """
    idx = [LEVELS[1:].index(k) for k in retained_K]
    E = np.asarray(real_E, dtype=np.float64)[idx]
    tot = E.sum()
    w = (E / tot if tot > 0 else np.full(len(E), 1.0 / len(E)))
    raw = w * float(budget)
    base = np.floor(raw).astype(np.int64)
    rem = budget - int(base.sum())
    frac = raw - base
    order = np.lexsort((np.asarray(retained_K), -frac))   # -frac, then lower K
    out = base.copy()
    if rem > 0:
        out[order[:rem]] += 1
    assert int(out.sum()) == budget
    return list(retained_K), out


def edge_energies(ch, n_samples):
    """Per-candidate-feature structural energies from one chain result.

    ch: output of models.nested_regimes.model.hierarchy_chain
    (want_deltas=True).  Returns dict K -> (K, F) edge/carrier energies.

    Computed EXACTLY inside hierarchy_chain per padding group (each
    kernel's own window and the group's own occupancy weights, the same
    convention as E), accumulated over kernels and divided by n -- see
    the chain's "edge_e" output.  Derived from per-sample quantities
    only -- no labels, no validation.
    """
    ee = ch.get("edge_e")
    if ee is None:
        # Raw chain_from_counts_sums result (single padding group): the
        # same math applied once, exactly.
        if "cnts" not in ch or "deltas" not in ch:
            raise RuntimeError(
                "chain output lacks 'edge_e' -- call hierarchy_chain with "
                "want_deltas=True (exact per-group edge energies)")
        nv = ch["n_support"].astype(np.float64)
        nvs = np.maximum(nv, 1.0)[:, None]
        ee = {}
        for li in range(len(LEVELS) - 1):
            K = LEVELS[1:][li]
            if K == 2:
                e2 = ch["E"][:, 0, :].sum(axis=0)
                ee[2] = np.stack([e2, e2], axis=0)
            else:
                piK = ch["cnts"][K] / nvs
                ee[K] = (piK[:, :, None]
                         * ch["deltas"][K] ** 2).sum(axis=0)
    return ee


def select_carriers(edge_e, budgets, retained_K):
    """Label-free top-energy carrier selection per level.

    edge_e: dict K -> (K, F) from edge_energies.
    budgets: dict K -> B_l ints (level_budgets output).
    Returns dict K -> (B_l, 2) int array of selected (kernel, child) ids,
    ordered (kernel, child) ascending; candidates exhausted -> loud fail.

    Candidate universe at level l: (kernel, child) pairs (F x K).  The two
    K=2 aliases share one underlying parent edge (edge_energies maps both
    child ids to the same parent-level energy), so they are deduplicated to
    a single candidate per kernel: selecting child id c at K=2 contributes
    ONE detail column (the parent-relative residual pair) -- see gather.
    """
    sel = {}
    for K in retained_K:
        e = edge_e[K]                                    # (K, F)
        B_l = int(budgets[K])
        F = e.shape[1]
        # (kernel, child) candidate matrix (K=2 aliases collapse to one)
        child_ids = range(e.shape[0]) if K != 2 else (0,)
        cand = [(float(e[c, m]), m, c) for c in child_ids for m in range(F)]
        if len(cand) < B_l:
            raise RuntimeError(
                f"level K={K}: {len(cand)} candidates < budget {B_l} "
                f"-- failing loudly instead of changing the budget")
        # deterministic sort: energy desc, then kernel, then child id
        cand.sort(key=lambda t: (-t[0], t[1], t[2]))
        pick = cand[:B_l]
        arr = np.asarray([(m, c) for _, m, c in pick], dtype=np.int64)
        sel[K] = arr[np.lexsort((arr[:, 1], arr[:, 0]))]  # (kernel, child)
    return sel


def gather_features(sel, deltas_by_split, n_features=BUDGET_H):
    """Stack selected detail columns into dense banks.

    sel: dict K -> (B_l, 2) (kernel, child) ids.
    deltas_by_split: dict split -> dict K -> (n, K, F) detail arrays.
    Returns dict split -> (n, B_H) float64 arrays (column order: level
    coarse->fine, then (kernel, child) as selected).
    """
    cols = []
    for K in sorted(sel.keys()):
        cols.append((K, sel[K]))
    out = {}
    for split, dmap in deltas_by_split.items():
        parts = []
        for K, arr in cols:
            d = dmap[K]                                  # (n, K, F)
            if K == 2:
                # one column per kernel: parent-relative residual of the
                # SELECTED child (bit 0/1 child of the single K=2 parent)
                n = d.shape[0]
                parts.append(d[np.arange(n)[:, None],
                               arr[None, :, 1], arr[None, :, 0]])
            else:
                parts.append(d[:, arr[:, 1], arr[:, 0]])
        bank = np.hstack(parts)
        assert bank.shape[1] == n_features, bank.shape
        out[split] = bank
    return out


def dataset_energy_summary(ch, retained_K):
    """Dataset-level descriptive summary (spec section 25): HI_hier (median
    train sample total energy), per-level fractions."""
    Es = ch["E_sample"]                                   # (n, 4)
    idx = [LEVELS[1:].index(k) for k in retained_K]
    tot = Es[:, idx].sum(axis=1)
    frac = Es[:, idx].sum(axis=0) / max(Es[:, idx].sum(), 1e-30)
    return {"HI_hier_median_sample_total_energy": float(np.median(tot)),
            "HI_hier_mean_sample_total_energy": float(np.mean(tot)),
            "energy_fractions": {int(k): float(f) for k, f in
                                 zip(retained_K, frac)}}
