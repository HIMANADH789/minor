"""Statistics for the Stack study: reuses the shared utilities with
Stack-specific seeds via a fresh RNG pool."""
import numpy as np

from src.diagnostics import statistics as _S

# Stack-specific seeds (independent of the Lite study)
_S.BOOTSTRAP_SEED = 5200
_S.PERMUTATION_SEED = 5300
_S.N_BOOTSTRAP = 2000
_S.N_PERMUTATION = 2000
_S.RNG_POOL = {}   # fresh pool so seeds rebind deterministically

bootstrap_ci = _S.bootstrap_ci
bootstrap_diff_ci = _S.bootstrap_diff_ci
bootstrap_auroc_ci = _S.bootstrap_auroc_ci
corr_with_inference = _S.corr_with_inference
paired_permutation_test = _S.paired_permutation_test
wilcoxon_signed = _S.wilcoxon_signed
mcnemar = _S.mcnemar
cohens_d_paired = _S.cohens_d_paired
cliffs_delta = _S.cliffs_delta
cliffs_label = _S.cliffs_label
rank_biserial_from_diffs = _S.rank_biserial_from_diffs
benjamini_hochberg = _S.benjamini_hochberg
HypothesisRegistry = _S.HypothesisRegistry
jsonable = _S.jsonable
dump_json = _S.dump_json
dump_csv = _S.dump_csv
