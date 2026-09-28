# R3 final -- Global Continuous Heterogeneity Gate + Differentiable Ridge (seed 42)

The clean rerun of R3 with the classifier confound removed: the R3
SGD-softmax head is REPLACED by an exact differentiable closed-form dual
Ridge.  One learned scalar w = sigmoid(theta) gates the frozen R2 H
branch: X(w) = [G || w*H], dim 9996.  At every outer step the dual Ridge
K(w) = Kg + w^2 Kh + alpha I (exact Gram decomposition) is re-solved
from TRAIN rows only; the validation one-hot MSE backpropagates through
the solve into theta.  Alpha frozen at the R2-selected values
(Haptics 4.2813, ECG5000_BAL 1.6238).

Identity guarantees (audited per dataset): at w=1 the feature matrix and
the Ridge decisions equal R2 exactly (agreement 1.000); at w=0 the
representation reduces to G.  Controls: G-only and [G||H] fit with the
canonical RidgeClassifierCV protocol.

Layout: per-dataset dirs hold audits.json, validation/test results,
gate_metrics, solver_diagnostics, diagnostics, predictions/, 
checkpoints/, training_logs/gate_trajectory.{json,csv}.  config.json,
REPORT.md, figures/.
