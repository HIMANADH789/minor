# R4 -- Differentiable-Ridge Regime-Level Heterogeneity Gate (seed 42)

Final refinement of R2/R3.  Stages 1-3 are the frozen audited R2 pipeline
(fixed MiniROCKET, frozen SSL encoder, frozen hard VQ K=8).  The only new
mechanisms are:

  * 8 regime-level sigmoid gates: H_m^gate = sum_k v_k H_{m,k},
    v_k = sigmoid(theta_k), init theta_k = -2
  * a closed-form differentiable Ridge as the classifier (the R3 softmax
    head was REMOVED to eliminate the classifier confound): the dual
    Ridge is re-solved in closed form at every outer step and gradients
    flow through the solve into theta via the validation one-hot MSE.

Identity guarantees (audited per dataset): v=1 reproduces the audited R2
H block and the R2 Ridge decisions EXACTLY (agreement 1.000); v=0 reduces
the representation to G.  Ridge alpha is frozen at the R2-selected values
(Haptics 4.2813, ECG5000_BAL 1.6238) and never re-optimized.

Layout: per-dataset dirs hold audits.json, validation/test results,
gate_metrics, regime_statistics, solver_diagnostics, diagnostics,
predictions/, checkpoints/, training_logs/gate_trajectory.{json,csv}.
config.json records the full frozen configuration.  REPORT.md is the
final report.  Figures live in figures/.
