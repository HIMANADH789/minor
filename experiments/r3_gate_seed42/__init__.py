"""R3: Learned Continuous Heterogeneity Gate (frozen-R2 extension, seed 42).

F = [G || w*H], w = sigmoid(theta) (ONE dataset-level scalar),
linear softmax classifier, L = CE + lambda_clf*||W||^2 + lambda_w*theta^2.
Stages 1-3 (MiniROCKET, SSL encoder, HardVQ) are the completely frozen
audited R2 pipeline; only the gate + linear classifier are trained.
"""
