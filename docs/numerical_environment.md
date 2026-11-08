# Numerical environment, BLAS determinism, and GPU runtime reconciliation

## 1. CPU frozen-feature BLAS determinism (`v1.1.2` / `v1.2.0`)

The `v1.1.1` frozen-feature repeated-partition tables (`results/generalization/REPEATED_SPLIT_METRICS.csv` and `REPEATED_SPLIT_SUMMARY.csv`) agree with numerical recomputation using a single BLAS thread (`threadpoolctl.threadpool_limits(limits=1)`). Multi-threaded linear algebra in some environments changes the tie-breaking order of near-equal floating-point probabilities and yielded tiny ROC-AUC/AP differences in three text-probe rows (up to `0.0005965` ROC-AUC and `0.0001427` AP), while leaving binary predictions, confusion matrices, Macro F1, and primary focal effects unchanged. On Apple Silicon ARM64 (`Accelerate`) versus x86_64 (`OpenBLAS`), `liblinear` coordinate descent exhibits a maximum probability difference of `0.00128` on 5 out of 15,640 rows while producing identical binary predictions (`15,640/15,640`) and identical Macro F1 (`< 1e-8`).

The archived `REPEATED_SPLIT_*` files **have not been modified**, and their SHA-256 hashes match `v1.1.1`.

## 2. GPU matmul precision and neural-head sensitivity reconciliation (`R1.2`)

In the V4 reviewer revision (`results/video/revision_20261108/head_sensitivity/HEAD_BASELINE_RECONCILIATION.csv`), the 48-run 2×3×2 factorial neural-head sensitivity grid (`weight_decay ∈ {1e-4, 1e-3}`, `dropout ∈ {0.1, 0.2, 0.3}`, `head_lr ∈ {5e-4, 1e-3}`) at seed `42` was executed in a single consistent PyTorch environment where `torch.set_float32_matmul_precision('high')` was not recorded. By contrast, the primary `GoldAdjudicated v1.1` training run (`VIDEO_OOF_PREDICTIONS.csv`) explicitly set `torch.set_float32_matmul_precision('high')` on an NVIDIA A100 GPU.

Consequently, at the matched hyperparameter cell `(weight_decay=1e-4, dropout=0.2, head_lr=1e-3)`, seed-42 Macro F1 differs slightly between the frozen primary run and the sensitivity replay:
- `visual_linear`: `0.48324` (frozen primary) vs `0.47657` (sensitivity replay, `Δ = −0.00668`)
- `audio_mlp`: `0.50670` (frozen primary) vs `0.51065` (sensitivity replay, `Δ = +0.00395`)
- `feature_mlp`: `0.56295` (frozen primary) vs `0.55679` (sensitivity replay, `Δ = −0.00616`)
- `healnet`: `0.56125` (frozen primary) vs `0.54865` (sensitivity replay, `Δ = −0.01260`)

Primary Table 3 retains the frozen `v1.1` predictions unchanged; the 48-run sensitivity grid is evaluated internally across its consistent runtime (`python scripts/validation/verify_revision.py --root .`).
