# Methodology: TB Hotspot Prioritization (Two-Stage Count Model)

## Problem

Active case finding (ACF) teams cannot screen an entire city every month.
The operational question is a **ranking** problem:

> Which H3 hexagons should be visited next month to find the most undiagnosed TB cases?

## Spatial unit

We use Uber **H3** cells (commonly resolution 9–10 for urban ACF).
Hexagons give roughly uniform area, clean neighbor rings, and stable ids across months.

## Target

For each hexagon H and month M:

- Features are computed from data available at month **M** (and earlier).
- Label is the case count in month **M+1** (`target_cases`).
- Binary flag `has_any_cases = 1[target_cases > 0]` trains the Stage-1 gate.

This temporal shift is mandatory — using same-month labels is leakage.

## Features (summary)

| Family | Examples |
| --- | --- |
| Clinical | `total_cases`, pulmonary share, relapse rate, HIV/diabetes counts, infectivity load |
| Temporal | lags T-1..T-3, 3/6-month rolling sums, MoM trend, persistence, days since last case |
| Contact tracing (optional) | `n_contacts` |
| Spatial | ring-1 neighbor case load, slum/vulnerability density |
| Interactions | cases×slum, neighbor×slum, transmission vulnerability score |

Bring-your-own data may omit optional columns; the pipeline fills sensible defaults.

## Model

**Two-stage hurdle** — this mirrors the production `TwoStageHurdleModel`
(AI-ACF-ML-Prediction, `api/services/two_stage_model.py`) so the open-source
notebooks are a faithful distillation of the deployed staging pipeline, not
a simplified rewrite:

1. **Gate** — `XGBClassifier` predicts P(any cases). Handles severe class imbalance via `scale_pos_weight`.
2. **Calibrate** — an `IsotonicRegression` calibrator is fit on the Stage-1
   *validation* fold (never on training rows) to correct gate probabilities
   distorted by the `scale_pos_weight` reweighting.
3. **Count** — `XGBRegressor` with `count:poisson`, trained **only on non-zero** labels.
4. **Combine** — `expected_cases = calibrated_gate_prob × count_pred`
   (multiplicative, continuous — not a hard threshold cutoff). A hard
   threshold would zero out borderline hexagons right at the gate boundary
   and discard exactly the probability information the calibration step
   just fixed.

A single-stage Poisson is trained as a baseline for comparison.

> **Known limitation (shared with production):** Stage 2 is trained on a
> zero-truncated subsample (`target_cases > 0` rows only) using a plain
> Poisson objective, not a formal zero-truncated Poisson. This systematically
> underestimates `E[cases | cases > 0]` and is part of why this hurdle
> architecture underperforms a Poisson-GAM baseline on cold-start hexagons
> in production benchmarks (see the internal model version review). It's an
> accepted simplification for ranking/Top-K use, not a bug — just something
> to be aware of if you need calibrated absolute counts rather than ranking.

## Evaluation

- **Classification:** AUC-ROC, precision, recall for the gate  
- **Regression:** RMSE / MAE / Spearman on counts  
- **Operational:** Precision@K, Efficiency@K (cases captured), Lift@K vs random  
- **Walk-forward:** retrain on months 1..t, score month t+1  

For field use, ranking metrics matter more than exact count calibration.

## Inference

Score the latest feature month, assign **High / Medium / Low** tiers from predicted count + gate probability, and export a compact JSON list for dashboards / maps.

## What this open release intentionally omits

- Real patient or facility datasets  
- Production FastAPI / database / cloud storage code  
- Pretrained weights fit on proprietary program data  

Use `sample_data/` to learn the method; swap in your own CSVs for real runs.
