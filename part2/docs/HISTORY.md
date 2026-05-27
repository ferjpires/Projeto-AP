# HISTORY — Discarded Experiments and Negative Results

This document catalogues approaches that were tried but did not make the final
pipeline (E0–E6 in `README.md`). They are kept here so the report can discuss
*why* each was discarded and what was learned. All numbers refer to test-set
F1 macro on the official MIQR-CC split (train 1 067 / val 234 / test 267).

The current best result is **E6 = 0.7483** (mixed-architecture stacking
ensemble of DenseNet-121 + ConvNeXt-tiny × 5 seeds).

---

## 1. Baselines without preprocessing or asymmetric loss

| Variant | F1 macro | Why discarded |
|---|---|---|
| ResNet-18 + weighted CE, no CLAHE | 0.4467 | Dominated by the same model with CLAHE (0.5094 → eventually 0.5442 with the final pipeline). |
| EfficientNet-B0 + weighted CE, no CLAHE | 0.3593 | Dominated by E1 (CLAHE variant = 0.5470). |
| EfficientNet-B0 + weighted CE + WeightedRandomSampler, *old pipeline* | 0.3506 | Same configuration re-trained on the final pipeline (drop\_last, RandAugment, 2-stage warmup, 320 px) reaches **0.6130** as E3, a +0.26 F1 jump. Documenting the gap motivates the pipeline contributions in the report. |

**Takeaway.** CLAHE alone is responsible for a ≈ +0.07 F1 lift on ResNet-18
and ≈ +0.07 on EfficientNet-B0 with the *old* pipeline. The combination of
augmentation + 2-stage training amplifies the effect of oversampling
specifically (E3 +0.26), suggesting the sampler benefits disproportionately
from richer per-batch diversity.

---

## 2. Single-seed reporting on the densenet-focal recipe

Before adopting multi-seed ensembling we ran the same recipe four times at
seed 42 with subtle variations of the pipeline. F1 macro spanned
**0.5421 – 0.7029** with σ ≈ 0.04 — i.e. a single run is a poor estimate of
the recipe's true quality on this dataset (Biliary_Leaks has 17 test samples,
so a single sample misclassified moves recall by 5.9 pp). This variance
discovery is what motivated:

1. The 5-seed protocol for ensemble feeders (E2, E4).
2. Reporting val/test pairs side-by-side so the reader sees the generalization
   gap.

---

## 3. Augmentation / preprocessing ablations that did not help

| Variant | F1 macro | Why discarded |
|---|---|---|
| ColorJitter with saturation & hue on grayscale fluoroscopy | — | Adds noise on single-channel images; replaced with brightness+contrast only when RandAugment is on (already covered by RandAugment otherwise). |
| Logit adjustment at test-time (subtract `log p(class)`) | hurts post-RandAugment | Was a small win on weakly-augmented runs, but RandAugment + RandomErasing already implicitly debias the model and the additional shift hurts macro F1. |
| ImageNet-22k pretrained ConvNeXt-tiny (`convnext_tiny.fb_in22k_ft_in1k`) | 0.6486 | Single-seed comparison vs in1k pretrain (0.6730); val F1 was 0.08 lower (0.59 vs 0.67) — the 22k features did not match our fine-tuning recipe well. Not worth retraining four more seeds. |

---

## 4. Test-Time Augmentation variants

We measured F1 of the mixed-architecture ensemble (10 ckpts) under several
TTA strategies (`scripts/eval_ensemble.py --combine avg` ablation):

| TTA mode | F1 macro |
|---|---|
| identity only | **0.7306** |
| identity + horizontal flip | 0.6636 |
| identity + h-flip + v-flip | 0.6597 |

Per-view diagnostics (logits averaged across all 10 ckpts):

| View | F1 alone |
|---|---|
| identity | 0.7306 |
| horizontal flip | 0.6477 |
| vertical flip | **0.3492** |

**Vertical flip is catastrophic** on ERCP fluoroscopy because the anatomy has
a fixed orientation (duodenoscope enters from above, ducts run inferiorly).
**Horizontal flip helps a single-architecture ConvNeXt ensemble** (+0.014)
but hurts the mixed ensemble (−0.067) — DenseNet predictions under hflip are
substantially worse than ConvNeXt, so mixing them only adds noise. The final
pipeline uses **no TTA**.

Multi-scale TTA (320 + 352) was also tested in [post-hoc Phase 1](#7-post-hoc-improvements-that-did-not-help)
and made things worse: the checkpoints were trained at 320, so 352 inference
is a domain-shifted view that lowers per-ckpt accuracy faster than ensembling
gains can compensate.

---

## 5. Architectural alternatives that were dropped

### 5.1 EfficientNet-B7 (`tf_efficientnet_b7.ns_jft_in1k`)

The other team's best individual model. We trained one seed at 320 px,
batch 8, identical recipe to ConvNeXt-tiny. Result:

| Metric | EfficientNet-B7 | ConvNeXt-tiny (reference) |
|---|---|---|
| Val F1 | 0.6120 | 0.6727 |
| **Test F1** | **0.5677** | **0.6730** |
| Biliary precision | 0.34 | 0.54 |
| Train F1 (last epoch) | 0.97 | ≈ 0.97 |
| Early stop epoch | 25 / 55 | ≈ 48 / 55 |

**Why discarded.** 64 M parameters on 1 067 training samples → severe
overfitting; train F1 ≈ 1.0 while val plateaus near 0.6. B7's native input
resolution is 600 px; running at 320 produces a domain mismatch with its
Noisy Student pretrain. Even if averaging across 5 seeds could lift its
ensemble contribution, starting at 0.57 single-seed (vs ConvNeXt-tiny's 0.67)
provides too weak a signal — adding B7 checkpoints would more likely add
noise than diversity to the LogReg meta-learner.

### 5.2 ConvNeXt-tiny at 512 × 512

Tested as a resolution upgrade. Training stalled — train F1 around 0.18 for
seven consecutive stage-2 epochs (vs ≈ 0.9 at 320 px). The most likely cause
is fp16/AMP overflow on the larger activation maps under ROCm on the AMD GPU.
Run was killed and not pursued further given the deadline. A potential fix
is to disable AMP at 512 px (fp32 only, ≈ 2× slower) but the expected gain
(+0.01–0.02) did not justify the debug time.

---

## 6. Intermediate ensembles superseded by the mixed-arch stacker

Each of these was the best result *at the time it was produced* and is
captured here to motivate the progression in the report:

| Ensemble | F1 macro | Superseded by |
|---|---|---|
| DenseNet-121 (5 seeds), simple logit averaging + hflip TTA | 0.6795 | Adding ConvNeXt-tiny (E4) and dropping hflip → 0.7306 |
| ConvNeXt-tiny (5 seeds), simple logit averaging + hflip TTA | 0.7043 | Mixing with DenseNet → 0.7306 |
| Mixed-arch (10 ckpts), simple logit averaging | 0.7306 | Logistic-Regression stacker on val probs → **0.7483** |
| Top-K subset of ConvNeXt by val F1 (top-2 / top-3) | 0.6683 / 0.6861 | Full 5-seed average is better — weak seeds still decorrelate errors. |

---

## 7. Post-hoc improvements that did not help

Once the 10-checkpoint mixed-arch ensemble was in place, we tried several
post-hoc strategies on the cached logits. None improved on the LogReg
stacker (C = 0.01) which sits at **0.7483 test / 0.7688 val**.

| Variant | Test F1 | Val F1 |
|---|---|---|
| **LogReg stacker, C = 0.01 (chosen)** | **0.7483** | **0.7688** |
| Plain average of logits (ablation) | 0.7306 | 0.6476 |
| LogReg on average probs only (4 features) | 0.7460 | 0.6784 |
| LogReg stacker, C = 0.001 (more reg.) | 0.7339 | 0.6907 |
| LogReg, C = 1.0 (under-regularized) | 0.6636 | **0.8698** ← memorized val |
| 352 px logits | 0.7253 | 0.7848 |
| Multi-scale 320 + 352 | 0.7233 | 0.7786 |
| Per-ckpt temperature scaling + LogReg | 0.7048 | 0.7689 |
| Youden's-J per-class threshold optimization | 0.6722–0.6864 | 0.6893 |
| **XGBoost stacker** | 0.6155 – 0.6533 | **1.0000** ← severe overfit |
| Top-3 by val F1 (subset ensemble) | 0.6765 | 0.6801 |
| Weighted average by val F1 | 0.7214 | 0.6776 |
| Weighted average by val F1² (sharpen) | 0.7124 | 0.6776 |

**Takeaways:**

1. **Threshold optimization is fragile with small val sets.** With only 24
   Biliary_Leaks validation samples, Youden's J on the per-class ROC curve
   recommended τ ≈ 0.028 — a 35× amplification of the Biliary score on test,
   which over-predicts the class. Floors of 0.05, 0.10, 0.15, 0.20 were
   tested; all still hurt.
2. **Per-ckpt temperature scaling adds noise.** Fitting one T per ckpt by
   minimizing NLL on 234 samples produces noisy T's (1.0–3.3) that the
   stacker does not benefit from.
3. **Multi-scale TTA is a domain shift, not a free signal.** Inferring
   training-at-320 ckpts at 352 lowers per-ckpt accuracy; averaging with
   320 only smears the signal.
4. **XGBoost has too much capacity for 234 stacking samples.** Even at
   `n_estimators=200, max_depth=3` it perfectly fits the val set and
   generalizes poorly. LogReg is the right choice at this dataset size.
5. **Strong regularization is essential for the stacker.** `C=0.01` (heavy
   L2) is the sweet spot. `C=1.0` looks brilliant on val (0.87) but
   collapses on test (0.66) — a useful caution for anyone tuning by val F1
   alone.

---

## 8. Dataset issues (chronological note)

The original experiments (E0–E5 in early iterations) were briefly run with a
**stratified split we generated ourselves**, before the official MIQR-CC
split was applied. Results on that split inflated F1 macro by ~0.20 due to
leakage between the train and val/test sets. The numbers reported in this
project — including the discarded variants above — all use the **official
fixed split** from the MIQR-CC repository
(train 1 067 / val 234 / test 267). The investigation of single-seed
variance described in §2 was triggered by the F1 collapse observed when we
moved to the official split.

---

## 9. Reproducibility note for the discarded approaches

The dropped checkpoints and metrics CSVs live under `models/archive/` and
`outputs/archive/`. They can be deleted without affecting the active
pipeline, but are kept so reviewers can verify the negative results
referenced in this document.
