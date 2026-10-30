# Final-Year Project Presentation Blueprint

## Positioning

**Title:** Explainable Multimodal Virality Prediction for Short-Form Social Media Content

**Contribution:** A pre-publication machine-learning system that combines visual, acoustic, textual, and temporal signals, evaluates fusion under a temporal split, and explains predictions with TreeSHAP.

The presentation must state dataset scope clearly and must not claim universal real-world virality prediction.

## Twelve-slide sequence

1. **Title** — project, team, mentor, department, Emerging Technologies.
2. **Problem context** — creators need a pre-publication estimate; raw views depend on account size and platform dynamics.
3. **Research gap** — metadata-only and single-modality systems miss cross-modal interactions; unexplained predictions are difficult to use.
4. **Objectives and research question** — target definition, four signal groups, fusion comparison, temporal evaluation, TreeSHAP.
5. **Methodology** — video/audio/caption/schedule → extraction → fusion → XGBoost → explanation.
6. **Target and leakage control** — follower-normalized engagement, percentile tiers, excluded post-publication features, temporal split.
7. **Feature engineering** — the 57-dimensional vector grouped by modality.
8. **Experimental design** — metadata, text, audio, visual, and full fusion using Macro-F1, weighted F1, ROC-AUC, confusion matrix, and regression errors.
9. **Results** — one authoritative table only; resolve conflicting Macro-F1 values first.
10. **Explainability** — one prediction with positive and negative feature contributions; SHAP is attribution, not causal proof.
11. **Limitations and ethics** — dataset scope, bias, platform dependence, uncertainty, privacy, and responsible collection.
12. **Conclusion and future work** — validated contribution, real longitudinal data, creator-disjoint testing, calibration, uncertainty, cross-platform evaluation.

## Viva answers to prepare

- **Why XGBoost?** The fused representation is heterogeneous tabular data, so XGBoost models nonlinear interactions efficiently and supports tree-based explanations.
- **Why temporal splitting?** It better approximates future-content prediction and reduces optimistic results from related trends appearing in both partitions.
- **Why Macro-F1?** Virality tiers are imbalanced, so each class receives equal importance.
- **What is the novelty?** The integrated, leakage-aware, multimodal, explainable pre-publication pipeline and ablation-based evaluation.
- **Can virality be guaranteed?** No. The system estimates historical patterns; algorithms, audience behavior, distribution, and trends remain partly unobserved.

## Work order

- Wednesday: freeze the story, resolve metric conflicts, select the authoritative result table.
- Thursday: generate methodology diagram, feature table, confusion matrix, ablation chart, and SHAP example.
- Friday: write the report sections and first slide draft.
- Saturday: edit claims, add limitations, and prepare speaker notes.
- Sunday: rehearse, export PDF, and freeze the final version.
- Monday: submit without adding new technical features.
