# Representation, Transferability, and Trustworthy Uncertainty in Machine-Learning Prediction of the HOMO–LUMO Gap of Transition-Metal Complexes

Fully reproducible, DFT-free machine-learning study built on the public **tmQM** database (108,541 mononuclear transition-metal complexes, 3d/4d/5d). **One notebook reproduces every figure and table of the paper and of its revision.**

## Findings in one paragraph
Accuracy rises monotonically with representational richness — composition + coordination (R² ≈ 0.57) → + RDKit descriptors (0.63) → SchNet on 3D geometry (0.77) → DimeNet++ (0.79) — but in-domain accuracy is a poor guide to reliability: under leave-one-series-out and leave-one-metal-out both geometry networks collapse to R² ≤ 0.3 and never beat the simple tabular model on the 4d/5d series, their out-of-domain scores vary by up to 0.3 between training replicates, and standard conformal intervals under-cover on an unseen series (81% vs 90%). Series-conditional (Mondrian) calibration and a distance-based applicability domain restore calibrated, selective predictions.

## Repository contents
| File | Description |
|---|---|
| `tmQM.ipynb` | Complete end-to-end notebook (Kaggle/Colab GPU). Recommended. |
| `tmqm.py` | Equivalent single script, auto-generated from the notebook. |
| `revision_analyses.py` | CPU-only re-analysis (bootstrap CIs, repeated splits, target statistics, outliers, timings) used for the revision; all of it is also inside the notebook. |
| `requirements.txt`, `LICENSE` (MIT), `.gitignore` | |

## Run
1. Open `tmQM.ipynb` on Kaggle (recommended: 12-h GPU sessions) or Colab with a GPU runtime.
2. First pass: set `QUICK = True` in the CONFIG cell and *Run all* (≈ 10 min smoke test). Then `QUICK = False` and *Run all*.
3. On Kaggle use *Save & Run All (Commit)*; `tmqm_results.zip` appears at the top level of the Output tab.

`PROFILE = "full"` (default) uses the paper's exact settings for every model (SchNet 40/20 epochs; DimeNet++ 3 blocks, 5 Å, 15 epochs) and takes ≈ 10 h on an NVIDIA T4. A time guard defers any fold that would exceed 11.5 h — run the notebook again and it resumes from the saved folds. `PROFILE = "12h"` uses a lighter DimeNet++ (≈ 8 h) for slower GPUs.

No API tokens are required. Two optional extras can be enabled in CONFIG: `USE_TABPFN` (a tuning-free TabPFN baseline; needs a Prior Labs `TABPFN_TOKEN`) and `HF_SYNC` (archive/resume through a private Hugging Face dataset; needs a Hugging Face token with the *Write* role).

## What the notebook produces (`tmQM_data/results/`)
- Tabular: `model_comparison.csv` (with 95% bootstrap CIs), `indomain_repeated_splits.csv`, `xgb_sensitivity.csv`, `timing.json`, `outliers_xgb_gt2eV.csv`, `fig_parity.png`, `fig_shap_beeswarm.png`, `shap_global.csv`, `shap_labels.csv`
- Target statistics: `gap_statistics.csv`, `figS_gap_hist.png`, `target_shift_loso.csv`
- Transferability: `transfer_loso.csv`, `transfer_lomo.csv` (with CIs and per-metal target SD), `loso_tabular_per_metal.csv`, `fig_transfer_loso.png`, `fig_transfer_lomo.png`
- Uncertainty: `conformal_reliability.csv`, `conformal_applicability.csv`, `conformal_mondrian.csv`, `conformal_applicability_domain.csv`, `conformal_selective_curve.csv` and the four `fig_conformal_*.png`
- Geometry models: `gnn_results.json` (SchNet and DimeNet++: in-domain, LOSO, LOMO-Zn, bootstrap CIs), `predictions_*.csv`, `learning_curves.csv`, `fig_S2_learning_curves.png`, `schnet_on_outliers.csv`, `schnet_best.pt`, `dimenet_best.pt`
- `final_summary.json` and `tmqm_results.zip`

## Notes on reproducibility
A fixed seed (42) is used throughout. Tabular results are deterministic. GPU training of the geometry models is not bit-reproducible (non-deterministic scatter operations), and the paper therefore reports independent training replicates for every SchNet and DimeNet++ fold: in-domain scores are stable to within 0.01, out-of-domain scores are not — which is itself one of the paper's findings.

## Data and citation
tmQM (2024 release): Balcells, D.; Skjelstad, B. B. *J. Chem. Inf. Model.* **2020**, *60*, 6135–6146 — https://github.com/uiocompcat/tmQM
If you use this code, please cite the accompanying article (Journal of Computational Chemistry, in revision) and the tmQM dataset.

## License
MIT — see `LICENSE`.
