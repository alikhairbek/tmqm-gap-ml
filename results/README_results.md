# Results of the complete end-to-end run (tmQM.ipynb, PROFILE="full", NVIDIA T4, Kaggle, 10.25 h)

Every file here is produced by `tmQM.ipynb` (equivalently `tmqm.py`) from the public tmQM data, which is NOT redistributed
in this deposit: the notebook downloads it from https://github.com/uiocompcat/tmQM (Balcells & Skjelstad, JCIM 2020).

- Tabular models: model_comparison.csv (with 95% bootstrap CIs), indomain_repeated_splits.csv, xgb_sensitivity.csv,
  timing.json, outliers_xgb_gt2eV.csv, indomain_test_predictions.csv, fig_parity.png, fig_shap_beeswarm.png,
  shap_global.csv, shap_labels.csv
- Target statistics: gap_statistics.csv, figS_gap_hist.png, target_shift_loso.csv
- Transferability: transfer_loso.csv, transfer_lomo.csv (with CIs), loso_tabular_per_metal.csv, fig_transfer_loso.png,
  fig_transfer_lomo.png
- Conformal / applicability domain: conformal_*.csv, fig_conformal_*.png
- Geometry models: gnn_results.json (SchNet and DimeNet++: in-domain, LOSO 3d/4d/5d, LOMO-Zn, bootstrap CIs),
  predictions_*.csv, learning_curves.csv, fig_S2_learning_curves.png, schnet_on_outliers.csv,
  schnet_best.pt and dimenet_best.pt (PyTorch state_dicts of the in-domain models; architectures as in the notebook)
- final_summary.json

The paper reports independent training replicates for the geometry models; this folder contains the replicate produced
by the reproducibility run (SchNet replicate 4, DimeNet++ replicate 3). GPU training is not bit-reproducible.
