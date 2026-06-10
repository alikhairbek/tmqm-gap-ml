[README.md](https://github.com/user-attachments/files/28807245/README.md)
# Representation, Transferability, and Trustworthy Uncertainty in Machine-Learning Prediction of the HOMO–LUMO Gap of Transition-Metal Complexes

A fully reproducible, DFT-free machine-learning study built entirely on the public **tmQM** database (108,541 mononuclear transition-metal complexes, 3d/4d/5d). One notebook (or one script) reproduces every figure and table in the paper.

## What it does (three pillars)

1. **Representation hierarchy** - how much structural detail does the HOMO-LUMO gap need?
   composition + coordination -> + RDKit descriptors -> SchNet on 3D geometry
   (test R2 ~ 0.57 -> 0.63 -> 0.77; MAE 0.44 -> 0.30 eV), with SHAP interpretation.
2. **Transferability** - leave-one-series-out and leave-one-metal-out for both tabular and
   geometry models. In-domain accuracy collapses out-of-domain (median LOMO R2 ~ 0.34;
   d10 Zn < 0), and the geometry model does **not** extrapolate better.
3. **Trustworthy uncertainty** - normalized conformal prediction, series-conditional
   (Mondrian) calibration, and a distance-based applicability domain with selective
   prediction that restores calibrated coverage on novel chemistry.

## Repository contents

| File | Description |
|------|-------------|
| `tmQM.ipynb` | Complete end-to-end notebook (recommended; Colab/Kaggle-ready). |
| `tmqm.py`    | Equivalent single script (auto-generated from the notebook). |
| `requirements.txt` | Python dependencies. |
| `LICENSE`    | MIT. |

## Installation

```bash
pip install -r requirements.txt
```
`torch` is preinstalled on Google Colab and Kaggle GPU runtimes. A **GPU is required**
for the SchNet stages. The SchNet implementation uses a pure-PyTorch dense radius graph,
so no `torch_cluster` / `pyg-lib` compilation is needed.

## Run everything at once

**Notebook (recommended):** open `tmQM.ipynb`, enable a GPU runtime, and *Run all*.
Set `QUICK_TEST = True` in the first cell for a few-minute end-to-end check, then set it
back to `False` for the full run.

**Script:**
```bash
python tmqm.py
```
Edit the `CONFIG` block at the top of the file to toggle `QUICK_TEST` or adjust epochs.

## Runtime

A full GPU run trains ~5 networks (1 main SchNet + 3 leave-one-series-out + 1
leave-one-metal-out) and takes roughly **2-3 hours on an NVIDIA T4**. Kaggle GPU is
recommended for long, stable sessions. Lower `SCHNET_EPOCHS` / `TRANSFER_EPOCHS` to
trade accuracy for speed.

## Outputs

All results are written to `tmQM_data/results/` and bundled into `tmqm_results.zip`:

- `model_comparison.csv`, `shap_global.csv`, `fig_parity.png`, `fig_shap_beeswarm.png`
- `transfer_loso.csv`, `transfer_lomo.csv`, `fig_transfer_loso.png`, `fig_transfer_lomo.png`
- `schnet_metrics.json`, `schnet_transfer.json`
- conformal: `conformal_reliability.csv`, `conformal_applicability.csv`,
  `conformal_mondrian.csv`, `conformal_applicability_domain.csv`,
  `conformal_selective_curve.csv`
- figures: `fig_conformal_reliability.png`, `fig_conformal_applicability.png`,
  `fig_conformal_mondrian.png`, `fig_conformal_selective.png`

## Reproducibility

A fixed random seed (42) is used throughout. No new DFT or experiments are performed;
all model inputs are inexpensive, DFT-free quantities derived from the tmQM data.

## Data and citation

Data: the tmQM dataset (2024 release).
> Balcells, D.; Skjelstad, B. B. tmQM Dataset - Quantum Geometries and Properties of 86k
> Transition Metal Complexes. *J. Chem. Inf. Model.* **2020**, *60*, 6135-6146.
> Dataset repository: https://github.com/uiocompcat/tmQM

If you use this code, please also cite the accompanying paper (see the manuscript).

## License

MIT - see `LICENSE`.
