#!/usr/bin/env python3
"""
tmQM HOMO-LUMO gap: representation, transferability, trustworthy uncertainty -- complete pipeline.
Reproduces every table and figure of the paper and its revision in one run (GPU required):
features + fold-wise imputation -> target statistics -> tabular models, SHAP, bootstrap CIs, repeated splits,
sensitivity, outliers, timings -> (optional) TabPFN -> LOSO/LOMO with CIs -> conformal + domain-aware conformal
-> SchNet (learning curves, predictions, CIs) + transfer folds -> DimeNet++ (identical folds) -> Figure S2, summary, zip.
Usage:  pip install -r requirements.txt ; python tmqm.py        (edit the CONFIG block; QUICK=True for a smoke test)
Auto-generated from tmQM.ipynb; the two are equivalent. ~10 h on an NVIDIA T4 with PROFILE="full".
"""


# ============================== CONFIG ==============================
QUICK        = False        # True = ~10-min end-to-end smoke test on small subsets
PROFILE      = "full"       # "full" = paper settings (~10 h on a T4, one Kaggle session); "12h" = lighter DimeNet++ if your GPU is slower
USE_TABPFN   = False        # optional extra baseline (TabPFN); needs TABPFN_TOKEN (Prior Labs API) — not required for the paper
HF_SYNC      = False        # optional: archive/resume via Hugging Face; needs an HF token with WRITE role — not required
HF_REPO_NAME = "tmqm-gap-results"
SEED         = 42
MAXATOM      = 150
TIME_BUDGET_H = 11.5        # stop starting new GPU folds when the session would exceed this (Kaggle limit is 12 h)
SCHNET_EPOCHS, FOLD_EPOCHS = 40, 20                         # as in the paper
DIMENET = {"12h":  dict(num_blocks=2, cutoff=4.5, max_num_neighbors=16, batch=16, epochs=10),
           "full": dict(num_blocks=3, cutoff=5.0, max_num_neighbors=20, batch=8,  epochs=15)}[PROFILE]
HOLDOUT_METALS = ["Zn"]
QUICK_TEST = QUICK          # compatibility with reused cells
import time; T_START = time.time()
print(f"QUICK={QUICK} | PROFILE={PROFILE} | DimeNet++={DIMENET} | TabPFN={USE_TABPFN} | HF sync={HF_SYNC}")

# ============================== INSTALL =============================
pass  # [notebook shell command] !pip -q install torch_geometric rdkit pyarrow xgboost lightgbm catboost shap scikit-learn matplotlib scipy 2>&1 | tail -1
if USE_TABPFN:
    pass  # [notebook shell command] !pip -q install tabpfn tabpfn-client 2>&1 | tail -1
import torch, torch_geometric; print("torch", torch.__version__, "| PyG", torch_geometric.__version__, "| CUDA:", torch.cuda.is_available())

# ============================== SETUP ==============================
import os, sys, gzip, re, time, json, warnings, urllib.request, zipfile, io
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SchNet
from torch_geometric.nn.models.dimenet import DimeNetPlusPlus
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import shap
np.random.seed(SEED); torch.manual_seed(SEED)
WONG = {"3d": "#0072B2", "4d": "#E69F00", "5d": "#D55E00"}
DFT_COLS = ["HL_Gap_Ha","HL_Gap_eV","HOMO_Ha","LUMO_Ha","Dipole_D","Metal_q_DFT","Polarizability"]

def detect_data_dir():
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or os.path.isdir("/kaggle/working"): return "/kaggle/working/tmQM_data"   # Kaggle first: only /kaggle/working is saved as Output
    if "google.colab" in sys.modules: return "/content/tmQM_data"
    return os.path.join(os.getcwd(), "tmQM_data")
DATA_DIR = detect_data_dir(); os.makedirs(DATA_DIR, exist_ok=True)
OUT = os.path.join(DATA_DIR, "results"); os.makedirs(OUT, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("data dir:", DATA_DIR, "| device:", DEVICE, "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")

# ---- secrets (optional) ----
HF_TOKEN = TABPFN_TOKEN = None
try:
    from kaggle_secrets import UserSecretsClient
    _s = UserSecretsClient()
    for _lab in ["HF_TOKEN", "TABPFN_TOKEN"]:
        try: globals()[_lab] = _s.get_secret(_lab)
        except Exception: pass
except Exception: pass
HF_TOKEN = HF_TOKEN or os.environ.get("HF_TOKEN"); TABPFN_TOKEN = TABPFN_TOKEN or os.environ.get("TABPFN_TOKEN")
if HF_TOKEN: os.environ["HF_TOKEN"] = HF_TOKEN
print("HF token:", "yes" if HF_TOKEN else "no", "| TabPFN token:", "yes" if TABPFN_TOKEN else "no")

# ---- Hugging Face Hub: resume previous sessions / archive results (optional) ----
HF_REPO = None
def hf_sync_down():
    global HF_REPO
    if not (HF_SYNC and HF_TOKEN): return
    try:
        from huggingface_hub import HfApi, snapshot_download, create_repo
        api = HfApi(token=HF_TOKEN); user = api.whoami()["name"]; repo = f"{user}/{HF_REPO_NAME}"
        try: create_repo(repo, repo_type="dataset", private=True, exist_ok=True, token=HF_TOKEN)
        except Exception as e:
            print("HF: cannot create/access", repo, "-> the token needs WRITE permission (create a token with role 'Write', or a fine-grained token with write access to your repos). Sync disabled.", repr(e)[:90]); return
        HF_REPO = repo
        try: snapshot_download(HF_REPO, repo_type="dataset", local_dir=OUT, token=HF_TOKEN); print("HF resume: restored", len(os.listdir(OUT)), "files from", HF_REPO)
        except Exception as e: print("HF: repo ready, nothing to restore yet")
    except Exception as e: print("HF resume skipped:", repr(e)[:160])
def hf_sync_up(msg="update"):
    if not (HF_SYNC and HF_TOKEN and HF_REPO): return
    try:
        from huggingface_hub import HfApi
        HfApi(token=HF_TOKEN).upload_folder(folder_path=OUT, repo_id=HF_REPO, repo_type="dataset", commit_message=msg,
                                            ignore_patterns=["*.pt"] if PROFILE == "12h" else None)
        print("HF sync:", msg)
    except Exception as e: print("HF sync skipped:", repr(e)[:160])
hf_sync_down()

def elapsed_h(): return (time.time() - T_START) / 3600
def budget_ok(est_min):
    ok = elapsed_h() + est_min / 60 <= TIME_BUDGET_H
    if not ok: print(f"  time budget: {elapsed_h():.1f} h elapsed, next stage needs ~{est_min:.0f} min -> deferred to the next session")
    return ok

def lgbm(n=900):
    return LGBMRegressor(n_estimators=n, learning_rate=0.05, num_leaves=63, subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=SEED, verbose=-1)
def metrics(a, b):
    return {"R2": float(r2_score(a, b)), "MAE": float(mean_absolute_error(a, b)), "RMSE": float(np.sqrt(mean_squared_error(a, b)))}
def boot(t, p, B=1000, seed=0):
    B = 100 if QUICK else B
    rng = np.random.RandomState(seed); n = len(t); r2 = []; mae = []
    for _ in range(B):
        i = rng.randint(0, n, n); r2.append(r2_score(t[i], p[i])); mae.append(mean_absolute_error(t[i], p[i]))
    return dict(R2_lo=float(np.percentile(r2, 2.5)), R2_hi=float(np.percentile(r2, 97.5)), MAE_lo=float(np.percentile(mae, 2.5)), MAE_hi=float(np.percentile(mae, 97.5)))

PT = {"H":1,"B":5,"C":6,"N":7,"O":8,"F":9,"Si":14,"P":15,"S":16,"Cl":17,"As":33,"Se":34,"Br":35,"I":53,
      "Sc":21,"Ti":22,"V":23,"Cr":24,"Mn":25,"Fe":26,"Co":27,"Ni":28,"Cu":29,"Zn":30,"Y":39,"Zr":40,"Nb":41,
      "Mo":42,"Tc":43,"Ru":44,"Rh":45,"Pd":46,"Ag":47,"Cd":48,"La":57,"Hf":72,"Ta":73,"W":74,"Re":75,"Os":76,
      "Ir":77,"Pt":78,"Au":79,"Hg":80}
# pure-PyTorch graph ops -> no torch_cluster / torch_sparse needed for SchNet or DimeNet++
def radius_graph_dense(x, r, batch=None, loop=False, max_num_neighbors=32, flow="source_to_target", **kw):
    if batch is None: batch = x.new_zeros(x.size(0), dtype=torch.long)
    d = torch.cdist(x, x); same = batch[:, None] == batch[None, :]
    d = torch.where(same, d, torch.full_like(d, float("inf")))
    if not loop: d.fill_diagonal_(float("inf"))
    k = min(max_num_neighbors, x.size(0) - 1)
    val, idx = torch.topk(d, k, dim=1, largest=False); keep = val <= r
    row = torch.arange(x.size(0), device=x.device).unsqueeze(1).expand_as(idx)[keep]; col = idx[keep]
    return torch.stack([col, row], 0) if flow == "source_to_target" else torch.stack([row, col], 0)
def triplets_dense(edge_index, num_nodes):
    row, col = edge_index; E = row.size(0); dev = row.device; eid = torch.arange(E, device=dev)
    order = torch.argsort(col, stable=True); counts = torch.bincount(col, minlength=num_nodes); starts = torch.cumsum(counts, 0) - counts
    num_t = counts[row]; tot = int(num_t.sum())
    if tot == 0:
        z_ = eid[:0]; return col, row, z_, z_, z_, z_, z_
    idx_ji = eid.repeat_interleave(num_t); grp = torch.cumsum(num_t, 0) - num_t
    offs = torch.arange(tot, device=dev) - grp.repeat_interleave(num_t)
    idx_kj = order[starts[row].repeat_interleave(num_t) + offs]
    idx_k = row[idx_kj]; idx_j = row[idx_ji]; idx_i = col[idx_ji]; mask = idx_i != idx_k
    return col, row, idx_i[mask], idx_j[mask], idx_k[mask], idx_kj[mask], idx_ji[mask]
import torch_geometric.nn.models.schnet as _sch;  _sch.radius_graph = radius_graph_dense
import torch_geometric.nn.models.dimenet as _dim; _dim.radius_graph = radius_graph_dense; _dim.triplets = triplets_dense
print("setup done")

# ============================== DOWNLOAD tmQM ==============================
RAW = "https://raw.githubusercontent.com/uiocompcat/tmQM/master/tmQM"
FILES = ["tmQM_X1.xyz.gz", "tmQM_X2.xyz.gz", "tmQM_X3.xyz.gz", "tmQM_y.csv"]
for f in FILES:
    dst = os.path.join(DATA_DIR, f)
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        print("skip", f, f"({os.path.getsize(dst)/1e6:.1f} MB)"); continue
    for attempt in range(3):
        try:
            print("downloading", f, "...", end=" ", flush=True)
            urllib.request.urlretrieve(f"{RAW}/{f}", dst + ".part"); os.replace(dst + ".part", dst)
            print(f"{os.path.getsize(dst)/1e6:.1f} MB"); break
        except Exception as e:
            print("retry:", e); time.sleep(2 * (attempt + 1))
print("data ready")

_fp = os.path.join(DATA_DIR, "tmqm_features_plus.parquet")
if os.path.exists(_fp):
    print("features already built -> skip"); df = pd.read_parquet(_fp)
else:
    # ============================== FEATURES + RDKit ==============================
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors as rdd
    from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
    
    # metal -> (Z, group, period, series, Pauling EN, empirical radius pm)
    TM = {
     "Sc":(21,3,4,"3d",1.36,162),"Ti":(22,4,4,"3d",1.54,147),"V":(23,5,4,"3d",1.63,134),
     "Cr":(24,6,4,"3d",1.66,128),"Mn":(25,7,4,"3d",1.55,127),"Fe":(26,8,4,"3d",1.83,126),
     "Co":(27,9,4,"3d",1.88,125),"Ni":(28,10,4,"3d",1.91,124),"Cu":(29,11,4,"3d",1.90,128),
     "Zn":(30,12,4,"3d",1.65,134),"Y":(39,3,5,"4d",1.22,180),"Zr":(40,4,5,"4d",1.33,160),
     "Nb":(41,5,5,"4d",1.60,146),"Mo":(42,6,5,"4d",2.16,139),"Tc":(43,7,5,"4d",1.90,136),
     "Ru":(44,8,5,"4d",2.20,134),"Rh":(45,9,5,"4d",2.28,134),"Pd":(46,10,5,"4d",2.20,137),
     "Ag":(47,11,5,"4d",1.93,144),"Cd":(48,12,5,"4d",1.69,151),"La":(57,3,6,"5d",1.10,187),
     "Hf":(72,4,6,"5d",1.30,159),"Ta":(73,5,6,"5d",1.50,146),"W":(74,6,6,"5d",2.36,139),
     "Re":(75,7,6,"5d",1.90,137),"Os":(76,8,6,"5d",2.20,135),"Ir":(77,9,6,"5d",2.20,136),
     "Pt":(78,10,6,"5d",2.28,139),"Au":(79,11,6,"5d",2.54,144),"Hg":(80,12,6,"5d",2.00,151),
    }
    SERIES_ORD = {"3d":1,"4d":2,"5d":3}
    LIG = ["C","H","N","O","P","S","B","Si","As","Se","F","Cl","Br","I"]
    DONORS = ["N","O","P","S","As","Se"]; HAL = ["F","Cl","Br","I"]
    H2EV = 27.211386245988
    _re = re.compile(r"([A-Z][a-z]?)(\d*)")
    def formula(s): return {e:(int(n) if n else 1) for e,n in _re.findall(s) if e}
    
    def read_meta():
        rows = []
        for fn in ["tmQM_X1.xyz.gz","tmQM_X2.xyz.gz","tmQM_X3.xyz.gz"]:
            with gzip.open(os.path.join(DATA_DIR, fn), "rt") as fh:
                while True:
                    line = fh.readline()
                    if not line: break
                    s = line.strip()
                    if s == "": continue
                    n = int(s); c = fh.readline()
                    for _ in range(n): fh.readline()
                    d = {}
                    for p in c.split("|"):
                        if "=" in p: k,v = p.split("=",1); d[k.strip()] = v.strip()
                    rows.append({"CSD_code":d.get("CSD_code"),"charge":int(d.get("q",0)),
                                 "MND":int(d.get("MND",0)),"n_atoms":n,"Stoich":d.get("Stoichiometry","")})
        return pd.DataFrame(rows)
    
    meta = read_meta()
    ycsv = pd.read_csv(os.path.join(DATA_DIR,"tmQM_y.csv"), sep=";")
    raw = meta.merge(ycsv, on="CSD_code", how="inner")
    print("merged:", raw.shape)
    
    def build_row(r):
        comp = formula(r.Stoich); metal = next((e for e in comp if e in TM), None)
        if metal is None: return None
        Z,g,per,ser,en,rad = TM[metal]; lig = {e:comp.get(e,0) for e in LIG}; nH = lig["H"]
        row = {"CSD_code":r.CSD_code,"metal":metal,"metal_Z":Z,"metal_group":g,"metal_period":per,
               "metal_series":ser,"metal_series_ord":SERIES_ORD[ser],"metal_EN":en,"metal_radius_pm":rad,
               "n_metal_atoms":comp.get(metal,1),"MND":r.MND,"charge":r.charge,"n_atoms":r.n_atoms,
               "n_heavy":r.n_atoms-nH,"n_halogen":sum(lig[e] for e in HAL),
               "n_donor":sum(lig[e] for e in DONORS),"HL_Gap_eV":r.HL_Gap*H2EV,"HOMO_Ha":r.HOMO_Energy,
               "LUMO_Ha":r.LUMO_Energy,"Dipole_D":r.Dipole_M,"Metal_q_DFT":r.Metal_q,
               "Polarizability":r.Polarizability,
               "SMILES": r.SMILES if isinstance(r.SMILES,str) else ""}
        for e in LIG: row[f"n_{e}"] = lig[e]
        return row
    
    df = pd.DataFrame([x for x in (build_row(r) for r in raw.itertuples(index=False)) if x])
    print("base features:", df.shape)
    
    DCOLS = ["rd_MolWt","rd_nAromRing","rd_nAliphRing","rd_nRing","rd_FracCSP3","rd_nHBA",
             "rd_nHBD","rd_nRotB","rd_TPSA","rd_nHetero","rd_nAromHetero","rd_MolMR"]
    def desc(s):
        m = Chem.MolFromSmiles(s) if s else None
        if m is None: return [np.nan]*12
        return [Descriptors.MolWt(m),rdd.CalcNumAromaticRings(m),rdd.CalcNumAliphaticRings(m),
                rdd.CalcNumRings(m),rdd.CalcFractionCSP3(m),rdd.CalcNumHBA(m),rdd.CalcNumHBD(m),
                rdd.CalcNumRotatableBonds(m),rdd.CalcTPSA(m),rdd.CalcNumHeteroatoms(m),
                rdd.CalcNumAromaticHeterocycles(m),Descriptors.MolMR(m)]
    t = time.time()
    D = pd.DataFrame(df["SMILES"].apply(desc).tolist(), columns=DCOLS)
    df = pd.concat([df.drop(columns=["SMILES"]), D], axis=1)
    print(f"RDKit descriptors in {time.time()-t:.0f}s | full-SMILES rows: {df[DCOLS].notna().all(1).sum():,}")
    
    if QUICK_TEST:
        df = df.sample(min(6000, len(df)), random_state=SEED).reset_index(drop=True)
        print("QUICK_TEST subset:", df.shape)
    
    df.to_parquet(os.path.join(DATA_DIR, "tmqm_features_plus.parquet"), index=False)
    DFT = ["HL_Gap_eV","HOMO_Ha","LUMO_Ha","Dipole_D","Metal_q_DFT","Polarizability"]
    FEATURES = [c for c in df.columns if c not in (["CSD_code","metal","metal_series"]+DFT)]
    print("total features:", len(FEATURES))
    print("series balance:\n", df["metal_series"].value_counts().to_string())
FEATURES = [c for c in df.columns if c not in (["CSD_code","metal","metal_series"] + DFT_COLS)]
print("features:", len(FEATURES), "| complexes:", len(df), "| missing-RDKit rows:", int(df[[c for c in FEATURES if c.startswith("rd_")]].isna().any(axis=1).sum()))

# ============================== TARGET STATISTICS (Table S2, Figure S1, target shift) ==============================
if QUICK and len(df) > 12000: df = df.sample(12000, random_state=SEED).reset_index(drop=True)
Xraw = df[FEATURES].astype(float).values; yv = df["HL_Gap_eV"].values
metal = df["metal"].values; series = df["metal_series"].values; idx = np.arange(len(df))
def XI(train_idx):
    """fold-wise imputation: median computed on the training rows only, applied to all rows"""
    med = np.nanmedian(Xraw[train_idx], axis=0); return np.where(np.isnan(Xraw), med, Xraw)
def stats(v): return dict(n=int(len(v)), mean=float(np.mean(v)), median=float(np.median(v)), std=float(np.std(v)), min=float(np.min(v)), q25=float(np.percentile(v,25)), q75=float(np.percentile(v,75)), max=float(np.max(v)))
S = {"all": stats(yv)}; [S.__setitem__(s, stats(yv[series==s])) for s in ["3d","4d","5d"]]
pd.DataFrame(S).T.to_csv(os.path.join(OUT, "gap_statistics.csv")); print(pd.DataFrame(S).T.round(3).to_string())
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
ax[0].hist(yv, bins=80, color="#4a4a4a", alpha=0.85); ax[0].set_xlabel("HOMO–LUMO gap (eV)"); ax[0].set_ylabel("count"); ax[0].set_title(f"All complexes (n = {len(yv):,})")
for s in ["3d","4d","5d"]: ax[1].hist(yv[series==s], bins=80, histtype="step", lw=1.8, color=WONG[s], label=f"{s} (n={int((series==s).sum()):,})", density=True)
ax[1].set_xlabel("HOMO–LUMO gap (eV)"); ax[1].set_ylabel("density"); ax[1].set_title("By metal series"); ax[1].legend(fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "figS_gap_hist.png"), dpi=300); plt.close(fig)
pd.DataFrame([dict(series=s, mean_gap=float(yv[series==s].mean()), std_gap=float(yv[series==s].std()), mean_gap_training_other_two=float(yv[series!=s].mean()), shift=float(yv[series==s].mean()-yv[series!=s].mean())) for s in ["3d","4d","5d"]]).to_csv(os.path.join(OUT, "target_shift_loso.csv"), index=False)
print("target statistics saved")

# ============================== TABULAR MODELS + SHAP + CIs + REPEATED SPLITS + SENSITIVITY + OUTLIERS ==============================
n_est = 200 if QUICK else 900
tr, tmp = train_test_split(idx, test_size=0.30, random_state=SEED, stratify=metal)
va, te = train_test_split(tmp, test_size=0.50, random_state=SEED, stratify=metal[tmp])
X = XI(tr)   # imputed with the training-partition median (fold-wise)
print(f"train {len(tr):,} | val {len(va):,} | test {len(te):,}")
MODELS = {
 "Linear":("lin", LinearRegression()),
 "RandomForest":("tree", RandomForestRegressor(n_estimators=min(300, n_est), n_jobs=-1, random_state=SEED)),
 "XGBoost":("tree", XGBRegressor(n_estimators=n_est, learning_rate=0.05, max_depth=7, subsample=0.8, colsample_bytree=0.8, tree_method="hist", n_jobs=-1, random_state=SEED)),
 "LightGBM":("tree", lgbm(n_est)),
 "CatBoost":("tree", CatBoostRegressor(iterations=n_est, learning_rate=0.05, depth=7, random_seed=SEED, verbose=0)),
}
sc = StandardScaler().fit(X[tr]); res = {}; fitt = {}; preds = {}; timing = {}
for name,(k,m) in MODELS.items():
    Xtr = sc.transform(X[tr]) if k=="lin" else X[tr]; Xte = sc.transform(X[te]) if k=="lin" else X[te]
    t0 = time.time(); m.fit(Xtr, yv[tr]); tf = time.time()-t0; t0 = time.time(); p = m.predict(Xte); tp = time.time()-t0
    res[name] = {**metrics(yv[te], p), **boot(yv[te], p)}; fitt[name] = m; preds[name] = p; timing[name] = dict(fit_s=tf, predict_ms_per_complex=tp/len(te)*1000)
    print(f"  {name:13s} R2={res[name]['R2']:.4f} [{res[name]['R2_lo']:.3f},{res[name]['R2_hi']:.3f}]  MAE={res[name]['MAE']:.4f}  RMSE={res[name]['RMSE']:.4f}  (fit {tf:.0f}s)")
pd.DataFrame(res).T.to_csv(os.path.join(OUT, "model_comparison.csv"))
pd.DataFrame({"CSD_code": df.CSD_code.values[te], "metal": metal[te], "series": series[te], "y": yv[te], **{f"pred_{k}": v for k, v in preds.items()}}).to_csv(os.path.join(OUT, "indomain_test_predictions.csv"), index=False)
prim = max(["LightGBM","XGBoost","CatBoost"], key=lambda n: res[n]["R2"]); pm = fitt[prim]; pred = preds[prim]; print("primary (interpreted) model:", prim)
# parity (Figure 1)
fig, ax = plt.subplots(figsize=(5.2, 5.2))
for s in ["3d","4d","5d"]:
    msk = series[te]==s; ax.scatter(yv[te][msk], pred[msk], s=4, alpha=0.25, c=WONG[s], label=s)
lim = [float(min(yv[te].min(), pred.min())), float(max(yv[te].max(), pred.max()))]
ax.plot(lim, lim, "k--", lw=1); ax.set_xlabel("DFT gap (eV)"); ax.set_ylabel("Predicted (eV)")
ax.set_title(f"{prim}: R²={res[prim]['R2']:.3f}, MAE={res[prim]['MAE']:.3f} eV"); ax.legend(markerscale=3)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_parity.png"), dpi=300); plt.close(fig)
# SHAP with human-readable labels (Figure 2, Table S5)
LABEL = {"metal_Z":"Metal atomic number","metal_group":"Metal group","metal_period":"Metal period","metal_series_ord":"Metal series (3d/4d/5d)","metal_EN":"Metal electronegativity","metal_radius_pm":"Metal radius (pm)","MND":"Coordination number (MND)","charge":"Total charge","n_atoms":"Number of atoms","n_heavy":"Heavy-atom count","n_metal_atoms":"Metal-atom count","n_donor":"Donor-atom count","n_halogen":"Halogen count","rd_MolWt":"Molecular weight","rd_nAromRing":"Aromatic rings","rd_nAliphRing":"Aliphatic rings","rd_nRing":"Ring count","rd_FracCSP3":"sp³ carbon fraction","rd_nHBA":"H-bond acceptors","rd_nHBD":"H-bond donors","rd_nRotB":"Rotatable bonds","rd_TPSA":"Polar surface area (TPSA)","rd_nHetero":"Heteroatom count","rd_nAromHetero":"Aromatic heterocycles","rd_MolMR":"Molar refractivity"}
lab = lambda c: LABEL.get(c, f"{c[2:]} count" if c.startswith("n_") else c)
samp = np.random.RandomState(SEED).choice(te, min(4000, len(te)), replace=False)
sv = shap.TreeExplainer(pm).shap_values(X[samp])
glob = pd.Series(np.abs(sv).mean(0), index=FEATURES).sort_values(ascending=False); glob.to_csv(os.path.join(OUT, "shap_global.csv"))
pd.DataFrame({"code": FEATURES, "label": [lab(c) for c in FEATURES], "mean_abs_shap": np.abs(sv).mean(0)}).sort_values("mean_abs_shap", ascending=False).to_csv(os.path.join(OUT, "shap_labels.csv"), index=False)
plt.figure(figsize=(8, 7.5)); shap.summary_plot(sv, X[samp], feature_names=[lab(c) for c in FEATURES], show=False, max_display=18)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_shap_beeswarm.png"), dpi=300, bbox_inches="tight"); plt.close()
print("top SHAP drivers:", list(glob.head(8).index))
# repeated splits (Table S10)
rep = []
for sd in ([1, 2] if QUICK else [1, 2, 3, 4, 5]):
    a, b = train_test_split(idx, test_size=0.30, random_state=sd, stratify=metal); _, c = train_test_split(b, test_size=0.50, random_state=sd, stratify=metal[b]); Xs = XI(a)
    for name, mk in [("XGBoost", lambda: XGBRegressor(n_estimators=n_est, learning_rate=0.05, max_depth=7, subsample=0.8, colsample_bytree=0.8, tree_method="hist", n_jobs=-1, random_state=SEED)), ("LightGBM", lambda: lgbm(n_est))]:
        rep.append(dict(seed=sd, model=name, **metrics(yv[c], mk().fit(Xs[a], yv[a]).predict(Xs[c]))))
rep = pd.DataFrame(rep); rep.to_csv(os.path.join(OUT, "indomain_repeated_splits.csv"), index=False)
print("repeated splits:\n", rep.groupby("model")[["R2","MAE"]].agg(["mean","std"]).round(3).to_string())
# hyperparameter sensitivity (Table S8)
sens = [dict(n_estimators=n, **metrics(yv[te], XGBRegressor(n_estimators=n, learning_rate=0.05, max_depth=7, subsample=0.8, colsample_bytree=0.8, tree_method="hist", n_jobs=-1, random_state=SEED).fit(X[tr], yv[tr]).predict(X[te]))) for n in ([100, 300] if QUICK else [300, 900, 1500])]
pd.DataFrame(sens).to_csv(os.path.join(OUT, "xgb_sensitivity.csv"), index=False); print("XGB tree sweep:", [(d["n_estimators"], round(d["R2"], 3)) for d in sens])
# outliers |error| > 2 eV (Table S14)
p = preds["XGBoost"]; err = p - yv[te]; mask = np.abs(err) > 2.0
out = pd.DataFrame({"CSD_code": df.CSD_code.values[te][mask], "metal": metal[te][mask], "series": series[te][mask], "DFT_gap": yv[te][mask], "pred": p[mask], "error": err[mask], "charge": df.charge.values[te][mask], "n_atoms": df.n_atoms.values[te][mask], "MND": df.MND.values[te][mask]})
out.to_csv(os.path.join(OUT, "outliers_xgb_gt2eV.csv"), index=False); OUTLIER_CODES = out.CSD_code.tolist()
print(f"outliers |err|>2 eV: {mask.sum()} ({mask.mean()*100:.2f}%), over-estimated {int((err[mask]>0).sum())}, mean DFT gap {yv[te][mask].mean():.2f} eV, frac <1.5 eV {(yv[te][mask]<1.5).mean()*100:.0f}%")
# timings (Table S9): RDKit descriptor cost
try:
    from rdkit import Chem; from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen
    smi = pd.read_csv(os.path.join(DATA_DIR, "tmQM_y.csv"), sep=";")["SMILES"].dropna().astype(str).head(1500).tolist(); t0 = time.time(); k = 0
    for s_ in smi:
        m_ = Chem.MolFromSmiles(s_)
        if m_ is None: continue
        Descriptors.MolWt(m_); rdMolDescriptors.CalcNumAromaticRings(m_); rdMolDescriptors.CalcFractionCSP3(m_); rdMolDescriptors.CalcTPSA(m_); Crippen.MolMR(m_); k += 1
    timing["rdkit_ms_per_complex"] = (time.time()-t0)/max(k,1)*1000
except Exception as e: print("rdkit timing skipped:", e)
json.dump(timing, open(os.path.join(OUT, "timing.json"), "w"), indent=1); print("tabular stage done")

# ============================== TabPFN (optional): tabular foundation model, no hyperparameters ==============================
# Fair comparison on a 10,000-complex training subset (TabPFN's context limit) against XGBoost on the SAME subset, evaluated on the
# full held-out test set: a tuning-free reference for the tabular ceiling (reviewer 2, minor 6). Local weights first; on any failure
# (e.g. the one-time licence prompt, which cannot be answered in a notebook) the Prior Labs API is used with TABPFN_TOKEN.
if USE_TABPFN:
    n_sub = min(10000, len(tr)); sub = train_test_split(tr, train_size=n_sub, random_state=SEED, stratify=metal[tr])[0] if len(tr) > n_sub else tr
    Xs = XI(sub); reg = None; backend = None
    def _local():
        from tabpfn import TabPFNRegressor
        r = TabPFNRegressor(device=DEVICE.type); r.fit(Xs[sub], yv[sub]); return r, "tabpfn (local, GPU)"
    def _client():
        import tabpfn_client
        if TABPFN_TOKEN:
            os.environ["TABPFN_ACCESS_TOKEN"] = TABPFN_TOKEN
            for fn in ("set_access_token",):
                if hasattr(tabpfn_client, fn): getattr(tabpfn_client, fn)(TABPFN_TOKEN)
        from tabpfn_client import TabPFNRegressor
        r = TabPFNRegressor(); r.fit(Xs[sub], yv[sub]); return r, "tabpfn-client (Prior Labs API)"
    for attempt in (_local, _client):
        try: reg, backend = attempt(); break
        except Exception as e: print(f"  {attempt.__name__} unavailable: {repr(e)[:150]}")
    if reg is not None:
        try:
            t0 = time.time(); tp = [reg.predict(Xs[te][i:i+2000]) for i in range(0, len(te), 2000)]
            p_t = np.concatenate(tp); r_t = {**metrics(yv[te], p_t), **boot(yv[te], p_t)}
            xg = XGBRegressor(n_estimators=n_est, learning_rate=0.05, max_depth=7, subsample=0.8, colsample_bytree=0.8, tree_method="hist", n_jobs=-1, random_state=SEED).fit(Xs[sub], yv[sub])
            p_x = xg.predict(Xs[te]); r_x = {**metrics(yv[te], p_x), **boot(yv[te], p_x)}
            comp = {"backend": backend, "n_train_subset": int(len(sub)), "TabPFN": r_t, "XGBoost_same_subset": r_x, "XGBoost_full_train": {k: res["XGBoost"][k] for k in ["R2","MAE","RMSE"]}, "predict_seconds": time.time()-t0}
            json.dump(comp, open(os.path.join(OUT, "tabpfn_comparison.json"), "w"), indent=1)
            print(f"TabPFN [{backend}] ({len(sub):,} train): R2={r_t['R2']:.3f} [{r_t['R2_lo']:.3f},{r_t['R2_hi']:.3f}] MAE={r_t['MAE']:.3f} | XGBoost same subset: R2={r_x['R2']:.3f} MAE={r_x['MAE']:.3f} | XGBoost full train: R2={res['XGBoost']['R2']:.3f}")
        except Exception as e: print("TabPFN prediction failed:", repr(e)[:200])
    else: print("TabPFN skipped (no local weights and no working API token)")
else: print("USE_TABPFN = False -> skipped")

# ============================== TRANSFERABILITY (tabular): LOSO + LOMO with CIs, per-metal, target shift ==============================
def score(tr_mask, te_mask, n=900):
    tr_i = np.where(tr_mask)[0]; te_i = np.where(te_mask)[0]; Xf = XI(tr_i)
    m = lgbm(n).fit(Xf[tr_i], yv[tr_i]); p = m.predict(Xf[te_i])
    return {**metrics(yv[te_i], p), **boot(yv[te_i], p, B=600), "n_test": int(len(te_i))}, p, te_i
ref = res[prim]; print(f"[in-domain] R2={ref['R2']:.4f} MAE={ref['MAE']:.4f} eV")
loso = {}; pm_rows = []
for s in ["3d","4d","5d"]:
    loso[s], p, te_i = score(series!=s, series==s, n=n_est)
    print(f"  LOSO {s}: R2={loso[s]['R2']:+.3f} [{loso[s]['R2_lo']:+.3f},{loso[s]['R2_hi']:+.3f}] MAE={loso[s]['MAE']:.3f} eV (n={loso[s]['n_test']:,})")
    for m_ in np.unique(metal[te_i]):
        k = metal[te_i]==m_; pm_rows.append(dict(held_out_series=s, metal=m_, n=int(k.sum()), MAE=float(mean_absolute_error(yv[te_i][k], p[k])), R2=float(r2_score(yv[te_i][k], p[k])) if k.sum()>2 else np.nan))
pd.DataFrame(loso).T.to_csv(os.path.join(OUT, "transfer_loso.csv")); pd.DataFrame(pm_rows).to_csv(os.path.join(OUT, "loso_tabular_per_metal.csv"), index=False)
lomo = {}
metals_run = sorted(pd.unique(metal)) if not QUICK else ["Zn", "Fe", "Pt"]
for mm in metals_run:
    if (metal==mm).sum() < 5: continue
    lomo[mm], _, _ = score(metal!=mm, metal==mm, n=min(400, n_est))
    lomo[mm]["gap_std"] = float(yv[metal==mm].std()); lomo[mm]["shift_vs_rest"] = float(yv[metal==mm].mean() - yv[metal!=mm].mean())
lomo_df = pd.DataFrame(lomo).T
lomo_df["series"] = [df.loc[df.metal==mm, "metal_series"].iloc[0] for mm in lomo_df.index]; lomo_df["Z"] = [df.loc[df.metal==mm, "metal_Z"].iloc[0] for mm in lomo_df.index]
lomo_df = lomo_df.sort_values("Z"); lomo_df.to_csv(os.path.join(OUT, "transfer_lomo.csv"))
print(f"  LOMO median R2={lomo_df['R2'].median():.3f}  median MAE={lomo_df['MAE'].median():.3f} eV | hardest: {list(lomo_df.sort_values('R2').head(3).index)}")
if len(lomo_df) > 5:
    from scipy.stats import spearmanr
    print(f"  Spearman(|mean shift|, R2) = {spearmanr(np.abs(lomo_df.shift_vs_rest), lomo_df.R2)[0]:.2f};  Spearman(within-metal SD, R2) = {spearmanr(lomo_df.gap_std, lomo_df.R2)[0]:.2f}")
fig, ax = plt.subplots(figsize=(5.2, 4)); xs = list(loso)
ax.bar(xs, [loso[s]["R2"] for s in xs], color=[WONG[s] for s in xs], yerr=[[loso[s]["R2"]-loso[s]["R2_lo"] for s in xs], [loso[s]["R2_hi"]-loso[s]["R2"] for s in xs]], capsize=4)
ax.axhline(ref["R2"], ls="--", c="k", lw=1, label=f"in-domain {ref['R2']:.2f}"); ax.axhline(0, c="0.4", lw=.8)
ax.set_ylabel("test R² (held-out series)"); ax.set_title("Leave-one-series-out"); ax.legend(); fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_transfer_loso.png"), dpi=300); plt.close(fig)
fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(range(len(lomo_df)), lomo_df["R2"].values, color=[WONG[s] for s in lomo_df["series"]], yerr=[lomo_df["R2"]-lomo_df["R2_lo"], lomo_df["R2_hi"]-lomo_df["R2"]], capsize=2, error_kw=dict(lw=0.8))
ax.axhline(ref["R2"], ls="--", c="k", lw=1); ax.axhline(0, c="0.4", lw=.8)
ax.set_xticks(range(len(lomo_df))); ax.set_xticklabels(lomo_df.index, rotation=90, fontsize=7); ax.set_ylabel("test R² (held-out metal)"); ax.set_title("Leave-one-metal-out (95% bootstrap intervals)")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_transfer_lomo.png"), dpi=300); plt.close(fig); print("transferability figures saved")

# ============================== UNCERTAINTY (conformal) ==============================
ALPHA = 0.10; SIGMA_FLOOR = 0.02

def fit_conformal(i_mu, i_sig, i_cal):
    mu = lgbm(n_est).fit(X[i_mu], yv[i_mu])
    r = np.abs(yv[i_sig] - mu.predict(X[i_sig]))
    sigma = lgbm(n_est).fit(X[i_sig], r)
    s = np.abs(yv[i_cal]-mu.predict(X[i_cal])) / np.clip(sigma.predict(X[i_cal]), SIGMA_FLOOR, None)
    n = len(i_cal); k = min(int(np.ceil((n+1)*(1-ALPHA))), n)
    return mu, sigma, np.sort(s)[k-1]

def evaluate(mu, sigma, q, i_te):
    sig = np.clip(sigma.predict(X[i_te]), SIGMA_FLOOR, None); half = q*sig
    err = np.abs(yv[i_te]-mu.predict(X[i_te]))
    return float(np.mean(err<=half)), float(np.mean(2*half)), sig, err

trv, t_te = train_test_split(idx, test_size=0.20, random_state=SEED, stratify=series)
i_mu, rest = train_test_split(trv, test_size=0.40, random_state=SEED, stratify=series[trv])
i_sig, i_cal = train_test_split(rest, test_size=0.50, random_state=SEED, stratify=series[rest])
X = XI(i_mu)   # fold-wise imputation: median of the mu-training subset
mu, sigma, q = fit_conformal(i_mu, i_sig, i_cal)
cov, wid, sig_te, err_te = evaluate(mu, sigma, q, t_te)
print(f"[in-domain] target {1-ALPHA:.0%} -> empirical coverage {cov:.1%} | mean width {wid:.3f} eV")

order = np.argsort(sig_te); rel = []
print("[reliability] uncertainty quintile -> actual MAE:")
for j, b in enumerate(np.array_split(order, 5), 1):
    rel.append({"quintile":j,"pred_sigma":float(sig_te[b].mean()),"MAE":float(err_te[b].mean())})
    print(f"  Q{j}: pred σ={sig_te[b].mean():.3f}  MAE={err_te[b].mean():.3f} eV")
pd.DataFrame(rel).to_csv(os.path.join(OUT,"conformal_reliability.csv"), index=False)

is5 = series=="5d"
if is5.sum()>20 and (~is5).sum()>20:
    te5 = t_te[series[t_te]=="5d"]
    covA, widA, sigA, errA = evaluate(mu, sigma, q, te5)
    src = np.where(~is5)[0]
    b_mu, b_rest = train_test_split(src, test_size=0.40, random_state=SEED, stratify=series[src])
    b_sig, b_cal = train_test_split(b_rest, test_size=0.50, random_state=SEED, stratify=series[b_rest])
    X = XI(b_mu)
    muB, sigmaB, qB = fit_conformal(b_mu, b_sig, b_cal)
    covB, widB, sigB, errB = evaluate(muB, sigmaB, qB, np.where(is5)[0])
    print("[applicability domain] same 5d chemistry:")
    print(f"  5d KNOWN : coverage {covA:.1%}  width {widA:.3f}  MAE {errA.mean():.3f}")
    print(f"  5d UNSEEN: coverage {covB:.1%}  width {widB:.3f}  MAE {errB.mean():.3f}")
    pd.DataFrame([{"regime":"5d_known","coverage":covA,"width":widA,"MAE":float(errA.mean())},
                  {"regime":"5d_unseen","coverage":covB,"width":widB,"MAE":float(errB.mean())}]
                 ).to_csv(os.path.join(OUT,"conformal_applicability.csv"), index=False)

    fig, ax = plt.subplots(1,2, figsize=(8,3.6))
    ax[0].bar(["5d known","5d unseen"], [covA,covB], color=["#0072B2","#D55E00"])
    ax[0].axhline(1-ALPHA, ls="--", c="k", lw=1, label=f"target {1-ALPHA:.0%}")
    ax[0].set_ylabel("coverage"); ax[0].set_ylim(0,1); ax[0].legend()
    ax[1].bar(["5d known","5d unseen"], [widA,widB], color=["#0072B2","#D55E00"])
    ax[1].set_ylabel("mean interval width (eV)")
    fig.suptitle("Applicability domain: 5d known vs unseen"); fig.tight_layout()
    fig.savefig(os.path.join(OUT,"fig_conformal_applicability.png"), dpi=300); plt.close(fig)

fig, ax = plt.subplots(figsize=(5,4)); qn=[r["quintile"] for r in rel]
ax.plot(qn,[r["pred_sigma"] for r in rel],"o-",color="#0072B2",label="predicted σ")
ax.plot(qn,[r["MAE"] for r in rel],"s--",color="#D55E00",label="actual MAE")
ax.set_xlabel("predicted-uncertainty quintile"); ax.set_ylabel("eV"); ax.set_xticks(qn)
ax.set_title("Uncertainty is informative"); ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(OUT,"fig_conformal_reliability.png"), dpi=300); plt.close(fig)
print("conformal figures saved")

# ============================== DOMAIN-AWARE CONFORMAL ==============================
from sklearn.neighbors import NearestNeighbors
ALPHA = 0.10; SIGMA_FLOOR = 0.02; KNN = 10
yc = yv

def _ms(i_mu, i_sig):
    mu = lgbm(n_est).fit(X[i_mu], yc[i_mu])
    sg = lgbm(n_est).fit(X[i_sig], np.abs(yc[i_sig] - mu.predict(X[i_sig])))
    return mu, sg
def _nc(mu, sg, i):
    return np.abs(yc[i] - mu.predict(X[i])) / np.clip(sg.predict(X[i]), SIGMA_FLOOR, None)
def _q(s):
    n = len(s); k = min(int(np.ceil((n + 1) * (1 - ALPHA))), n); return np.sort(s)[k - 1]

# (A) Mondrian (series-conditional) vs global, in-domain
trv, teA = train_test_split(idx, test_size=0.20, random_state=SEED, stratify=series)
imu, rest = train_test_split(trv, test_size=0.40, random_state=SEED, stratify=series[trv])
isig, ical = train_test_split(rest, test_size=0.50, random_state=SEED, stratify=series[rest])
X = XI(imu)   # fold-wise imputation
muA, sgA = _ms(imu, isig); scal = _nc(muA, sgA, ical); qg = _q(scal)
qs = {s: _q(scal[series[ical] == s]) for s in ["3d", "4d", "5d"]}
sigA = np.clip(sgA.predict(X[teA]), SIGMA_FLOOR, None); errA = np.abs(yc[teA] - muA.predict(X[teA]))
mrows = []
for s in ["3d", "4d", "5d"]:
    m = series[teA] == s
    mrows.append({"series": s, "coverage_global": float(np.mean(errA[m] <= qg * sigA[m])),
                  "coverage_mondrian": float(np.mean(errA[m] <= qs[s] * sigA[m]))})
mond = pd.DataFrame(mrows); mond.to_csv(os.path.join(OUT, "conformal_mondrian.csv"), index=False)
print("Mondrian per-series coverage (target 90%):"); print(mond.to_string(index=False))

# (B) Applicability domain + selective prediction on a mixed pool (in-domain + unseen 5d)
is5 = series == "5d"; src = np.where(~is5)[0]
btr, bhold = train_test_split(src, test_size=0.30, random_state=SEED, stratify=series[src])
bmu, bsig = train_test_split(btr, test_size=0.40, random_state=SEED, stratify=series[btr])
bcal, btein = train_test_split(bhold, test_size=0.50, random_state=SEED, stratify=series[bhold])
X = XI(bmu)
ood = np.where(is5)[0]
muB, sgB = _ms(bmu, bsig); qB = _q(_nc(muB, sgB, bcal))
scaler2 = StandardScaler().fit(X[bmu]); nn = NearestNeighbors(n_neighbors=KNN).fit(scaler2.transform(X[bmu]))
adcal = nn.kneighbors(scaler2.transform(X[bcal]))[0].mean(1); adthr = float(np.quantile(adcal, 0.95))
pool = np.concatenate([btein, ood]); isood = np.concatenate([np.zeros(len(btein), bool), np.ones(len(ood), bool)])
adp = nn.kneighbors(scaler2.transform(X[pool]))[0].mean(1)
sigP = np.clip(sgB.predict(X[pool]), SIGMA_FLOOR, None); cov = np.abs(yc[pool] - muB.predict(X[pool])) <= qB * sigP
acc = adp <= adthr
covfull = float(cov.mean()); covacc = float(cov[acc].mean()) if acc.sum() else float("nan")
print(f"AD rejects {1-acc.mean():.0%} of pool ({(~acc & isood).sum()/isood.sum():.0%} of unseen-5d, {(~acc & ~isood).sum()/(~isood).sum():.0%} of in-domain)")
print(f"coverage: full pool={covfull:.1%}  accepted={covacc:.1%}  (retain {acc.mean():.0%})")
curve = []
for p in np.linspace(0.5, 1.0, 26):
    a = adp <= np.quantile(adcal, p)
    if a.sum() < 30: continue
    curve.append({"retention": float(a.mean()), "coverage": float(cov[a].mean())})
curve = pd.DataFrame(curve).sort_values("retention"); curve.to_csv(os.path.join(OUT, "conformal_selective_curve.csv"), index=False)
pd.DataFrame([{"coverage_full": covfull, "coverage_accepted": covacc, "retention": float(acc.mean()),
               "reject_ood": float((~acc & isood).sum()/isood.sum()),
               "reject_indomain": float((~acc & ~isood).sum()/(~isood).sum())}]
             ).to_csv(os.path.join(OUT, "conformal_applicability_domain.csv"), index=False)
fig, ax = plt.subplots(figsize=(5.4, 4)); xs = np.arange(3); w = 0.36
ax.bar(xs - w/2, mond["coverage_global"], w, label="global", color="#999999")
ax.bar(xs + w/2, mond["coverage_mondrian"], w, label="Mondrian", color="#0072B2")
ax.axhline(1 - ALPHA, ls="--", c="k", lw=1, label="target 90%"); ax.set_xticks(xs); ax.set_xticklabels(["3d", "4d", "5d"])
ax.set_ylim(0.7, 1.0); ax.set_ylabel("coverage"); ax.set_title("(A) Series-conditional conformal"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_conformal_mondrian.png"), dpi=300); plt.close(fig)
fig, ax = plt.subplots(figsize=(5.4, 4))
ax.plot(curve["retention"], curve["coverage"], "o-", color="#D55E00", ms=4, label="selective coverage")
ax.axhline(1 - ALPHA, ls="--", c="k", lw=1, label="target 90%")
ax.scatter([acc.mean()], [covacc], s=90, color="#0072B2", zorder=5, label="AD cutoff")
ax.set_xlabel("retention"); ax.set_ylabel("coverage on accepted"); ax.set_title("(B) AD selective prediction"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_conformal_selective.png"), dpi=300); plt.close(fig)
print("domain-aware conformal done")

# ============================== LOAD GRAPHS (SchNet / DimeNet++) ==============================
hf_sync_up("tabular + conformal results")
label_map = {r.CSD_code: (float(r.HL_Gap_eV), r.metal_series, r.metal) for r in df.itertuples(index=False)}
limit = 3000 if QUICK else None; maxatom = 100 if QUICK else MAXATOM
def load_graphs():
    G = []
    for fn in ["tmQM_X1.xyz.gz", "tmQM_X2.xyz.gz", "tmQM_X3.xyz.gz"]:
        with gzip.open(os.path.join(DATA_DIR, fn), "rt") as fh:
            while True:
                line = fh.readline()
                if not line: break
                s = line.strip()
                if s == "": continue
                n = int(s); c = fh.readline()
                code = next(p.split("=")[1].strip() for p in c.split("|") if p.strip().startswith("CSD_code"))
                z, pos = [], []
                for _ in range(n):
                    pr = fh.readline().split(); z.append(PT[pr[0]]); pos.append([float(v) for v in pr[1:4]])
                if code not in label_map or n > maxatom: continue
                g, ser, met = label_map[code]
                G.append(Data(z=torch.tensor(z, dtype=torch.long), pos=torch.tensor(pos, dtype=torch.float), y=torch.tensor([g], dtype=torch.float), series=ser, metal=met, code=code))
                if limit and len(G) >= limit: return G
    return G
t0 = time.time(); graphs = load_graphs()
gser = np.array([g.series for g in graphs]); gmet = np.array([g.metal for g in graphs]); gi = np.arange(len(graphs))
gtrv, gte = train_test_split(gi, test_size=0.15, random_state=SEED, stratify=gser)
gtr, gva = train_test_split(gtrv, test_size=0.15/0.85, random_state=SEED, stratify=gser[gtrv])
print(f"loaded {len(graphs):,} graphs in {time.time()-t0:.0f}s | split {len(gtr):,}/{len(gva):,}/{len(gte):,} | elapsed {elapsed_h():.1f} h")

# ============================== GNN HELPERS (training, resume-safe, bootstrap) ==============================
def make_model(arch, interactions=5):
    if arch == "schnet":
        return SchNet(hidden_channels=128, num_filters=128, num_interactions=3 if QUICK else interactions, num_gaussians=50, cutoff=8.0)
    return DimeNetPlusPlus(hidden_channels=64, out_channels=1, num_blocks=2 if QUICK else DIMENET["num_blocks"], int_emb_size=32, basis_emb_size=4,
                           out_emb_channels=64, num_spherical=5, num_radial=5, cutoff=DIMENET["cutoff"], max_num_neighbors=DIMENET["max_num_neighbors"])
def run_epoch(model, ld, ym, ys, opt=None):
    train = opt is not None; model.train() if train else model.eval(); P, T, Cc = [], [], []
    for b in ld:
        b = b.to(DEVICE); yt = (b.y - ym) / ys
        with torch.set_grad_enabled(train):
            out = model(b.z, b.pos, b.batch).view(-1); loss = torch.nn.functional.l1_loss(out, yt)
            if train: opt.zero_grad(); loss.backward(); opt.step()
        P.append(out.detach().cpu() * ys + ym); T.append(b.y.detach().cpu()); Cc += list(b.code)
    p = torch.cat(P).numpy(); t = torch.cat(T).numpy(); return float(r2_score(t, p)), float(mean_absolute_error(t, p)), p, t, Cc
RESULTS_FILE = os.path.join(OUT, "gnn_results.json"); CURVES_FILE = os.path.join(OUT, "learning_curves.csv")
results = json.load(open(RESULTS_FILE)) if os.path.exists(RESULTS_FILE) else {}
COMPLETED = {}   # repository version: train everything from scratch (resume works through the saved files in OUT)
for _k, _v in COMPLETED.items(): results.setdefault(_k, _v)
curves = pd.read_csv(CURVES_FILE).to_dict("records") if os.path.exists(CURVES_FILE) else []
def save_partial(): json.dump(results, open(RESULTS_FILE, "w"), indent=2); pd.DataFrame(curves).to_csv(CURVES_FILE, index=False)
LAST = None
def train_eval(arch, tr_ix, va_ix, te_ix, epochs, tag, interactions=5, est_min=60):
    global LAST
    key = f"{arch}_{tag}"; pfile = os.path.join(OUT, f"predictions_{key}.csv")
    if key in results: print(f"  [{key}] already done -> skip"); return results[key]
    if not budget_ok(est_min): return None
    ys_ = torch.tensor([graphs[i].y.item() for i in tr_ix]); ym, ystd = ys_.mean(), ys_.std()
    bs = 32 if arch == "schnet" else DIMENET["batch"]
    trL = DataLoader([graphs[i] for i in tr_ix], batch_size=bs, shuffle=True); vaL = DataLoader([graphs[i] for i in va_ix], batch_size=bs) if len(va_ix) else None; teL = DataLoader([graphs[i] for i in te_ix], batch_size=bs)
    model = make_model(arch, interactions).to(DEVICE); opt = torch.optim.Adam(model.parameters(), lr=5e-4); best = (1e9, None); t0 = time.time()
    for ep in range(1, epochs + 1):
        _, tr_mae, *_ = run_epoch(model, trL, ym, ystd, opt); va_mae = float("nan")
        if vaL is not None:
            _, va_mae, *_ = run_epoch(model, vaL, ym, ystd)
            if va_mae < best[0]: best = (va_mae, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        curves.append(dict(tag=tag, arch=arch, epoch=ep, train_MAE=tr_mae, val_MAE=va_mae)); print(f"    [{key}] epoch {ep:3d} train MAE={tr_mae:.4f} val MAE={va_mae:.4f} ({(time.time()-t0)/60:.1f} min)")
    if best[1] is not None: model.load_state_dict(best[1])
    LAST = dict(model=model, ym=ym, ys=ystd, arch=arch, tag=tag)
    r2, mae, p, t, codes = run_epoch(model, teL, ym, ystd)
    pd.DataFrame({"CSD_code": codes, "y": t, "pred": p}).to_csv(pfile, index=False)
    if tag == "indomain": torch.save(model.state_dict(), os.path.join(OUT, f"{arch}_best.pt"))
    results[key] = dict(R2=r2, MAE=mae, n_test=int(len(te_ix)), epochs=epochs, minutes=(time.time()-t0)/60, **boot(t, p)); save_partial(); hf_sync_up(f"{key} done")
    print(f"  [{key}] TEST R2={r2:+.3f} [{results[key]['R2_lo']:+.3f},{results[key]['R2_hi']:+.3f}] MAE={mae:.3f} eV ({(time.time()-t0)/60:.0f} min)")
    return results[key]
print("GNN helpers ready")

# ============================== SchNet: main model (curves, predictions, CIs, per-series, outliers) + transfer folds ==============================
ep_main = 3 if QUICK else SCHNET_EPOCHS; ep_fold = 2 if QUICK else FOLD_EPOCHS
train_eval("schnet", gtr, gva, gte, ep_main, "indomain", interactions=5, est_min=75)
if os.path.exists(os.path.join(OUT, "predictions_schnet_indomain.csv")):
    pr = pd.read_csv(os.path.join(OUT, "predictions_schnet_indomain.csv"))
    for s in ["3d","4d","5d"]:
        m = gser[gte]==s
        if m.sum() > 5: results[f"schnet_indomain_{s}"] = dict(R2=float(r2_score(pr.y[m], pr.pred[m])), MAE=float(mean_absolute_error(pr.y[m], pr.pred[m])), n_test=int(m.sum()))
    # SchNet on the Figure-1 outlier set: (a) held-out in the SchNet split, (b) all outliers with the trained model
    pri = pr.set_index("CSD_code"); held = [c for c in OUTLIER_CODES if c in pri.index]
    if held:
        e = (pri.loc[held, "pred"] - pri.loc[held, "y"]).values
        results["schnet_on_xgb_outliers_heldout"] = dict(n=int(len(held)), MAE=float(np.abs(e).mean()), mean_signed_error=float(e.mean()), frac_abs_err_gt2=float((np.abs(e)>2).mean()))
    if LAST is not None and LAST["arch"] == "schnet" and LAST["tag"] == "indomain":
        code2i = {g.code: i for i, g in enumerate(graphs)}; oi = [code2i[c] for c in OUTLIER_CODES if c in code2i]
        if oi:
            trset = set(gtr.tolist()) | set(gva.tolist()); _, _, p_o, t_o, c_o = run_epoch(LAST["model"], DataLoader([graphs[i] for i in oi], batch_size=32), LAST["ym"], LAST["ys"]); e = p_o - t_o
            results["schnet_on_xgb_outliers_all"] = dict(n=int(len(oi)), n_in_schnet_training=int(sum(i in trset for i in oi)), MAE=float(np.abs(e).mean()), mean_signed_error=float(e.mean()), frac_abs_err_gt2=float((np.abs(e)>2).mean()))
            pd.DataFrame({"CSD_code": c_o, "y": t_o, "schnet_pred": p_o, "in_schnet_training": [i in trset for i in oi]}).to_csv(os.path.join(OUT, "schnet_on_outliers.csv"), index=False)
    print("SchNet in-domain:", {k: v for k, v in results.items() if k.startswith("schnet_indomain") or "outliers" in k}); save_partial()
for s in ["3d","4d","5d"]: train_eval("schnet", gi[gser!=s], [], gi[gser==s], ep_fold, f"LOSO-{s}", interactions=4, est_min=35)
for m_ in HOLDOUT_METALS: train_eval("schnet", gi[gmet!=m_], [], gi[gmet==m_], ep_fold, f"LOMO-{m_}", interactions=4, est_min=45)
save_partial(); print(f"SchNet stage done | elapsed {elapsed_h():.1f} h")

# ============================== DimeNet++: second architecture, identical folds (profile-dependent cost) ==============================
ep_d = 2 if QUICK else DIMENET["epochs"]; est = {"12h": 55, "full": 100}[PROFILE]
try:
    _m = make_model("dimenet").to(DEVICE); _b = next(iter(DataLoader(graphs[:4], batch_size=4))).to(DEVICE)
    with torch.no_grad(): _o = _m(_b.z, _b.pos, _b.batch)
    print("DimeNet++ smoke test OK ->", tuple(_o.shape)); del _m, _b, _o
    train_eval("dimenet", gtr, gva, gte, ep_d, "indomain", est_min=est)
    for s in ["3d","4d","5d"]: train_eval("dimenet", gi[gser!=s], [], gi[gser==s], ep_d, f"LOSO-{s}", est_min=est)
    for m_ in HOLDOUT_METALS: train_eval("dimenet", gi[gmet!=m_], [], gi[gmet==m_], ep_d, f"LOMO-{m_}", est_min=est*1.2)
except Exception as e: print("DimeNet++ stage error:", repr(e)[:300])
save_partial(); print(f"DimeNet++ stage done | elapsed {elapsed_h():.1f} h")

# ============================== FIGURE S2 (learning curves) + FINAL SUMMARY ==============================
cdf = pd.DataFrame(curves); Cc = {"LOSO-3d":"#0072B2","LOSO-4d":"#E69F00","LOSO-5d":"#D55E00","LOMO-Zn":"#117733"}
if len(cdf):
    fig, ax = plt.subplots(2, 2, figsize=(10.5, 7.2))
    for r_, arch in enumerate(["schnet", "dimenet"]):
        d0 = cdf[cdf.arch==arch]; d = d0[d0.tag=="indomain"]
        if len(d):
            ax[r_,0].plot(d.epoch, d.train_MAE, "-", lw=1.6, color="#0072B2", label="train"); ax[r_,0].plot(d.epoch, d.val_MAE, "--o", ms=3, lw=1.4, color="#D55E00", label="validation")
            ax[r_,0].set_title(f"{'SchNet' if arch=='schnet' else 'DimeNet++'}, in-domain"); ax[r_,0].legend(fontsize=9)
        for tg, c in Cc.items():
            q = d0[d0.tag==tg]
            if len(q): ax[r_,1].plot(q.epoch, q.train_MAE, "-", lw=1.6, color=c, label=tg)
        ax[r_,1].set_title(f"{'SchNet' if arch=='schnet' else 'DimeNet++'} transfer folds (training MAE)"); ax[r_,1].legend(fontsize=8)
        for a in ax[r_]: a.set_xlabel("epoch"); a.set_ylabel("MAE (standardized units)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_S2_learning_curves.png"), dpi=300); plt.close(fig)
summary = {"profile": PROFILE, "elapsed_hours": round(elapsed_h(), 2), "tabular": {k: {m: round(v[m], 4) for m in ["R2","MAE","RMSE","R2_lo","R2_hi"]} for k, v in res.items()},
           "loso_tabular": {k: {m: round(v[m], 4) for m in ["R2","MAE","R2_lo","R2_hi"]} for k, v in loso.items()},
           "lomo_median_R2": round(float(lomo_df["R2"].median()), 4), "gnn": results}
json.dump(summary, open(os.path.join(OUT, "final_summary.json"), "w"), indent=2)
print(f"=== SUMMARY (elapsed {elapsed_h():.1f} h) ===")
for k, v in results.items():
    if "R2_lo" in v: print(f"  {k:26s} R2={v['R2']:+.3f} [{v['R2_lo']:+.3f},{v['R2_hi']:+.3f}]  MAE={v['MAE']:.3f}")
pending = [k for k in ["schnet_indomain","schnet_LOSO-3d","schnet_LOSO-4d","schnet_LOSO-5d","schnet_LOMO-Zn","dimenet_indomain","dimenet_LOSO-3d","dimenet_LOSO-4d","dimenet_LOSO-5d","dimenet_LOMO-Zn"] if k not in results]
print("pending folds (re-run the notebook to continue; HF resume keeps finished ones):", pending or "none")

# ============================== ZIP + ARCHIVE ==============================
hf_sync_up("final results")
zip_path = os.path.join(os.path.dirname(DATA_DIR), "tmqm_results.zip")   # Kaggle: /kaggle/working/tmqm_results.zip (Output tab, top level)
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for fn in sorted(os.listdir(OUT)):
        if not fn.endswith(".pt"): z.write(os.path.join(OUT, fn), arcname=fn)
print("zipped ->", zip_path, f"({os.path.getsize(zip_path)/1e6:.1f} MB) | total elapsed {elapsed_h():.2f} h")
try:
    from google.colab import files; files.download(zip_path)
except Exception: print("Kaggle: tmqm_results.zip is at the top level of the Output tab after 'Save & Run All (Commit)'; models *.pt stay in", OUT)
