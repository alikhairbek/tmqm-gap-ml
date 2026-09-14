#!/usr/bin/env python3
"""
tmQM revision — CPU analyses added in response to the reviewers.
Run from the repository root (expects tmQM_data/tmqm_features_plus.parquet, tmQM_y.csv, results/*.csv).
Produces, in tmQM_data/results/revision/:
  gap_statistics.csv, figS_gap_hist.png, target_shift_loso.csv, target_shift_lomo.csv   (Table S2, Fig S1, Table S11)
  indomain_*.json, indomain_repeated_splits.csv, indomain_test_predictions.csv          (Table S10)
  outliers_xgb_gt2eV.csv                                                                (Table S14)
  loso_tabular_ci.csv, loso_tabular_per_metal.csv, lomo_tabular_ci.csv                 (Tables S15-S17)
  xgb_sensitivity.csv, timing.json                                                      (Tables S8-S9)
  fig_shap_beeswarm_relabeled.png, shap_labels.csv                                      (Figure 2, Table S5)
All imputation is fold-wise (training-partition median only). ~1 h on a single CPU core.
"""
import os, sys, time, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from scipy.stats import spearmanr
DATA_DIR = os.environ.get("TMQM_DATA", "tmQM_data"); RES = os.path.join(DATA_DIR, "results"); OUT = os.path.join(RES, "revision"); os.makedirs(OUT, exist_ok=True)

SEED = 42
df = pd.read_parquet(os.path.join(DATA_DIR, "tmqm_features_plus.parquet"))
DFT = ["HL_Gap_Ha","HL_Gap_eV","HOMO_Ha","LUMO_Ha","Dipole_D","Metal_q_DFT","Polarizability"]
FEATURES = [c for c in df.columns if c not in (["CSD_code","metal","metal_series"]+DFT)]
y = df["HL_Gap_eV"].values; series = df["metal_series"].values; metal = df["metal"].values
idx = np.arange(len(df)); Xraw = df[FEATURES].astype(float).values
def impute(Xtr, Xte):
    med = np.nanmedian(Xtr, axis=0); return np.where(np.isnan(Xtr), med, Xtr), np.where(np.isnan(Xte), med, Xte)
def xgb(n=900): return XGBRegressor(n_estimators=n, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=SEED, verbosity=0)
def lgb(n=700): return LGBMRegressor(n_estimators=n, learning_rate=0.05, num_leaves=63, subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=SEED, verbose=-1)
def rf(n=200):  return RandomForestRegressor(n_estimators=n, n_jobs=-1, random_state=SEED)
def metrics(t,p): return dict(R2=float(r2_score(t,p)), MAE=float(mean_absolute_error(t,p)), RMSE=float(np.sqrt(mean_squared_error(t,p))))
def boot(t,p,B=1000,seed=0):
    rng=np.random.RandomState(seed); n=len(t); r2=[]; mae=[]
    for _ in range(B):
        i=rng.randint(0,n,n); r2.append(r2_score(t[i],p[i])); mae.append(mean_absolute_error(t[i],p[i]))
    return dict(R2_lo=float(np.percentile(r2,2.5)), R2_hi=float(np.percentile(r2,97.5)), MAE_lo=float(np.percentile(mae,2.5)), MAE_hi=float(np.percentile(mae,97.5)))
def split(seed):
    trv, te = train_test_split(idx, test_size=0.15, random_state=seed, stratify=series)
    tr, va = train_test_split(trv, test_size=0.15/0.85, random_state=seed, stratify=series[trv]); return tr, va, te

# A. target statistics, histograms, target shift
def stats(v): return dict(n=int(len(v)), mean=float(np.mean(v)), median=float(np.median(v)), std=float(np.std(v)), min=float(np.min(v)), q25=float(np.percentile(v,25)), q75=float(np.percentile(v,75)), max=float(np.max(v)))
S={"all":stats(y)}; [S.__setitem__(s, stats(y[series==s])) for s in ["3d","4d","5d"]]
pd.DataFrame(S).T.to_csv(f"{OUT}/gap_statistics.csv")
WONG={"3d":"#0072B2","4d":"#E69F00","5d":"#D55E00"}; fig,ax=plt.subplots(1,2,figsize=(9,3.6))
ax[0].hist(y,bins=80,color="#4a4a4a",alpha=0.85); ax[0].set_xlabel("HOMO\u2013LUMO gap (eV)"); ax[0].set_ylabel("count"); ax[0].set_title(f"All complexes (n = {len(y):,})")
for s in ["3d","4d","5d"]: ax[1].hist(y[series==s],bins=80,histtype="step",lw=1.8,color=WONG[s],label=f"{s} (n={int((series==s).sum()):,})",density=True)
ax[1].set_xlabel("HOMO\u2013LUMO gap (eV)"); ax[1].set_ylabel("density"); ax[1].set_title("By metal series"); ax[1].legend(fontsize=9); fig.tight_layout(); fig.savefig(f"{OUT}/figS_gap_hist.png",dpi=300); plt.close(fig)
lomo_old=pd.read_csv(f"{RES}/transfer_lomo.csv",index_col=0); rows=[]
for m in lomo_old.index:
    mm=metal==m; rows.append(dict(metal=m,series=lomo_old.loc[m,"series"],n=int(mm.sum()),mean_gap=float(y[mm].mean()),std_gap=float(y[mm].std()),shift_vs_rest=float(y[mm].mean()-y[~mm].mean()),R2_lomo=float(lomo_old.loc[m,"R2"]),MAE_lomo=float(lomo_old.loc[m,"MAE"])))
shift=pd.DataFrame(rows); shift.to_csv(f"{OUT}/target_shift_lomo.csv",index=False)
print("LOMO Spearman(|shift|,R2)=%.2f  Spearman(std,R2)=%.2f" % (spearmanr(np.abs(shift.shift_vs_rest),shift.R2_lomo)[0], spearmanr(shift.std_gap,shift.R2_lomo)[0]))
pd.DataFrame([dict(series=s,mean_gap=float(y[series==s].mean()),std_gap=float(y[series==s].std()),mean_gap_training_other_two=float(y[series!=s].mean()),shift=float(y[series==s].mean()-y[series!=s].mean())) for s in ["3d","4d","5d"]]).to_csv(f"{OUT}/target_shift_loso.csv",index=False)

# B. in-domain CIs, repeated splits, sensitivity, SHAP relabel, outliers
tr,va,te=split(SEED); Xtr,Xte=impute(Xraw[tr],Xraw[te]); preds={}; timing={}
for name,mk in [("XGBoost",xgb),("LightGBM",lgb),("RandomForest",rf)]:
    t0=time.time(); m=mk().fit(Xtr,y[tr]); timing[f"{name}_fit_s"]=time.time()-t0; p=m.predict(Xte); preds[name]=p
    json.dump(dict(**metrics(y[te],p),**boot(y[te],p)),open(f"{OUT}/indomain_{name}.json","w"),indent=1)
pd.DataFrame({"CSD_code":df.CSD_code.values[te],"metal":metal[te],"series":series[te],"y":y[te],**{f"pred_{k}":v for k,v in preds.items()}}).to_csv(f"{OUT}/indomain_test_predictions.csv",index=False)
rep=[]
for sd in [1,2,3,4,5]:
    tr_,va_,te_=split(sd); Xa,Xb=impute(Xraw[tr_],Xraw[te_])
    for name,mk in [("XGBoost",xgb),("LightGBM",lgb)]: rep.append(dict(seed=sd,model=name,**metrics(y[te_],mk().fit(Xa,y[tr_]).predict(Xb))))
pd.DataFrame(rep).to_csv(f"{OUT}/indomain_repeated_splits.csv",index=False)
pd.DataFrame([dict(n_estimators=n,**metrics(y[te],xgb(n).fit(Xtr,y[tr]).predict(Xte))) for n in [300,900,1500]]).to_csv(f"{OUT}/xgb_sensitivity.csv",index=False)
p=preds["XGBoost"]; err=p-y[te]; mask=np.abs(err)>2.0
pd.DataFrame({"CSD_code":df.CSD_code.values[te][mask],"metal":metal[te][mask],"series":series[te][mask],"DFT_gap":y[te][mask],"pred":p[mask],"error":err[mask],"charge":df.charge.values[te][mask],"n_atoms":df.n_atoms.values[te][mask],"MND":df.MND.values[te][mask]}).to_csv(f"{OUT}/outliers_xgb_gt2eV.csv",index=False)
try:
    from rdkit import Chem; from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen
    smi=pd.read_csv(os.path.join(DATA_DIR,"tmQM_y.csv"),sep=";")["SMILES"].dropna().astype(str).head(1500).tolist(); t0=time.time(); k=0
    for s_ in smi:
        m_=Chem.MolFromSmiles(s_)
        if m_ is None: continue
        Descriptors.MolWt(m_); rdMolDescriptors.CalcNumAromaticRings(m_); rdMolDescriptors.CalcFractionCSP3(m_); rdMolDescriptors.CalcTPSA(m_); Crippen.MolMR(m_); k+=1
    timing["rdkit_ms_per_complex"]=(time.time()-t0)/max(k,1)*1000
except Exception as e: print("rdkit timing skipped:",e)
json.dump(timing,open(f"{OUT}/timing.json","w"),indent=1)
import shap
LABEL={"metal_Z":"Metal atomic number","metal_group":"Metal group","metal_period":"Metal period","metal_series_ord":"Metal series (3d/4d/5d)","metal_EN":"Metal electronegativity","metal_radius_pm":"Metal radius (pm)","MND":"Coordination number (MND)","charge":"Total charge","n_atoms":"Number of atoms","n_heavy":"Heavy-atom count","n_metal_atoms":"Metal-atom count","n_donor":"Donor-atom count","n_halogen":"Halogen count","rd_MolWt":"Molecular weight","rd_nAromRing":"Aromatic rings","rd_nAliphRing":"Aliphatic rings","rd_nRing":"Ring count","rd_FracCSP3":"sp\u00b3 carbon fraction","rd_nHBA":"H-bond acceptors","rd_nHBD":"H-bond donors","rd_nRotB":"Rotatable bonds","rd_TPSA":"Polar surface area (TPSA)","rd_nHetero":"Heteroatom count","rd_nAromHetero":"Aromatic heterocycles","rd_MolMR":"Molar refractivity"}
lab=lambda c: LABEL.get(c, f"{c[2:]} count" if c.startswith("n_") else c)
m=lgb().fit(Xtr,y[tr]); sub=np.random.RandomState(0).choice(len(te),3000,replace=False); sv=shap.TreeExplainer(m).shap_values(Xte[sub])
plt.figure(figsize=(8,7.5)); shap.summary_plot(sv,Xte[sub],feature_names=[lab(c) for c in FEATURES],max_display=18,show=False); plt.tight_layout(); plt.savefig(f"{OUT}/fig_shap_beeswarm_relabeled.png",dpi=300,bbox_inches="tight"); plt.close()
pd.DataFrame({"code":FEATURES,"label":[lab(c) for c in FEATURES],"mean_abs_shap":np.abs(sv).mean(0)}).sort_values("mean_abs_shap",ascending=False).to_csv(f"{OUT}/shap_labels.csv",index=False)

# C. LOSO / LOMO with bootstrap CIs (fold-wise imputation)
loso=[]; pm=[]
for s in ["3d","4d","5d"]:
    tr_=idx[series!=s]; te_=idx[series==s]; Xa,Xb=impute(Xraw[tr_],Xraw[te_]); pp=lgb().fit(Xa,y[tr_]).predict(Xb)
    loso.append(dict(series=s,n_test=int(len(te_)),**metrics(y[te_],pp),**boot(y[te_],pp)))
    for m_ in np.unique(metal[te_]):
        k=metal[te_]==m_; pm.append(dict(held_out_series=s,metal=m_,n=int(k.sum()),MAE=float(mean_absolute_error(y[te_][k],pp[k])),R2=float(r2_score(y[te_][k],pp[k]))))
pd.DataFrame(loso).to_csv(f"{OUT}/loso_tabular_ci.csv",index=False); pd.DataFrame(pm).to_csv(f"{OUT}/loso_tabular_per_metal.csv",index=False)
rows=[]
for m_ in lomo_old.sort_values("Z").index:
    tr_=idx[metal!=m_]; te_=idx[metal==m_]; Xa,Xb=impute(Xraw[tr_],Xraw[te_]); pp=lgb().fit(Xa,y[tr_]).predict(Xb)
    rows.append(dict(metal=m_,series=str(series[te_][0]),Z=int(lomo_old.loc[m_,"Z"]),n=int(len(te_)),gap_std=float(y[te_].std()),**metrics(y[te_],pp),**boot(y[te_],pp,B=600)))
    pd.DataFrame(rows).to_csv(f"{OUT}/lomo_tabular_ci.csv",index=False)
print("revision analyses complete ->", OUT)
