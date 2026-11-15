import pandas as pd, numpy as np, lightgbm as lgb, warnings, json, sys
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")
EPS=0.005
SEP=["sum_pr_z","sum_pr_ratio","sum_t_z","sum_soil_z","lrp","lrp2","dens2","dens2_rel","clim_pr","clim_wt","clim_snow","lat","lon"]
DEC=SEP+["aut_pr_z","aut_t_z","nov_snow_z","nov_snow","nov_t_z","aut_snowfall_z","aut_soil_z"]
JAN=DEC+["dec_snow_z","dec_snow","dec_t_z","dens","dens_rel"]
HIND=JAN+["win_t_z","win_tmin_z","win_snow_z","win_snow","win_snowfall_z","spr_t_z","spr_wind_z","spr_gust_z","spr_pr_z","spr_snow_z"]
SETS={"sep":SEP,"dec":DEC,"jan":JAN,"hind":HIND}
def prep(D):
    D=D.copy()
    for c in D.columns:
        if c.endswith("_z"): D[c]=D[c].clip(-4,4)
    D["lrp"]=np.log(D.rate_prev+EPS); D["lrp2"]=np.log(D.rate_prev2+EPS); D["y"]=np.log(D.rate+EPS)
    return D
def gbm(seed=0): return lgb.LGBMRegressor(n_estimators=250,learning_rate=0.03,num_leaves=5,min_child_samples=12,subsample=0.8,subsample_freq=1,colsample_bytree=0.6,reg_lambda=3,verbose=-1,random_state=seed)
def ridge(): return make_pipeline(StandardScaler(),RidgeCV(alphas=np.logspace(-1,3,20)))
def fitpred(kind,Xtr,ytr,Xte):
    if kind=="gbm": return np.mean([gbm(s).fit(Xtr,ytr).predict(Xte) for s in range(5)],axis=0)
    Xtr=Xtr.fillna(Xtr.median()); Xte=Xte.fillna(Xtr.median()); return ridge().fit(Xtr,ytr).predict(Xte)
def evaluate(D,cols,kind,mode,first=2000):
    out=[]
    for Y in sorted(D.year.unique()):
        if mode=="wf":
            if Y<first: continue
            tr=D[D.year<Y]
        else: tr=D[D.year!=Y]
        te=D[D.year==Y]
        p=fitpred(kind,tr[cols],tr.y,te[cols]); o=te[["code","year","rate","y"]].copy(); o["p"]=p; out.append(o)
    return pd.concat(out)
def score(o,thr=0.10):
    o=o.copy(); o["pr"]=np.exp(o.p)-EPS; hit=(o.rate>=thr).astype(int)
    wy=o.groupby("year").apply(lambda d: spearmanr(d.p,d.y)[0]).mean()
    yr=o.groupby("year").agg(a=("y","mean"),p=("p","mean")); 
    k=int(round(len(o)*0.15)); top=o.sort_values("p",ascending=False).head(k)
    return dict(n=len(o),spearman=spearmanr(o.p,o.y)[0],r2_log=1-((o.y-o.p)**2).sum()/((o.y-o.y.mean())**2).sum(),within_year=wy,year_corr=np.corrcoef(yr.a,yr.p)[0,1],auc=roc_auc_score(hit,o.p),disasters=int(hit.sum()),caught_top15=float((top.rate>=thr).sum()/hit.sum()),precision_top15=float((top.rate>=thr).mean()))
if __name__=="__main__":
    A=prep(pd.read_parquet("aimag_panel.parquet")); A=A[(A.code!="511")&(A.year>=1984)&A.rate.notna()&A.sum_pr_z.notna()&A.rate_prev2.notna()]
    print("rows",len(A),"years",A.year.min(),A.year.max(),"aimags",A.code.nunique(),"disasters>=10%:",(A.rate>=.1).sum())
    res=[]
    for mode in ("wf","loyo"):
        for name,cols in SETS.items():
            for kind in ("gbm","ridge"):
                o=evaluate(A,cols,kind,mode); s=score(o); s.update(mode=mode,info=name,model=kind); res.append(s)
                o.to_parquet(f"pred_{mode}_{name}_{kind}.parquet")
    R=pd.DataFrame(res).set_index(["mode","info","model"]).round(3); pd.set_option("display.width",220); print(R.to_string())
    # baselines (walk-forward): own historical mean, last year's rate
    B=A.sort_values(["code","year"]).copy(); B["p_clim"]=B.groupby("code").y.transform(lambda s:s.shift(1).expanding().mean()); B["p_pers"]=B.lrp
    for b in ("p_clim","p_pers"):
        o=B[B.year>=2000][["code","year","rate","y",b]].rename(columns={b:"p"}); print(b,{k:round(v,3) for k,v in score(o).items()})
