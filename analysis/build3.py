import pandas as pd, numpy as np, glob, warnings; warnings.filterwarnings("ignore")
fr=[]
for f in glob.glob("raw/ghcn/MG*.csv.gz"):
    d=pd.read_csv(f,header=None,usecols=[0,1,2,3,5],names=["id","date","el","v","q"],dtype={"q":str}); fr.append(d[d.el.isin(["TAVG","TMIN","TMAX","SNWD"])&d.q.isna()&(d.date>=19930601)])
D=pd.concat(fr); D["date"]=pd.to_datetime(D.date.astype(str),format="%Y%m%d")
W=D.pivot_table(index=["id","date"],columns="el",values="v",aggfunc="first").reset_index()
for c in ("TAVG","TMIN","TMAX","SNWD"): W[c]=W[c]/10.0
W["t"]=W.TAVG.fillna((W.TMIN+W.TMAX)/2); W=W[W.t.notna()]; W["snow"]=W.SNWD.fillna(0)
W["Y"]=np.where(W.date.dt.month>=6,W.date.dt.year+1,W.date.dt.year); W["m"]=W.date.dt.month
W["tmax"]=W.TMAX.fillna(W.t+5)
cold=W[W.m.isin([10,11,12,1,2,3,4])]
def f(g):
    nd=g[g.m.isin([11,12])]; core=g[g.m.isin([11,12,1,2,3])]; djf=g[g.m.isin([12,1,2])]
    return pd.Series({"n":len(core),"d_snowdays":(core.snow>=1).sum()/max(len(core),1)*151,"d_deepdays":(core.snow>=10).sum()/max(len(core),1)*151,
      "d_early_snowdays":(g[g.m.isin([10,11])].snow>=1).sum(),"d_nd_snowdays":(nd.snow>=1).sum(),"d_nd_deep":(nd.snow>=8).sum(),
      "d_thaw_on_snow":((core.tmax>0)&(core.snow>=1)).sum(),"d_nd_thaw_on_snow":((nd.tmax>0)&(nd.snow>=1)).sum(),
      "d_cold35":(djf.TMIN<-35).sum(),"d_cold7":djf.t.rolling(7).mean().min() if len(djf)>20 else np.nan,"d_snowmax":core.snow.max(),"d_spr_snowdays":(g[g.m.isin([3,4])].snow>=1).sum()})
F=cold.groupby(["id","Y"]).apply(f).reset_index(); F=F[F.n>=120]
cols=[c for c in F.columns if c.startswith("d_")]
b=F[(F.Y>=1994)&(F.Y<=2007)].groupby("id")[cols]; mu,sd=b.mean(),b.std()
Z=F[["id","Y"]].copy()
for c in cols: Z[c+"_z"]=((F[c]-F.id.map(mu[c]))/F.id.map(sd[c]).clip(lower=1.0)).clip(-5,5); Z[c]=F[c]
st=pd.read_csv("stations.csv").set_index("id")
S=pd.read_parquet("soum_panel2.parquet"); A=pd.read_parquet("aimag_panel2.parquet")
so=S.drop_duplicates("pcode").set_index("pcode")[["lat","lon","acode"]]
def km(lat1,lon1,lat2,lon2):
    p=np.pi/180; a=np.sin((lat2-lat1)*p/2)**2+np.cos(lat1*p)*np.cos(lat2*p)*np.sin((lon2-lon1)*p/2)**2; return 12742*np.arcsin(np.sqrt(a))
Dm=pd.DataFrame({i:km(so.lat.values,so.lon.values,st.lat[i],st.lon[i]) for i in st.index},index=so.index)
vals=[c for c in Z.columns if c not in ("id","Y")]; out=[]
for Y,z in Z.groupby("Y"):
    z=z.set_index("id"); res=pd.DataFrame(index=so.index,columns=vals,dtype=float)
    for v in vals:
        ok=z[v].dropna()
        if len(ok)<15: continue
        d=Dm[ok.index].values; idx=np.argsort(d,axis=1)[:,:4]; dd=np.take_along_axis(d,idx,1); w=1/(dd+20)**2; res[v]=(ok.values[idx]*w).sum(1)/w.sum(1)
    res["year"]=Y; out.append(res.reset_index())
SS=pd.concat(out); S=S.merge(SS,on=["pcode","year"],how="left")
w=S.groupby("pcode").herd.mean().fillna(0)+1; SS["w"]=SS.pcode.map(w).fillna(1); SS["code"]=SS.pcode.map(so.acode)
AA=SS.dropna(subset=["code"]).groupby(["code","year"]).apply(lambda d: pd.Series({v:np.average(d[v],weights=d.w) if d[v].notna().all() else np.nan for v in vals})).reset_index()
A=A.merge(AA,on=["code","year"],how="left"); S.to_parquet("soum_panel3.parquet"); A.to_parquet("aimag_panel3.parquet")
a=A[(A.year>=1994)&(A.code!="511")].copy(); a["y"]=np.log(a.rate+0.005)
print(a[[c for c in a.columns if c.startswith("d_")]+["y"]].corr(method="spearman").y.sort_values().round(2).to_string())
print(a.groupby("year")[["rate","d_nd_snowdays_z","d_snowdays_z","d_deepdays_z","d_thaw_on_snow_z","d_cold35_z","sum_pr_z"]].mean().round(2).T.to_string())
# three-ingredient table
a["dry"]=a.sum_pr_z<-0.5; a["snowy"]=a.d_snowdays_z>0.5; a["coldw"]=a.s_win_t_z<-0.5; a["dis"]=a.rate>=0.10
print("base rate",a.dis.mean().round(3),len(a))
print(a.groupby(["dry","snowy","coldw"]).agg(n=("dis","size"),p=("dis","mean"),med=("rate","median"),mean=("rate","mean")).round(3).to_string())
a["k"]=a.dry.astype(int)+a.snowy.astype(int)+a.coldw.astype(int); print(a.groupby("k").agg(n=("dis","size"),p=("dis","mean"),mean=("rate","mean"),med=("rate","median")).round(3).to_string())
