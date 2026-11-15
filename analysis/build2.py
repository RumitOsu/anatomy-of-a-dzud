import pandas as pd, numpy as np, warnings; warnings.filterwarnings("ignore")
M=pd.read_parquet("station_monthly.parquet"); M=M[M.n>=18]
M["Y"]=np.where(M.m>=6,M.y+1,M.y)
def agg(months,var,how="mean",minm=None):
    x=M[M.m.isin(months)].groupby(["id","Y"])[var].agg([how,"count"]); x=x[x["count"]>=(minm or len(months))]; return x[how]
F=pd.DataFrame({"s_nov_snow":agg([11],"snow"),"s_dec_snow":agg([12],"snow"),"s_aut_snowdays":agg([10,11],"snow_days","sum"),
 "s_win_snow":agg([12,1,2],"snow"),"s_win_snowmax":agg([12,1,2,3],"snow_max","max"),"s_win_snowdays":agg([12,1,2],"snow_days","sum"),"s_spr_snow":agg([3,4],"snow"),
 "s_nov_t":agg([11],"t"),"s_dec_t":agg([12],"t"),"s_aut_t":agg([9,10,11],"t"),"s_win_t":agg([12,1,2],"t"),"s_win_cold":agg([12,1,2],"cold_days","sum"),"s_janfeb_t":agg([1,2],"t"),"s_spr_t":agg([3,4,5],"t"),"s_sum_t":agg([6,7,8],"t")}).reset_index()
cols=[c for c in F.columns if c.startswith("s_")]
base=F[(F.Y>=1994)&(F.Y<=2007)].groupby("id")[cols]; mu,sd=base.mean(),base.std()
Z=F[["id","Y"]].copy()
for c in cols:
    s=F.id.map(sd[c]); s=s.where(s>0.3,0.3) if "snow" in c and "days" not in c else s.replace(0,np.nan)
    Z[c+"_z"]=((F[c]-F.id.map(mu[c]))/s).clip(-5,5); Z[c]=F[c]
st=pd.read_csv("stations.csv").set_index("id")
S=pd.read_parquet("soum_panel.parquet"); A=pd.read_parquet("aimag_panel.parquet")
so=S.drop_duplicates("pcode").set_index("pcode")[["lat","lon","acode"]]
def km(lat1,lon1,lat2,lon2):
    p=np.pi/180; a=np.sin((lat2-lat1)*p/2)**2+np.cos(lat1*p)*np.cos(lat2*p)*np.sin((lon2-lon1)*p/2)**2; return 12742*np.arcsin(np.sqrt(a))
Dm=pd.DataFrame({i:km(so.lat.values,so.lon.values,st.lat[i],st.lon[i]) for i in st.index},index=so.index)
vals=[c for c in Z.columns if c not in ("id","Y")]
out=[]
for Y,z in Z.groupby("Y"):
    z=z.set_index("id"); res=pd.DataFrame(index=so.index,columns=vals,dtype=float)
    for v in vals:
        ok=z[v].dropna()
        if len(ok)<15: continue
        d=Dm[ok.index].values; idx=np.argsort(d,axis=1)[:,:4]; dd=np.take_along_axis(d,idx,1); w=1/(dd+20)**2
        res[v]=(ok.values[idx]*w).sum(1)/w.sum(1)
    res["near_km"]=Dm.min(axis=1); res["year"]=Y; out.append(res.reset_index())
SS=pd.concat(out)
S=S.merge(SS,on=["pcode","year"],how="left")
w=S.groupby("pcode").herd.mean().fillna(0)+1; SS["w"]=SS.pcode.map(w).fillna(1); SS["code"]=SS.pcode.map(so.acode)
AA=SS.dropna(subset=["code"]).groupby(["code","year"]).apply(lambda d: pd.Series({v:np.average(d[v].fillna(d[v].mean()),weights=d.w) if d[v].notna().any() else np.nan for v in vals})).reset_index()
A=A.merge(AA,on=["code","year"],how="left")
S.to_parquet("soum_panel2.parquet"); A.to_parquet("aimag_panel2.parquet")
print(A[(A.year>=1990)].groupby("year").agg(rate=("rate","mean"),nov_snow=("s_nov_snow_z","mean"),win_snow=("s_win_snow_z","mean"),win_t=("s_win_t_z","mean"),cold=("s_win_cold_z","mean"),n=("s_win_snow_z","count")).round(2).T.to_string())
a=A[(A.year>=1994)&(A.code!="511")]; a["y"]=np.log(a.rate+0.005)
print(a[[c for c in a.columns if c.startswith("s_")]+["y"]].corr(method="spearman").y.sort_values().round(2).to_string())
