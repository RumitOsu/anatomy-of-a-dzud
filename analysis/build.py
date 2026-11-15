import pandas as pd, numpy as np, json, glob, os, warnings; warnings.filterwarnings("ignore")
SU={"Horse":7,"Cattle":6,"Camel":5,"Sheep":1,"Goat":0.9}
L=pd.read_csv("raw/nso_029.csv",dtype={"code":str}); N=pd.read_csv("raw/nso_021.csv",dtype={"code":str})
for d in (L,N): d["value"]=pd.to_numeric(d.value,errors="coerce")
names=dict(L[["code","name"]].drop_duplicates().values)
g=json.load(open("raw/hdx/mng_admin2.geojson",encoding="utf-8"))
meta=pd.DataFrame([{"pcode":f["properties"]["adm2_pcode"],"lat":f["properties"]["center_lat"],"lon":f["properties"]["center_lon"],"area":f["properties"]["area_sqkm"],"name_mn":f["properties"]["adm2_name1"],"aimag_mn":f["properties"]["adm1_name1"]} for f in g["features"]]).set_index("pcode")
# ---------- climate, soum-monthly
rows=[]
for fn in glob.glob("raw/power/*.json"):
    pc=os.path.basename(fn)[:-5]; p=json.load(open(fn))["properties"]["parameter"]
    d=pd.DataFrame(p); d=d[~d.index.str.endswith("13")]; d=d.replace(-999,np.nan)
    d["pcode"]=pc; d["y"]=d.index.str[:4].astype(int); d["m"]=d.index.str[4:].astype(int); rows.append(d)
C=pd.concat(rows).reset_index(drop=True)
print("climate rows",len(C),"missing share:\n",C.isna().mean().round(3).to_string())
C.to_parquet("clim_soum.parquet")
# ---------- livestock wide tables
def wide(D,typ,ln): return D[(D.type==typ)&(D.code.str.len()==ln)].pivot(index="code",columns="year",values="value")
def panel(ln):
    herd=wide(N,"Total",ln); loss=wide(L,"Total",ln)
    su=sum(wide(N,t,ln).fillna(0)*w for t,w in SU.items())
    goat=wide(N,"Goat",ln)/herd
    o=pd.DataFrame({"herd":herd.stack(dropna=False),"loss":loss.reindex(columns=herd.columns).stack(dropna=False),"su":su.stack(dropna=False),"goat_share":goat.stack(dropna=False)}).reset_index()
    o.columns=["code","year","herd","loss","su","goat_share"]; o=o.sort_values(["code","year"])
    gb=o.groupby("code")
    o["herd_prev"]=gb.herd.shift(1); o["su_prev"]=gb.su.shift(1); o["su_prev2"]=gb.su.shift(2); o["herd_prev2"]=gb.herd.shift(2)
    o["rate"]=o.loss/o.herd_prev; o["chg"]=o.herd/o.herd_prev-1
    o["rate_prev"]=o.groupby("code").rate.shift(1); o["rate_prev2"]=o.groupby("code").rate.shift(2)
    o["name"]=o.code.map(names); return o
S=panel(5); A=panel(3)
S["pcode"]="MN"+S.code.str[1:]; S["acode"]=S.code.str[:3]; S=S[S.pcode.isin(meta.index)]
S=S.join(meta,on="pcode")
# ---------- seasonal features (function of a monthly table with key column)
V=["T2M","T2M_MIN","PRECTOTCORR","SNODP","PRECSNOLAND","GWETROOT","WS10M","WS10M_MAX"]
def feats(M,key):
    """M: monthly rows [key,y,m]+V.  Returns rows per (key, Y) describing the lead-up to the winter that ends in year Y."""
    M=M.copy(); M["Y"]=np.where(M.m>=6,M.y+1,M.y)      # Jun(Y-1)..May(Y) belong to "year Y"
    def agg(months,var,how="mean"):
        return M[M.m.isin(months)].groupby([key,"Y"])[var].agg(how)
    F=pd.DataFrame({
     "sum_pr":agg([6,7,8],"PRECTOTCORR"),"sum_t":agg([6,7,8],"T2M"),"sum_soil":agg([7,8],"GWETROOT"),
     "aut_pr":agg([9,10,11],"PRECTOTCORR"),"aut_t":agg([9,10,11],"T2M"),"nov_snow":agg([11],"SNODP"),"nov_t":agg([11],"T2M"),"aut_snowfall":agg([10,11],"PRECSNOLAND"),"aut_soil":agg([9,10],"GWETROOT"),
     "dec_snow":agg([12],"SNODP"),"dec_t":agg([12],"T2M"),
     "win_t":agg([12,1,2],"T2M"),"win_tmin":agg([12,1,2],"T2M_MIN","min"),"win_snow":agg([12,1,2,3],"SNODP","max"),"win_snowfall":agg([12,1,2],"PRECSNOLAND"),
     "spr_t":agg([3,4,5],"T2M"),"spr_wind":agg([3,4,5],"WS10M"),"spr_gust":agg([3,4,5],"WS10M_MAX","max"),"spr_pr":agg([3,4,5],"PRECTOTCORR"),"spr_snow":agg([3,4],"SNODP")})
    cnt=M.groupby([key,"Y"]).m.count(); F=F[cnt.reindex(F.index)==12]
    F=F.reset_index()
    base=F[(F.Y>=1982)&(F.Y<=2000)].groupby(key)      # fixed early baseline: always in the past for every test year
    mu,sd=base.mean(numeric_only=True).drop(columns="Y"),base.std(numeric_only=True).drop(columns="Y")
    Z=F.copy()
    for c in mu.columns:
        Z[c+"_z"]=(F[c]-F[key].map(mu[c]))/F[key].map(sd[c]).replace(0,np.nan)
    Z["clim_pr"]=Z[key].map(mu["sum_pr"]); Z["clim_wt"]=Z[key].map(mu["win_t"]); Z["clim_snow"]=Z[key].map(mu["win_snow"])
    Z["sum_pr_ratio"]=F.sum_pr/Z.clim_pr
    return Z
FS=feats(C,"pcode")
# aimag climate = herd-weighted mean of its soums (weather where the animals are)
w=S.groupby("pcode").herd.mean().fillna(0)+1
C2=C.copy(); C2["w"]=C2.pcode.map(w).fillna(1); C2["acode"]=C2.pcode.map(S.drop_duplicates("pcode").set_index("pcode").acode)
C2=C2.dropna(subset=["acode"])
CA=C2.groupby(["acode","y","m"]).apply(lambda d: pd.Series({v:np.average(d[v],weights=d.w) for v in V})).reset_index()
FA=feats(CA,"acode")
S=S.merge(FS.rename(columns={"Y":"year"}),on=["pcode","year"],how="left")
A=A.merge(FA.rename(columns={"Y":"year","acode":"code"}),on=["code","year"],how="left")
aa=S.groupby("acode").agg(area=("area","first")) ; area=S.drop_duplicates("pcode").groupby("acode").area.sum()
cen=S.drop_duplicates("pcode").groupby("acode")[["lat","lon"]].mean()
A["area"]=A.code.map(area); A["lat"]=A.code.map(cen.lat); A["lon"]=A.code.map(cen.lon)
for D in (S,A):
    D["dens"]=D.su_prev/D.area; D["dens2"]=D.su_prev2/D.area
    D["dens_rel"]=D.su_prev/D.groupby("code").su_prev.transform(lambda s:s.shift(1).rolling(10,min_periods=5).mean())
    D["dens2_rel"]=D.su_prev2/D.groupby("code").su_prev2.transform(lambda s:s.shift(1).rolling(10,min_periods=5).mean())
S.to_parquet("soum_panel.parquet"); A.to_parquet("aimag_panel.parquet")
print(S.shape,A.shape)
print(A[(A.year>=1982)].groupby("year").agg(n=("rate","count"),rate=("rate","mean"),sum_pr=("sum_pr_z","mean"),win_snow=("win_snow_z","mean"),win_t=("win_t_z","mean"),nov_snow=("nov_snow_z","mean")).round(2).to_string())
