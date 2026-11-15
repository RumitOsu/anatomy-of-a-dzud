import pandas as pd, numpy as np, glob, os
st=pd.read_fwf("raw/ghcn/st.txt",colspecs=[(0,11),(12,20),(21,30),(31,37),(41,71)],names=["id","lat","lon","elev","name"]); st=st[st.id.str.startswith("MG")].set_index("id")
fr=[]
for f in glob.glob("raw/ghcn/MG*.csv.gz"):
    d=pd.read_csv(f,header=None,usecols=[0,1,2,3,5],names=["id","date","el","v","q"],dtype={"q":str})
    d=d[d.el.isin(["TAVG","TMIN","TMAX","PRCP","SNWD"])&d.q.isna()]; fr.append(d)
D=pd.concat(fr); D["date"]=pd.to_datetime(D.date.astype(str),format="%Y%m%d")
W=D.pivot_table(index=["id","date"],columns="el",values="v",aggfunc="first").reset_index()
for c in ("TAVG","TMIN","TMAX","PRCP"): W[c]=W[c]/10.0        # tenths of degC / mm
W["SNWD"]=W["SNWD"]/10.0                                        # mm -> cm
W=W[W.date>="1973-01-01"]
print(W.groupby(W.date.dt.year)[["TAVG","TMIN","PRCP","SNWD"]].count().iloc[::4].to_string())
print(W.describe().round(1).to_string())
W["tday"]=W.TAVG.fillna((W.TMIN+W.TMAX)/2)
W["snow0"]=W.SNWD.fillna(0).where(W.tday.notna())        # a day with a temperature report but no snow report counts as bare ground
W["y"]=W.date.dt.year; W["m"]=W.date.dt.month
M=W.groupby(["id","y","m"]).agg(n=("tday","count"),t=("tday","mean"),tmin=("TMIN","min"),tmin_mean=("TMIN","mean"),pr=("PRCP","sum"),pr_n=("PRCP","count"),snow=("snow0","mean"),snow_max=("snow0","max"),snow_days=("snow0",lambda s:(s>0).sum()),cold_days=("TMIN",lambda s:(s<-30).sum()),snwd_n=("SNWD","count")).reset_index()
M=M.join(st[["lat","lon","elev","name"]],on="id")
M.to_parquet("station_monthly.parquet"); st.to_csv("stations.csv")
print(len(M), M[M.n>=20].groupby("y").id.nunique().iloc[::3].to_string())
x=M[(M.m.isin([12,1,2]))&(M.n>=20)].copy(); x["Y"]=np.where(x.m==12,x.y+1,x.y)
print(x.groupby("Y").agg(snow=("snow","mean"),snow_max=("snow_max","mean"),t=("t","mean"),stations=("id","nunique")).round(1).loc[1995:].T.to_string())
