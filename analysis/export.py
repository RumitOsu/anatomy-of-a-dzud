import pandas as pd, numpy as np, json, sys, warnings; warnings.filterwarnings("ignore")
from shapely.geometry import shape
OUT=sys.argv[1]
L=pd.read_csv("raw/nso_029.csv",dtype={"code":str}); N=pd.read_csv("raw/nso_021.csv",dtype={"code":str})
for d in (L,N): d["value"]=pd.to_numeric(d.value,errors="coerce")
names={k:v.strip() for k,v in L[["code","name"]].drop_duplicates().values}
K=45.0; C=np.cos(np.deg2rad(47))
def proj(lon,lat): return ((lon-87.6)*C*K,(52.25-lat)*K)
def path(geom,tol):
    g=geom.simplify(tol,preserve_topology=True); polys=[g] if g.geom_type=="Polygon" else list(g.geoms); out=[]
    for p in polys:
        if p.area<tol*tol*4: continue
        for ring in [p.exterior]+list(p.interiors):
            pts=[proj(x,y) for x,y in ring.coords]; out.append("M"+"L".join(f"{x:.1f},{y:.1f}" for x,y in pts)+"Z")
    return "".join(out)
# ---- national
nat=[]
tot=N[(N.code=="0")].pivot(index="year",columns="type",values="value"); los=L[(L.code=="0")&(L.type=="Total")].set_index("year").value
for y in range(1970,2026):
    r=tot.loc[y]; l=los.get(y,np.nan); prev=tot.loc[y-1,"Total"] if y>1970 else np.nan
    nat.append({"y":y,"herd":round(r.Total/1000,3),"loss":None if pd.isna(l) else round(l/1000,3),"rate":None if pd.isna(l) or pd.isna(prev) else round(l/prev,4),"sp":[round(r[t]/1000,2) for t in ("Sheep","Goat","Cattle","Horse","Camel")]})
# ---- provinces
A=pd.read_parquet("aimag_panel3.parquet")
g1=json.load(open("raw/hdx/mng_admin1.geojson",encoding="utf-8")); prov=[]; pidx={}
for f in g1["features"]:
    pc=f["properties"]["adm1_pcode"]; code=[c for c in names if len(c)==3 and c[1:]==pc[2:]]
    if not code: print("no code",pc); continue
    code=code[0]; geom=shape(f["geometry"]); cx,cy=proj(*geom.representative_point().coords[0]); a=A[A.code==code].set_index("year")
    prov.append({"code":code,"name":names[code],"d":path(geom,0.03),"cx":round(cx,1),"cy":round(cy,1),"r":[None if (y not in a.index or pd.isna(a.rate.get(y))) else round(float(a.rate[y]),4) for y in range(1971,2026)],"herd":[None if (y not in a.index or pd.isna(a.herd.get(y))) else round(float(a.herd[y]),1) for y in range(1970,2026)]})
    pidx[code]=len(prov)-1
# ---- districts
S=pd.read_parquet("soum_panel3.parquet"); g2=json.load(open("raw/hdx/mng_admin2.geojson",encoding="utf-8")); dist=[]
for f in g2["features"]:
    pc=f["properties"]["adm2_pcode"]; s=S[S.pcode==pc].set_index("year")
    if not len(s): continue
    code=s.code.iloc[0]; geom=shape(f["geometry"])
    r=[None if (y not in s.index or pd.isna(s.rate.get(y)) or not np.isfinite(s.rate[y]) or s.herd_prev[y]<3) else round(float(min(s.rate[y],1)),3) for y in range(2012,2026)]
    dist.append({"name":names.get(code,"").strip(),"p":pidx.get(code[:3],-1),"d":path(geom,0.02),"r":r,"h":None if pd.isna(s.herd_prev.get(2024)) else round(float(s.herd_prev[2024]),1)})
# ---- province-winters
a=A[(A.year>=1994)&(A.code!="511")&A.rate.notna()].copy()
a["dry"]=(a.sum_pr_z<-0.5).astype(int); a["snowy"]=(a.d_snowdays_z>0.5).astype(int); a["coldw"]=(a.s_win_t_z<-0.5).astype(int); a["early"]=(a.d_nd_snowdays_z>0.5).astype(int)
pw=[[int(r.year),pidx[r.code],round(float(r.rate),4),int(r.dry),int(r.snowy),int(r.coldw),int(r.early)] for r in a.itertuples()]
a["dis"]=a.rate>=0.10
print(a.groupby(["dry","snowy","coldw"]).dis.agg(["size","sum"]).to_string()); print(a.groupby(["dry","early"]).dis.agg(["size","sum"]).to_string()); print(a.groupby("dry").dis.agg(["size","sum"]).to_string())
st=pd.read_csv("stations.csv"); stations=[[round(v,1) for v in proj(r.lon,r.lat)] for r in st.itertuples()]
out={"nat":nat,"prov":prov,"dist":dist,"pw":pw,"stations":stations}
js=json.dumps(out,ensure_ascii=False,separators=(",",":"))
open(OUT+"/site/data.js","w",encoding="utf-8").write("window.DZ="+js+";")
print("bytes",len(js),"prov",len(prov),"dist",len(dist),"pw",len(pw))
print("geometry bytes prov",sum(len(p["d"]) for p in prov),"dist",sum(len(p["d"]) for p in dist))
# headline numbers
n=pd.DataFrame(nat).set_index("y"); print(n.loc[[1970,1990,1999,2002,2009,2010,2019,2022,2023,2024,2025],["herd","loss","rate"]].to_string()); print(n.loc[[1970,1990,2022],"sp"].to_dict())
d24=pd.DataFrame([{"name":d["name"],"prov":prov[d["p"]]["name"] if d["p"]>=0 else "","r":d["r"][12],"h":d["h"]} for d in dist]).dropna().sort_values("r",ascending=False)
print(d24.head(8).to_string()); print("districts >=50% in 2024:",(d24.r>=0.5).sum(),">=30%:",(d24.r>=0.3).sum(),"of",len(d24)); print("sum 2000-2002 loss:",n.loc[2000:2002,"loss"].sum())
