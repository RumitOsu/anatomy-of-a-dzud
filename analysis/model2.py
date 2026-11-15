import pandas as pd, numpy as np, warnings; warnings.filterwarnings("ignore")
import model as m
SEP=["sum_pr_z","sum_pr_ratio","sum_soil_z","s_sum_t_z","lrp","lrp2","dens2","dens2_rel","clim_pr","clim_wt","lat","lon"]
DEC=SEP+["s_aut_t_z","s_nov_t_z","s_nov_snow_z","s_nov_snow","s_aut_snowdays_z","aut_pr_z","aut_soil_z"]
JAN=DEC+["s_dec_snow_z","s_dec_snow","s_dec_t_z","dens","dens_rel"]
HIND=JAN+["s_win_t_z","s_win_cold_z","s_janfeb_t_z","s_win_snow_z","s_win_snow","s_win_snowmax_z","s_win_snowdays_z","s_spr_snow_z","s_spr_t_z","spr_wind_z","spr_gust_z"]
SETS={"sep":SEP,"dec":DEC,"jan":JAN,"hind":HIND}
def load():
    A=m.prep(pd.read_parquet("aimag_panel2.parquet"))
    return A[(A.code!="511")&(A.year>=1994)&A.rate.notna()&A.sum_pr_z.notna()&A.rate_prev2.notna()]
if __name__=="__main__":
    A=load(); print("rows",len(A),"disasters",(A.rate>=.1).sum())
    res=[]
    for mode,first in (("wf",2008),("loyo",0)):
        for name,cols in SETS.items():
            for kind in ("gbm","ridge"):
                o=m.evaluate(A,cols,kind,mode,first); s=m.score(o); s.update(mode=mode,info=name,model=kind); res.append(s); o.to_parquet(f"p2_{mode}_{name}_{kind}.parquet")
    pd.set_option("display.width",220); print(pd.DataFrame(res).set_index(["mode","info","model"]).round(3).to_string())
    B=A.sort_values(["code","year"]).copy(); B["p"]=B.lrp
    print("persistence wf>=2008",{k:round(float(v),3) for k,v in m.score(B[B.year>=2008][["code","year","rate","y","p"]]).items()})
    print("persistence all",{k:round(float(v),3) for k,v in m.score(B[["code","year","rate","y","p"]]).items()})
    for f in ("p2_wf_jan_gbm","p2_wf_hind_gbm","p2_loyo_jan_gbm","p2_loyo_hind_gbm"):
        o=pd.read_parquet(f+".parquet"); o["pr"]=np.exp(o.p)-m.EPS
        print(f); print(o.groupby("year").agg(actual=("rate","mean"),pred=("pr","mean")).round(3).T.to_string())
