import json, urllib.request, time, os
from concurrent.futures import ThreadPoolExecutor
g=json.load(open("raw/hdx/mng_admin2.geojson",encoding="utf-8"))
pts=[(f["properties"]["adm2_pcode"],f["properties"]["center_lat"],f["properties"]["center_lon"]) for f in g["features"]]
os.makedirs("raw/power",exist_ok=True)
P="T2M,T2M_MIN,T2M_MAX,PRECTOTCORR,SNODP,PRECSNOLAND,GWETROOT,GWETTOP,WS10M,WS10M_MAX,FROST_DAYS,TS"
def get(p):
    pc,lat,lon=p; fn=f"raw/power/{pc}.json"
    if os.path.exists(fn) and os.path.getsize(fn)>5000: return 1
    u=f"https://power.larc.nasa.gov/api/temporal/monthly/point?parameters={P}&community=AG&longitude={lon:.4f}&latitude={lat:.4f}&start=1981&end=2025&format=JSON"
    for a in range(6):
        try:
            d=urllib.request.urlopen(u,timeout=90).read(); j=json.loads(d)
            if "properties" in j: open(fn,"wb").write(d); return 1
            err=str(j)[:200]
        except Exception as e: err=e
        time.sleep(3+a*3)
    print("FAIL",pc,err,flush=True); return 0
with ThreadPoolExecutor(4) as ex: r=list(ex.map(get,pts))
print("ok",sum(r),"of",len(pts))
