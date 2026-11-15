import urllib.request, json, time, csv
B="https://data.1212.mn/api/v1/en/NSO/Industry,%20service/Livestock/"
def pull(tbl,key):
    m=json.load(open(f"meta_{key}.json",encoding="utf-8"))
    vt,vr,vy=m["variables"]
    regs=[c for c in vr["values"] if len(c)<=5]
    names=dict(zip(vr["values"],vr["valueTexts"])); years=dict(zip(vy["values"],vy["valueTexts"])); types=dict(zip(vt["values"],vt["valueTexts"]))
    rows=[]
    for t in vt["values"]:
        q={"query":[{"code":vt["code"],"selection":{"filter":"item","values":[t]}},{"code":vr["code"],"selection":{"filter":"item","values":regs}},{"code":vy["code"],"selection":{"filter":"all","values":["*"]}}],"response":{"format":"json"}}
        for a in range(5):
            try:
                req=urllib.request.Request(B+tbl,data=json.dumps(q).encode("utf-8"),headers={"Content-Type":"application/json"})
                j=json.load(urllib.request.urlopen(req,timeout=120)); break
            except Exception as e: print("retry",t,e); time.sleep(3)
        for d in j["data"]:
            k=d["key"]; rows.append((types[k[0]],k[1],names[k[1]].strip(),years[k[2]],d["values"][0]))
        print(tbl,types[t],len(j["data"]),flush=True); time.sleep(1)
    with open(f"nso_{key}.csv","w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["type","code","name","year","value"]); w.writerows(rows)
pull("DT_NSO_1001_029V1.px","029"); pull("DT_NSO_1001_021V1.px","021")
