#!/usr/bin/env bash
# Fetches every raw input into ./raw (run from the folder that holds these scripts). No API keys needed.
set -e
mkdir -p raw/ghcn raw/hdx raw/power
B="https://data.1212.mn/api/v1/en/NSO/Industry,%20service/Livestock"
curl -s "$B/DT_NSO_1001_029V1.px" -o raw/meta_029.json   # losses of adult animals, by district and year
curl -s "$B/DT_NSO_1001_021V1.px" -o raw/meta_021.json   # number of livestock, by district and year
(cd raw && python ../nso.py)                              # -> raw/nso_029.csv, raw/nso_021.csv
curl -sL "https://data.humdata.org/dataset/a9b0a8a6-cb14-448e-b35c-aa5eb51b0557/resource/de6b11d3-767d-449f-99fc-80c5b4e9abe3/download/mng_admin_boundaries.geojson.zip" -o raw/hdx/b.zip
(cd raw/hdx && unzip -o -q b.zip)
python power.py                                           # NASA POWER monthly climate at 339 district centres
curl -s "https://noaa-ghcn-pds.s3.amazonaws.com/ghcnd-stations.txt" -o raw/ghcn/st.txt
grep "^MG" raw/ghcn/st.txt | cut -c1-11 | while read id; do
  curl -s "https://noaa-ghcn-pds.s3.amazonaws.com/csv.gz/by_station/$id.csv.gz" -o "raw/ghcn/$id.csv.gz"
done
