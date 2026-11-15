# Anatomy of a Dzud

A data story about Mongolia's killing winters, told as a scroll-driven page with an explorable map and interactive charts.

![The opening screen: the word DZUD with snow settling on it](docs/hero.png)

The finished page is `site/index.html`. It is one self-contained file, so you can download it and open it in any browser.

**The question:** why do some winters kill a quarter of a country's livestock while most kill almost none, and how early can you tell?

**The answer:** across 672 province-winters (1994–2025), a dry summer, long snow cover, or a cold winter is each close to harmless alone. When all three coincide, the chance of a province losing a tenth of its animals rises from 2% to 36%. Most of that is knowable by New Year, two to four months before the animals die. A three-variable formula forecasts as well as a 46-input gradient-boosted model (0.78 vs 0.76 AUC, both scored on held-out years).

## Layout

| Path | What it is |
| --- | --- |
| `site/index.html` | The published page. Built from `site/src.html` + `site/data.js` by `build_site.py`. |
| `data/province_winters.csv` | The final analysis table: one row per province and winter, with loss rate and the three conditions. |
| `data/*.parquet` | Full province and district panels with every engineered feature. |
| `analysis/` | The pipeline, in run order below. |

## Pipeline

Run from inside `analysis/`. Raw downloads land in `analysis/raw/` (about 100 MB, not kept in the repo).

1. `00_download.sh` pulls livestock tables from the National Statistics Office API (`nso.py`), district boundaries from HDX, NASA POWER monthly climate (`power.py`), and NOAA GHCN-Daily station files.
2. `build.py` joins livestock counts and losses, computes loss rates, and builds seasonal climate features per district and province.
3. `stations.py`, `build2.py`, `build3.py` turn 734k station-days into snow and temperature features, spread them to districts by distance, and average to provinces by herd size.
4. `findings.py` prints every headline number on the page: the three-condition table, the odds at each point in the year, the cut-off sensitivity check, the year-resampled uncertainty range, and both forecast scores.
5. `model.py`, `model2.py` hold the wider forecasting tests: walk-forward and leave-one-year-out, at four lead times, for ridge and LightGBM.
6. `export.py <project folder>` writes `site/data.js` (map geometry, series, province-winter flags).
7. `python build_site.py` inlines the data into the page.

## Sources

- National Statistics Office of Mongolia, data.1212.mn, tables DT_NSO_1001_021V1 and DT_NSO_1001_029V1
- NOAA GHCN-Daily, 41 Mongolian stations
- NASA POWER (MERRA-2) monthly climate
- OCHA / NSO administrative boundaries via the Humanitarian Data Exchange

## Known limits

- District-level loss counts start in 2012, so the long analysis is at province level.
- Die-offs cluster in a few years. A year-clustered bootstrap puts the 18-fold risk ratio anywhere from about 5-fold upward.
- 2024 does not fit the rule cleanly: the east had extreme snow after a summer that was not dry.
- Herder preparation (hay, early sales, moving herds) is not in the data.
