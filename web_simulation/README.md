# Web simulation (static, offline)

Open `index.html` in a browser (double-click). No server, no internet, no build step.
Screens: Overview, Single application, Switching, Shared cache, Decision inspector, Results and tables.

## What is on the screen
- **Full run** badge: numbers computed on the complete traces by the C++ simulator (E1-E8 from `final_results`, the mixed-stream extension from `ext_results`).
- **Sample window** badge: an animated replay of 4,000 LLC accesses taken from the middle of a held-out test trace (never from the warm-up). It shows what the policies do on a short stretch. It is not a full-run statistic.
- Nothing is typed in by hand. Every number is copied or computed from result files by `export/build_web_data.py`.

## Data folder
`data/main.js`, `data/ext.js` (full-run results) and `data/win_*.js` (sample windows). They are plain `.js` files that set one global each, so the page also works from a folder, where browsers block `fetch()` of `.json`.

## Rebuild the data (on the machine that has the traces and models)
1. Sample windows (replays existing TEST traces with the frozen models; selects nothing; about 15 to 40 minutes):
   `python3 scripts/make_windows.py --out results/web/windows`
2. Pack everything into `data/`:
   `python3 export/build_web_data.py --final <folder of final_results.zip> --ext <folder of ext_results.zip> --windows results/web/windows --out web_simulation/data`

Without step 1 the page still works; the animated parts show a hint instead.

## Tests
`python3 tests/test_winexport.py` (exporter against hand-checked traces and the simulator) and `python3 tests/test_make_windows.py` (driver and data packer on small traces). Both are part of `make test`.

## Limits (also stated in the report)
Simulation only: no hardware was modified. Software decision times are from one core of a laptop. Sample windows come from one trace per application and one merged mixture.
