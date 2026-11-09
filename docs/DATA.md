# Data that is not in the repository

The code, models, small results and the report are in this repository. The large data is on Hugging Face:

https://huggingface.co/datasets/Shivanshfrm9thdim/learned-cache-replacement-data

| Data | Size | Where in the Hugging Face repo | Can it be rebuilt? |
|---|---|---|---|
| `results/traces/` (binary LLC-input traces of the 24 runs) | about 1.1 GB (150 MB packed) | `traces/traces.tar.gz` and `traces/CHECKSUMS.sha256` | Yes, with `bash scripts/make_traces.sh` (needs Valgrind, takes hours) |
| `results/datasets/` (Belady-labelled training data, 829 files) | about 26 GB | `datasets/` (same folder layout as `results/datasets/`) and `DATASETS.sha256` | In principle yes, from the traces with `bash scripts/make_datasets.sh`. The rebuild was not verified bit for bit, so use the stored files to reproduce the reported numbers exactly. |

## Get the traces
Two copies exist. Use either: Hugging Face (step 1 below) or the GitHub release v1.0, https://github.com/shivanshbtr/learned-cache-replacement/releases/tag/v1.0 (files `traces.tar.gz` and `CHECKSUMS.sha256`). If you take them from the release, put both files in a folder `hfdata/traces/` and continue with step 2.
1. Download them (needs `pip install -U huggingface_hub`):
   `hf download Shivanshfrm9thdim/learned-cache-replacement-data --repo-type dataset --include "traces/*" --local-dir hfdata`
2. Check the download: `cd hfdata/traces && sha256sum -c CHECKSUMS.sha256` (every line must say OK).
3. From the project folder, unpack: `tar -xzf hfdata/traces/traces.tar.gz` (creates `results/traces/`).

## Get the datasets
1. From the project folder, download into `results/`:
   `hf download Shivanshfrm9thdim/learned-cache-replacement-data --repo-type dataset --include "datasets/*" --local-dir results`
   This creates `results/datasets/`. It is large (`train` is about 19 GB). The `val`, `test` and `_merged` folders are about 6.7 GB together and can be fetched alone, for example with `--include "datasets/val/*"`.
2. Check: download `DATASETS.sha256` into the project folder (`hf download Shivanshfrm9thdim/learned-cache-replacement-data DATASETS.sha256 --repo-type dataset --local-dir .`) and run `sha256sum -c DATASETS.sha256`. Every line must say OK. If you fetched only some folders, check just those: `grep "results/datasets/test/" DATASETS.sha256 | sha256sum -c -` (change the folder name to match).

## Make the packages (maintainers)
`bash scripts/package_repo.sh` makes the small repository copy; `bash scripts/pack_data.sh` makes `traces.tar.gz`, the checksums and the datasets manifest.

## Reproducibility notes
Fixed program inputs (`configs/inputs.tsv`), seeds, ASLR off, empty environment and a hash of the configuration in every result file. Test inputs were run once with the frozen model set (`results/final/model_set.json`, SHA-256 of every model file); the marker files that block a second test run are the files `results/final/test_started` and `results/ext/test_started`.
