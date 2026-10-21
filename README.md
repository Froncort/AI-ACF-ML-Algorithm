<h1 align="center">
  <a href="https://www.froncort.ai/" target="_blank">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="docs/assets/brand/froncort-banner-dark.png" />
      <img
        align="center"
        alt="AI-ACF Open Source by Froncort.ai"
        src="docs/assets/brand/froncort-banner-light.png"
        style="width:100%;"
      />
    </picture>
  </a>
</h1>

<p align="center">
  <a href="LICENSE" target="_blank"><img src="https://img.shields.io/badge/license-Apache%202.0-A30050?style=for-the-badge" alt="License: Apache 2.0" /></a>&nbsp;
  <img src="https://img.shields.io/badge/python-3.10%2B-A30050?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+" />&nbsp;
  <img src="https://img.shields.io/badge/notebooks-Jupyter-A30050?style=for-the-badge&logo=jupyter&logoColor=white" alt="Jupyter notebooks" />
</p>

<p align="center">
  <a href="https://www.froncort.ai/" target="_blank"><img src="https://img.shields.io/badge/built%20by-Froncort.ai-A30050?style=flat-square" alt="Built by Froncort.ai" /></a>&nbsp;
  <img src="https://img.shields.io/badge/runs-100%25%20local-A30050?style=flat-square" alt="Runs 100% local" />&nbsp;
  <img src="https://img.shields.io/badge/no%20account-required-A30050?style=flat-square" alt="No account required" />&nbsp;
  <img src="https://img.shields.io/badge/model-XGBoost%20hurdle-A30050?style=flat-square" alt="Model: XGBoost hurdle" />&nbsp;
  <img src="https://img.shields.io/badge/grid-Uber%20H3-A30050?style=flat-square" alt="Grid: Uber H3" />
</p>

<p align="center">
  <b>Open-source TB hotspot prioritization: rank which hexagons field teams should visit next month.</b>
</p>

<p align="center">
  <a href="#five-minute-first-run"><b>Get Started</b></a>&nbsp;&nbsp;&nbsp;⚡&nbsp;&nbsp;&nbsp;
  <a href="#notebook-order"><b>Notebooks</b></a>&nbsp;&nbsp;&nbsp;🧭&nbsp;&nbsp;&nbsp;
  <a href="#inputs-at-a-glance"><b>Prepare Your Data</b></a>&nbsp;&nbsp;&nbsp;📋&nbsp;&nbsp;&nbsp;
  <a href="docs/methodology.md"><b>Methodology</b></a>&nbsp;&nbsp;&nbsp;🧮&nbsp;&nbsp;&nbsp;
  <a href="https://www.froncort.ai/ai-netra" target="_blank"><b>AI-NETRA</b></a>
</p>

<br>
<br>

# 🧬 Welcome to AI-ACF Open Source

**AI-ACF Open Source** is the free, local-first edition of Froncort's **AI-NETRA Active Case Finding (ACF)** method for tuberculosis. Use it to learn how limited screening teams can be pointed at the hexagons most likely to yield cases **next month**.

It runs on your own computer as numbered **Jupyter notebooks** plus a small Python package. The quick start uses **synthetic data only**. Real patient registers never need to leave your machine, and they are not shipped in this repository.

<br>

## 🩺 ACF in Brief

**Active case finding** sends field teams into communities to find people with TB who have not yet been notified. Teams cannot cover a whole district every month, so the operational question is a ranking problem:

> Which geographic cells should we visit next?

This release answers that on an Uber **H3 hexagon** grid:

> Case history → hexagon × month features → two-stage hurdle model → ranked hexes for next month

**Features at month M predict the case count at month M+1.** That time shift is required. Scoring a hexagon with same-month labels looks accurate offline and fails in the field, because next month's cases are not known when the plan is made.

The aim is not a perfect case count. It is a **priority list** (Precision@K, cases captured, lift versus random visits) that a programme can review before sending teams.

<br>

## 🔥 Why AI-ACF Open Source Is Different

- **🔒 Local-first and private**: Notebooks, models, and maps run on your machine. The synthetic path needs no account and no API key. Real extracts stay in a folder you choose and are gitignored.

- **📓 Built as a walkthrough, not a black box**: Seven notebooks (`00` → `06`) show the contract, features, target shift, leakage check, training, evaluation, and the ranked map. You can inspect every table.

- **🧮 The same hurdle idea as production**: A gate classifier estimates P(any cases), isotonic regression calibrates that probability, and a Poisson count model estimates how many cases to expect when the gate is open. The score is `calibrated gate probability × expected count`.

- **🗺️ Hexagons you can see**: Ranked cells are written as dashboard JSON and an interactive HTML map (colored High / Medium / Low polygons), the same spatial idea as the product map.

- **📋 Bring your own register**: A Nikshay-style Notification Register can be standardized, geocoded once per health facility, and turned into `patients.csv` + `hexagons.csv`. GPS already on the row can skip geocoding.

- **🔁 Reproducible synthetic demo**: `generate_synthetic_dataset(seed=42)` rebuilds the same learning dataset. Real programme data is never included.

- **🌱 Open and extensible**: Apache 2.0 licensed. Read the methodology, adapt features to your programme, and keep patient-level files out of public forks.

<br>

## 🛠️ What's Included

- [x] Documented input contract (`patients.csv`, `hexagons.csv`, optional `contacts.csv`)
- [x] Synthetic AOI generator (cluster of H3 cells, not a real district)
- [x] Nikshay-style ingest: metadata rows, facility hierarchy, flexible dates
- [x] Facility-centroid geocoding (Google Maps, or free Nominatim fallback)
- [x] Hexagon × month features: clinical, lags, contacts, neighbor risk, interactions
- [x] Future target (month M → month M+1) and a hard leakage check
- [x] Temporal train/test split (not a random row split)
- [x] Two-stage XGBoost hurdle + isotonic gate calibration
- [x] AUC, RMSE, Precision@K / Efficiency@K / Lift@K, and walk-forward
- [x] Top-K recommendations, risk tiers, dashboard JSON, Folium hexagon map

<br>

<a id="ai-acf-platform"></a>

# ⚖️ Open Source vs AI-NETRA ACF

Ranking hexagons is one part of running an active-case-finding programme. **[AI-NETRA](https://www.froncort.ai/ai-netra)** by **Froncort.ai** builds on the same foundation and adds the operational product around ingestion, maps, and programme workflows.

| Capability | What it does | Open Source<br><sub>Free</sub> | AI-NETRA ACF<br><sub>Platform</sub> |
| ---------- | ------------ | :----------------------------: | :---------------------------------: |
| **Synthetic learning path** | Run the full method on generated data with no patient file | ✅ | — |
| **Hexagon × month model** | Two-stage hurdle: gate, calibration, Poisson count, ranked list | ✅ | ✅ |
| **Local notebooks** | Step-through training, evaluation, and a browser map | ✅ | — |
| **Bring-your-own CSVs** | Point the notebooks at your own extracts | ✅ | ✅ |
| **Facility → H3 geocoding** | Turn facility or address text into hexagons (your API key) | ✅ | ✅ |
| **Live programme map** | District map, tiers, landmarks, and region labels for field teams | — | ✅ |
| **Register ingestion at scale** | Portal workflows, validation, and repeat monthly loads | — | ✅ |
| **Enrichment** | Reverse-geocoded landmarks and plain-language explanations | — | ✅ |
| **Saved programme models** | Versioned models, projects, and shared workspaces | — | ✅ |
| **Operational deployment** | APIs, access control, and production monitoring | — | ✅ |

<p align="center">
  <a href="https://www.froncort.ai/ai-netra" target="_blank"><img src="https://img.shields.io/badge/Learn%20about%20AI--NETRA-%E2%86%92-A30050?style=for-the-badge" alt="Learn about AI-NETRA" /></a>&nbsp;
  <a href="mailto:info@froncort.ai" target="_blank"><img src="https://img.shields.io/badge/Talk%20to%20Froncort-%E2%86%92-A30050?style=for-the-badge" alt="Talk to Froncort" /></a>
</p>

<br>
<br>

# 🚀 Getting Started

## 👥 Who Should Use Which Path?

| Audience | Recommended path |
| -------- | ---------------- |
| Students and new analysts | [Five-minute first run](#five-minute-first-run) on synthetic data |
| Modelling teams | Notebooks `00` → `06`, then [docs/methodology.md](docs/methodology.md) |
| Programme data teams | Notebook `00` Path B (real register) + [Inputs](#inputs-at-a-glance) |
| Integrators | Import `pipeline` in Python after the notebooks have been run once |

## ✅ Prerequisites

- **Python 3.10 or newer**
- **macOS, Linux, or Windows**
- **Internet** on first install (`pip install -r requirements.txt`). The synthetic path then runs offline.
- **Jupyter** (installed from `requirements.txt`)
- Synthetic quick start needs **no API key**
- Real Nikshay files without GPS need a **Google Maps Geocoding API** key (or the free Nominatim fallback). See [Geocoding](#geocoding).

## ⚡ Five-Minute First Run

<a id="five-minute-first-run"></a>

```bash
git clone https://github.com/Froncort/AI-ACF-open-source.git
cd AI-ACF-open-source

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install folium                 # hexagon map in notebook 06

python -c "from pipeline.sample_data import generate_synthetic_dataset; generate_synthetic_dataset(n_hex=300, n_months=20)"

jupyter notebook notebooks/
```

1. Open **`00_data_contract_and_sample.ipynb`** and confirm `USE_REAL_DATA = False`
2. Use **Run All**, then do the same for notebooks **`01` through `06` in order**
3. Open **`artifacts/hotspot_map.html`** in a browser to see ranked hexagons

Do not skip a notebook. Each one reads the files the previous notebook wrote.

<a id="notebook-order"></a>

### Notebook order

| # | Notebook | What it does | Writes |
| - | -------- | ------------ | ------ |
| **00** | [00_data_contract_and_sample.ipynb](notebooks/00_data_contract_and_sample.ipynb) | Schema, synthetic data, or real register → H3 | `sample_data/*.csv` |
| **01** | [01_feature_engineering.ipynb](notebooks/01_feature_engineering.ipynb) | Clinical, temporal, spatial, contact features | `artifacts/features.csv` |
| **02** | [02_target_creation.ipynb](notebooks/02_target_creation.ipynb) | Month M features → month M+1 case count | `artifacts/labeled.csv` |
| **03** | [03_dataset_preparation.ipynb](notebooks/03_dataset_preparation.ipynb) | Time split and leakage check | `artifacts/train.csv`, `test.csv`, `feature_cols.csv` |
| **04** | [04_model_training.ipynb](notebooks/04_model_training.ipynb) | Two-stage XGBoost + isotonic calibrator | `artifacts/models/` |
| **05** | [05_evaluation.ipynb](notebooks/05_evaluation.ipynb) | AUC, RMSE, Precision@K, walk-forward | Metrics in the notebook |
| **06** | [06_operational_inference.ipynb](notebooks/06_operational_inference.ipynb) | Top-K hexes, tiers, map, dashboard JSON | `recommendations.csv`, `dashboard_recommendations.json`, `hotspot_map.html` |

<br>

## 🗺️ How This Fits an ACF Planning Cycle

1. **Define scope**: district or AOI, H3 resolution, and the month you are planning for.
2. **Prepare cases**: notification register or an export that already has `h3_index`.
3. **Build the hexagon list**: every cell in the AOI, including cells with zero cases.
4. **Train on the past**: features at month M, labels at month M+1, hold out the latest months.
5. **Review the ranked list**: Precision@K and the map, then decide where teams go.
6. **Refresh**: add the new month of notifications and repeat. Do not reuse a score that was fit on future labels.

This repository supports stages **2–5** as a local method. Scoping, field operations, and clinical decisions stay with the programme. For a full operational deployment, see [AI-NETRA ACF](#ai-acf-platform).

<br>
<br>

# 📥 Inputs at a Glance

<a id="inputs-at-a-glance"></a>

Real programme data is **not** in this repository. The default folder is synthetic `sample_data/`.

| File | Required columns | Role |
| ---- | ---------------- | ---- |
| `patients.csv` | `patient_id`, `h3_index`, `diagnosis_date` | Notified cases |
| `hexagons.csv` | `h3_index` (`lat`, `lng` recommended) | **AOI hexagon list** for one region |
| `contacts.csv` | `h3_index`, `month`, `n_contacts` | Optional contact-tracing counts |

Optional clinical columns (`age`, `site_of_disease`, `hiv_status`, `diabetes_status`, `is_recidivism`, …) improve features when they are present. Missing optional columns are filled with safe defaults.

Point notebook `01` at your folder:

```python
DATA_DIR = "/path/to/your/csvs"   # instead of sample_data/
```

### What `hexagons.csv` is

**`hexagons.csv` is the list of H3 cells that cover your area of interest** — a district, TBU catchment, ward cluster, or demo AOI. It is not every hexagon on Earth.

The feature step builds a **complete** hexagon × month grid, including months with zero cases. Without that list, quiet cells never appear as negatives and the gate cannot learn “no cases next month.”

| Column | Required? | Meaning |
| ------ | --------- | ------- |
| `h3_index` | **Yes** | H3 cell id (same resolution as patients, often res 9 or 10) |
| `lat` | Recommended | Cell centroid latitude (maps and neighbor fallback) |
| `lng` | Recommended | Cell centroid longitude |
| `slum_density` | Optional | 0–1 vulnerability proxy used as a feature |

| How you got the file | What the rows represent |
| -------------------- | ----------------------- |
| Synthetic notebook `00` | A compact demo cluster around one origin. Not a real programme region. |
| Real ingest in notebook `00` | One row per H3 cell seen after facility geocoding, with centroids. |
| Your GIS / product boundary | Every cell covering the region, including cells that have never had a case. Prefer this for a full AOI. |

On the product map, this same cell list is what gets drawn as polygons.

<br>

## 🎯 Analysis Choices

<a id="analysis-choices"></a>

**Data path** (notebook `00`):

| Setting | When |
| ------- | ---- |
| `USE_REAL_DATA = False` | Learning, demos, and CI. Generates synthetic CSVs. **Default.** |
| `USE_REAL_DATA = True` | You have a Notification Register CSV. Set `REAL_REGISTER_PATH`. Never commit that file. |

**Model** (fixed in these notebooks, matching the production hurdle):

| Stage | What it learns |
| ----- | -------------- |
| Gate | `XGBClassifier` → P(any cases next month). Class imbalance uses `scale_pos_weight`. |
| Calibrate | Isotonic regression fit on the gate's validation fold only. |
| Count | `XGBRegressor` with `count:poisson`, trained on rows where `target_cases > 0`. |
| Combine | `expected_cases = calibrated_gate_probability × count_prediction` |

There is no hard cutoff that zeros every hexagon under a probability threshold. Ranking uses the continuous score.

**Risk tiers** in notebook `06` are a simple review rule on top of that score (High / Medium / Low). They are for the map, not a clinical diagnosis.

<a id="geocoding"></a>

### Geocoding (real registers without GPS)

Nikshay-style exports usually have a facility hierarchy and a free-text address, not coordinates. Production geocodes **once per health facility**, then assigns that centroid's H3 cell to every patient at the facility.

```text
facility name + TB unit + district + state
        │
        ▼
 Google Maps Geocoding API   (or Nominatim)
        │
        ▼
 lat, lng  →  H3 cell
        │
        ▼
 patients.csv  +  hexagons.csv
```

| Provider | Setup | When to use |
| -------- | ----- | ----------- |
| `google` | `export GOOGLE_MAPS_API_KEY=...` and `pip install googlemaps` | Real programmes. Same provider as V1 ingestion. |
| `nominatim` | `pip install geopy` | Free demos. About one request per second. |
| Manual | Fill `lat` / `lng` in `facility_centroids.csv` | Offline correction. Re-run only fills missing coordinates. |

```python
import os
os.environ["GOOGLE_MAPS_API_KEY"] = "YOUR_KEY"   # never commit this

facilities = geocode_facility_table(facilities, provider="google")
# facilities = geocode_facility_table(facilities, provider="nominatim")

patients_final = assign_h3_from_facility_table(patients_raw, facilities, resolution=10)
hexagons_final = build_hexagon_table(patients_final)
```

If each patient already has GPS, skip geocoding and use `assign_h3_from_latlon`.

**APIs for a full real-data run**

| Capability | What you need | Required? |
| ---------- | ------------- | --------- |
| Facility / address → lat, lng → H3 | Google Maps Geocoding API key | Yes, if the register has no GPS |
| Free geocode fallback | Nominatim via `geopy` | Optional |
| Hexagon map in the browser | `pip install folium` | Optional; notebook `06` |
| Landmark labels and AI explanations on the product map | Platform enrichment | Not in this repository |
| Hosted models and multi-user projects | Platform | Not in this repository |

Create the key in [Google Cloud Console](https://console.cloud.google.com/) with the **Geocoding API** enabled. Google bills this API. Keep the key in your shell or notebook session only.

<br>

## 📤 Outputs and How to Read Them

Notebooks write under `artifacts/` (gitignored). Nothing is uploaded.

| File | Contents |
| ---- | -------- |
| `features.csv` | One row per hexagon × month, before the future label |
| `labeled.csv` | Same rows with `target_cases` for month M+1. Last month dropped. |
| `train.csv` / `test.csv` | Earlier months vs the last held-out months |
| `feature_cols.csv` | Numeric inputs the model is allowed to see |
| `artifacts/models/` | Gate, Poisson count, isotonic calibrator, single-stage baseline |
| `recommendations.csv` | Top-K hexes for the latest feature month |
| `dashboard_recommendations.json` | Rank, H3 id, score, tier, rationale, lat/lng |
| `hotspot_map.html` | Interactive hexagon map. Open it in a browser. |

**Read the ranking before the exact count.** Precision@K is the share of the top K hexes that actually had a case. Efficiency@K is the share of all cases captured in those K hexes. Lift@K compares that capture with visiting K hexes at random.

The map colors match the tier column. A High tier means the score rule fired. It does not mean a team must visit, and it does not replace programme guidelines.

<br>

## 🏗️ Using Your Own Register

1. Keep the CSV **outside git**. `sample_data/*` is ignored except this README.
2. In notebook `00`, set `USE_REAL_DATA = True` and `REAL_REGISTER_PATH`.
3. Run the ingest cells: load → standardize → facility table → geocode → H3 → `patients.csv` and `hexagons.csv`.
4. If you need every cell in the district, replace `hexagons.csv` with your GIS hex list (same `h3_index` values as the patients).
5. Run notebooks `01` → `06` with `DATA_DIR` pointing at that folder.
6. Review `artifacts/hotspot_map.html` and the Precision@K table with the programme team.

> [!NOTE]
> Do not commit notification registers, facility centroid files filled from real programmes, API keys, or trained models fit on patient data.

<br>
<br>

# 🧑‍💻 Developer Reference

## 🐍 Python API

Generate data and build the feature table without opening Jupyter:

```python
from pipeline.sample_data import generate_synthetic_dataset
from pipeline.features import build_feature_table
from pipeline.targets import create_future_target
from pipeline.dataset import prepare_train_test, get_feature_columns
from pipeline.training import train_two_stage_models, predict_two_stage

patients, hexagons, contacts = generate_synthetic_dataset(n_hex=300, n_months=20)
features = build_feature_table(patients, hexagons, contacts)
labeled = create_future_target(features, patients)
train_df, test_df = prepare_train_test(labeled, test_months=2)
cols = get_feature_columns(train_df)
bundle = train_two_stage_models(train_df, cols)
pred, gate = predict_two_stage(
    bundle["stage1"], bundle["stage2"], test_df[cols].fillna(0), bundle["calibrator"]
)
```

## 📚 Documentation Map

| Document | For |
| -------- | --- |
| This README | Install, run order, inputs, geocoding, outputs |
| [docs/methodology.md](docs/methodology.md) | Target shift, feature families, hurdle model, known limits |
| [CONTRIBUTING.md](CONTRIBUTING.md) | How to send a change |
| [SECURITY.md](SECURITY.md) | How to report a security bug in private |
| [sample_data/README.md](sample_data/README.md) | What the synthetic CSVs are, and what not to commit |
| Notebook `00` | Column contract and real-register ingest |
| Notebook `05` | How to read AUC, RMSE, and Precision@K against baselines |

## 📁 Repository Layout

```text
notebooks/          Run 00 → 06 in order
pipeline/           Features, targets, training, ingest, hexagon map
sample_data/        Synthetic CSVs only (generated locally)
artifacts/          Local outputs (gitignored)
docs/methodology.md Technical write-up
utils.py            Local CSV helpers and project paths
requirements.txt
LICENSE             Apache-2.0
NOTICE              Copyright attribution
CONTRIBUTING.md     How to send a change
CODE_OF_CONDUCT.md  Community expectations
SECURITY.md         Private vulnerability reports
```

<br>
<br>

# 🧭 Responsible Interpretation

Ranked hexagons are **analytical evidence for planning**, not an instruction to visit and not a diagnosis.

Before acting on a list, confirm the hexagon resolution, the months held out for testing, geocoding coverage, and whether quiet cells are actually in `hexagons.csv`. A model trained only on cells that already had cases will miss new foci. A facility centroid is not a patient's home. National TB guidelines and local clinical judgment stay in charge.

<br>
<br>

# 🌐 Contributions

We welcome contributions of every size, from a clearer notebook note to a better baseline. Contributions are licensed under the Apache License 2.0. See [CONTRIBUTING.md](CONTRIBUTING.md). Community expectations are in the [Code of Conduct](CODE_OF_CONDUCT.md). Report security issues privately using [SECURITY.md](SECURITY.md).

1. **Fork** the repository and create a feature branch (`git checkout -b feature/my-improvement`).
2. **Open a pull request** describing what changed and why.
3. **Froncort members review** the pull request and merge it into `main`.

Found a bug or have an idea? [Open an issue](https://github.com/Froncort/AI-ACF-open-source/issues).

Do not include real patient rows, credentials, or `.env` files in a pull request.

## 🦫 Contributors

**Froncort AI tech team**

<br>
<br>

# 💭 Join Our Community

<p align="center">
  <a href="https://www.froncort.ai/" target="_blank"><img src="https://img.shields.io/badge/Website-froncort.ai-A30050?style=for-the-badge&logo=googlechrome&logoColor=white" alt="Froncort website" /></a>&nbsp;
  <a href="https://www.linkedin.com/company/froncortai/" target="_blank"><img src="https://img.shields.io/badge/LinkedIn-Froncort-A30050?style=for-the-badge&logo=linkedin&logoColor=white" alt="Froncort on LinkedIn" /></a>&nbsp;
  <a href="https://github.com/Froncort" target="_blank"><img src="https://img.shields.io/badge/GitHub-Froncort-A30050?style=for-the-badge&logo=github&logoColor=white" alt="Froncort on GitHub" /></a>
</p>

<br>
<br>

# 📜 License

Copyright 2026 Froncort.

**AI-ACF Open Source** is released under the [Apache License, Version 2.0](LICENSE). See [NOTICE](NOTICE) for attribution.

**AI-NETRA ACF** (live maps, programme ingestion, enrichment, shared projects, and production deployment) is a separate product from Froncort.ai and is **not** fully contained in this repository. See the [feature comparison](#ai-acf-platform) or [contact Froncort](mailto:info@froncort.ai) for access.

<br>

<p align="center">
  <a href="https://www.froncort.ai/" target="_blank">
    <img src="https://img.shields.io/badge/Made%20by-Froncort.AI-A30050?style=for-the-badge" alt="Made by Froncort.AI" />
  </a>
  <br>
  <sub>Part of the <a href="https://www.froncort.ai/ai-netra">AI-NETRA</a> family of public-health intelligence tools.</sub>
</p>
