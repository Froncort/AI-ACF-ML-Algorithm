# Sample data

This folder holds **synthetic** CSVs only (`patients.csv`, `hexagons.csv`, `contacts.csv`).

## What each file is

| File | Meaning |
| --- | --- |
| `patients.csv` | Synthetic TB cases with an `h3_index` already assigned |
| `hexagons.csv` | **List of H3 cells for the demo AOI / region** (not the whole globe) — every cell the model may score, including quiet months |
| `contacts.csv` | Optional contact-tracing counts per hex × month |

Generate them with:

```bash
python -c "from pipeline.sample_data import generate_synthetic_dataset; generate_synthetic_dataset(n_hex=300, n_months=20)"
```

or by running notebook `00_data_contract_and_sample.ipynb`.

For a **real district**, build `hexagons.csv` from your AOI boundary (all H3 cells covering that region) or from `build_hexagon_table()` after Google Maps facility geocoding — see the root README.

Do **not** place real patient extracts here if you plan to publish or push this repository.
