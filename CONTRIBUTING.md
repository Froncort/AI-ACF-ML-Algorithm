# Contributing

AI-ACF Open Source is developed by Froncort and released under the [Apache License, Version 2.0](LICENSE).
By participating, you agree to the [Code of Conduct](CODE_OF_CONDUCT.md).

## License of your contribution

Any contribution you intentionally submit for inclusion in this repository is licensed under the Apache License, Version 2.0, without extra terms. That is section 5 of the license. You keep the copyright in your work. New source files should use the same Apache License 2.0 header as the existing Python modules.

Do not submit code you cannot license this way.

## What to contribute

Changes to the local notebooks, feature pipeline, two-stage hurdle model, evaluation, hexagon map, ingestion helpers, docs, and synthetic sample data are in scope.

Live programme maps, register ingestion at scale, landmark enrichment, shared projects, cloud storage, and production APIs belong to the separate AI-NETRA ACF platform. Do not add them here.

Do not include real patient data, facility registers, API keys, or confidential programme data. Use the synthetic generator in `pipeline/sample_data.py`.

## How to contribute

1. Fork the repository and create a feature branch from `main`.
   ```bash
   git checkout -b feature/my-improvement
   ```
2. Make the change. Before you open the pull request, install dependencies and run the synthetic notebooks in order (`00` through `06`) with `USE_REAL_DATA = False`.
   ```bash
   pip install -r requirements.txt
   ```
3. Open a pull request that says what changed and why.
4. Froncort members review the pull request.
5. A Froncort member merges it into `main` after review.

Security issues follow [SECURITY.md](SECURITY.md). Do not report them in a public issue or pull request.
