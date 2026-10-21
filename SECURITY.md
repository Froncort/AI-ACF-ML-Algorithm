# Security

AI-ACF Open Source is developed by Froncort and released under the [Apache License, Version 2.0](LICENSE). Do not send vulnerability reports to `security@apache.org`. Froncort handles them.

The license does not include a warranty. Section 7 states that the software is provided on an "AS IS" basis. If you find a vulnerability, report it so it can be fixed.

## How to report

Report privately. Do not open a public GitHub issue.

- Use **Report a vulnerability** on the Security tab of this repository, or
- Email [info@froncort.ai](mailto:info@froncort.ai) with `[SECURITY]` in the subject line.

Include the version or commit, the steps to reproduce the issue, and the impact. Use synthetic data only. Do not send patient or confidential programme data.

We will acknowledge a report within 7 business days. Please wait until a fix is released before you publish details.

## What is in scope

These notebooks run locally. There is no account system and no hosted service in this repository. Reports should be about that use.

In scope:

- Code execution or unsafe loading of a crafted patient CSV, hexagon file, or notebook input.
- Path traversal when reading inputs or writing `artifacts/`.
- A vulnerability in a dependency listed in `requirements.txt` that this project can reach.
- The optional geocoding helpers sending more than the facility query you chose to send.

Out of scope:

- Putting Jupyter on a public network.
- A very large training run that only exhausts CPU or memory on the machine running it.
- An attack that already requires access to the operator's computer.
- Scanning Froncort websites or the commercial AI-NETRA platform.
- A missing or invalid Google Maps API key. That is a configuration issue, not a vulnerability in this repository.
