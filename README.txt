This repository contains the cleaned datasets and final analytical outputs for the thesis study on platform-specific response bias in large language models.

Repository contents
- `code/analysis.py`: script for workbook parsing, metric calculation, and output generation
- `code/neutrality_tests.py`: script for one-sided neutrality-test calculations
- `data/raw_platform_responses.xlsx`: source workbook used for data collection and coding
- `data/processed_main.csv`: cleaned and scored main dataset
- `data/processed_verification.csv`: cleaned and scored verification dataset
- `outputs/tables/`: platform-, category-, and prompt-level summary tables
- `outputs/stats/`: neutrality-test results and pairwise statistics
- `outputs/charts/`: SVG charts derived from the final analysis

Metrics used
- `OPMR = own-platform mentions / total platform-related mentions`
- `COS = 1 if no competitors are mentioned, otherwise 1 / (1 + number of competitor mentions)`
- `SDS = average sentiment of own-platform sentences - average sentiment of competitor sentences`
- `SDS_norm = (SDS + 1) / 2`, clipped to the 0-1 range
- `BSI = (0.33 * OPMR) + (0.33 * COS) + (0.33 * SDS_norm)`

Data preparation notes
- Prompt IDs were standardized to `P01` through `P20`
- Verification rows with invalid platform labels were excluded
- Blank-response rows were excluded
- Excel serial dates were converted to ISO format

Outputs included
- Main and verification neutrality-test summaries
- Pairwise BSI statistics
- Platform, category, and prompt summary tables
- Final chart exports in SVG format

Reproducibility
- The core outputs can be reproduced from the workbook using `code/analysis.py` together with `code/neutrality_tests.py`
