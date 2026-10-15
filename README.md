# FIFA World Cup 2026 - Objective 2 (Linear Regression)

Group project, Assessment 3.

## Files
- `matches.csv` - raw match-level data of 104 matches
- `build_datasets.py` - wrangling: builds both regression datasets from `matches.csv` (+ FIFA ranking of 11 June 2026, hard-coded in the script)
- `lr21_match_dataset.csv` - Model 2.1 dataset (104 rows, target `goal_diff`, 8 explanatory variables)
- `lr22_teammatch_dataset.csv` - Model 2.2 dataset (208 rows, target `goals_scored`, 8 explanatory variables)
- `analysis_regression.py` - OLS models, cross-validation, diagnostics, figures
- `results.json`, `fig_models.png` - outputs used in the report

## How to run
```
pip install -r requirements.txt
python build_datasets.py
python analysis_regression.py
```
Run both scripts from this folder. `match_id`, team names are ID fields only.
