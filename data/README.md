# Data (not included)

The data behind this project belongs to **UNICEF Netherlands** and was shared
with us under a confidentiality agreement for the Erasmus University Rotterdam
course *Seminar Case Studies (FEM11153)*. It contains real donor records,
including survey answers and free-text comments, so **none of it is published
here**. The `.gitignore` blocks every data format (`.csv`, `.xlsx`, `.jsonl`,
...) to keep it that way.

If you have authorised access, place the files below in the working directory
of the script that reads them.

| File | Used by | Contents |
|---|---|---|
| `Merged_dataset_CD.xlsx` | `src/01_nps_model` | Dataset C (donor survey) merged with Dataset D (background data): 6,073 donors × 71 variables, one row per donor, cross-sectional |
| `fragile_group_donors.csv` | `src/02_synthetic_donors`, `src/04_post_intervention_scoring` | The 176 "fragile" donors with the engineered features below |
| `fragile_donors_with_prompts.csv` | `src/03_intervention_simulation` | Output of step 2: fragile donors + their LLM conditioning prompt |
| `simulated_intervention_results_*.csv` | `src/04_post_intervention_scoring` | Output of step 3: simulated post-intervention perceptions |

## Variables used in the model

| Variable | Meaning | How it is built |
|---|---|---|
| `nps` | Net Promoter Score, 0–10 (outcome) | Survey. ≤ 8 = at risk, ≥ 9 = promoter |
| `n_yrs_active` | Tenure in years | Background data |
| `tot25_medium` | Donation level | Background data |
| `idx_capacity` | Financial capacity | First principal component of income, education and property value (WOZ), standardised. Explains 84.3% of their variance |
| `idx_trust` | Trust in UNICEF | Mean of "I appreciate the way UNICEF works" and "UNICEF's work is vital" |
| `idx_impact` | Perceived impact | Mean of "UNICEF really makes a difference for children" and "it is right to support good causes and UNICEF is one of them" |
| `perc_breadth` | Awareness of UNICEF's work (0–8) | Number of the 8 programme areas the donor says UNICEF is involved in |
| `info_spending_more` | Wants more info on spending (0/1) | 1 if the answer to "how we spend your donations" contains "more" |
| `alignment_pct` | Priority–perception alignment (0–1) | Share of the donor's priority areas that they also see UNICEF working on |

Perception items are coded *Yes = 1, No = 0, Don't know = 0.5*. Rows with a
missing value in any model variable are dropped (4,780 of 6,073 donors remain).

**Fragile donor segment:** tenure ≤ 2 years, `idx_capacity` above the median,
and NPS ≤ 8 → 176 donors.
