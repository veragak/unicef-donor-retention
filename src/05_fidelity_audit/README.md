# Step 5: Fidelity audit of the synthetic donors

Can we trust the LLM's answers? This step tests whether the simulated donors
behave like the real fragile donors they are copied from, following Argyle et
al. (2023) and the guardrails in Madden (2025) and Amirova et al. (2024).
Written by Ved.

| Script | What it does | Reads → writes |
|---|---|---|
| `phase1_prompt_generation.py` | First-person persona prompts (same as step 2) plus a template for a demographics-only calibration check (not implemented) | `fragile_group_donors.csv` → `fragile_donors_with_prompts.csv` |
| `phase1_prompt_generation_unblinded.py` | Alternative third-person profiles with baseline trust and impact kept out of the persona, so the LLM is not anchored on them (Gui & Toubia). Not wired into phase 2 | `fragile_group_donors.csv` → `fragile_donors_unblinded.csv` |
| `phase2_ensemble_simulation.py` | 5 independent draws per donor at temperature 0.7 (880 calls), averaged per donor, with cost tracking | `fragile_donors_with_prompts.csv` → `ensemble_intervention_results.csv`, `ensemble_details_all_draws.csv` |
| `phase3_fidelity_report.py` | Compares simulated vs. real trust and impact: distribution shift (bootstrapped KL divergence), variance collapse (SD ratio < 0.7), and whether tenure/capacity/age relate to attitudes the same way (Cramér's V, Δ < 0.10). Runs locally, no API | ensemble results → `fidelity_report_*.txt/.csv` |

Run in order: `phase1_prompt_generation.py` → `phase2_ensemble_simulation.py`
(needs `GROQ_API_KEY`) → `phase3_fidelity_report.py`. All inputs and outputs
contain donor data and are git-ignored.

**Note.** This script measures distribution shift with KL divergence. The
fidelity table in our final report uses Wasserstein distance (trust 0.494,
impact 0.444), so those numbers came from a later variant of this audit. The
conclusions are the same: the LLM overestimates trust and impact, variance does
not collapse, the tenure × trust pattern is reproduced and tenure × impact is not.
