# Predicting and Improving Donor Retention at UNICEF Netherlands

Group project for **Seminar Case Studies (FEM11153)**, Erasmus University
Rotterdam, in collaboration with UNICEF Netherlands.

**Team 5:** Aris, Francesco, Ved and Vera

We built a model of what drives donor loyalty, found the group of new donors
most at risk of leaving, and then used an LLM to simulate how those donors
would react to different emails before anyone sends one to a real person.

> **No data in this repository.** The donor data is confidential UNICEF data.
> Only code and documentation are published. See [`data/README.md`](data/README.md).

---

## Why this project exists

UNICEF Netherlands runs on recurring donations. Recruiting a donor costs money
up front, and that cost is only earned back if the donor keeps giving for years.
A donor who stops in the first year or two is a loss.

The donor data showed that instability is concentrated in those first years:
short-tenure donors have lower Net Promoter Scores (NPS) and stop giving more
often. So the question UNICEF cares about is not "how do we make everyone a bit
happier?" but:

> **Which new donors are at risk of leaving, why, and what message would
> actually keep them?**

There is a second constraint. UNICEF cannot keep surveying donors or running
message experiments on them; every extra email risks annoying the very people
it is trying to keep. That is why we added a simulation step: test ideas on
*synthetic* copies of real donors first, and only take the promising ones to a
real pilot.

## The approach

```
 Donor survey + background data (6,073 donors)
            │
            ▼
 1. Predict NPS ── linear regression vs. XGBoost ──► what drives loyalty?
            │
            ▼
    Select "fragile" donors: tenure ≤ 2 yrs, capacity > median, NPS ≤ 8  (176)
            │
            ▼
 2. Turn each fragile donor into a written persona (LLM prompt)
            │
            ▼
 3. Show each persona an intervention email ──► LLM (Qwen3-32B via Groq)
    returns the donor's new trust, impact, alignment, ...
            │
            ▼
 4. Re-score every donor with the same XGBoost model, before vs. after
            │
            ▼
    Δ predicted NPS per intervention
```

### 1. Predicting NPS — [`src/01_nps_model`](src/01_nps_model/nps_model_training.R)

We cleaned and merged the survey (Dataset C) with background data (Dataset D)
and built eight predictors: tenure, donation level, a financial-capacity index
(PCA of income, education and property value), a trust index, a perceived
impact index, how many UNICEF programmes the donor knows, whether they want
more information on spending, and how well their priorities match what they
think UNICEF does. Full definitions are in [`data/README.md`](data/README.md).

We compared four models on an 80/20 train/test split (4,780 complete cases):

| Model | R² (test) | RMSE (test) |
|---|---|---|
| Naive mean baseline | 0.000 | 1.66 |
| Linear, 5 core predictors | 0.101 | 1.58 |
| Linear, + 3 perception predictors | 0.123 | 1.56 |
| **XGBoost** (depth 3, η 0.05, subsample 1.0, 100 rounds) | **0.182** | **1.52** |

*Why XGBoost?* Trust, impact and tenure interact in ways we couldn't specify
in advance; boosted trees pick that up on their own. We kept the linear model
because its coefficients are easy to explain to UNICEF. An R² of 18% is modest,
which is normal for survey data on personal attitudes. The model is good for
ranking donors and comparing scenarios, not for exact individual predictions.

### Who is "fragile", and why those donors?

Donors with **≤ 2 years tenure**, **above-median financial capacity** and
**NPS ≤ 8**. They are in the period when people decide whether to stay, they
are financially the most valuable, and there is still room to move them.
Targeting donors who would stay anyway, or who can't be won back, wastes
effort (Ascarza, 2018).

### 2. Synthetic donors — [`src/02_synthetic_donors`](src/02_synthetic_donors/synthetic_donor_prompts.ipynb)

Instead of generating random fake donors (which produced unrealistic
combinations of bounded indices), each synthetic donor is **anchored to one real
fragile donor**: their tenure, capacity, trust, impact, alignment, awareness and
their own survey comments are written out as a first-person persona prompt.

### 3. Simulating interventions — [`src/03_intervention_simulation`](src/03_intervention_simulation)

Each persona is shown an email and the LLM answers, in JSON, how that donor's
perceptions would change. Every donor is an independent call with no shared
history, at temperature 0.

| Notebook | What it tests |
|---|---|
| `simulation_intervention.ipynb` | **Proof of Impact**: personalised breakdown of the donor's own euros, with photos, GPS, records and a dashboard |
| `intervention_simulator_SEGMENTED.ipynb` | Different messages per alignment segment; the high-alignment branch holds the **neutral GDPR/privacy notice** used as a control |
| `final_intervention_simulator_COMPLETE.ipynb` | General harness with **donor fatigue** (simulated contact history) and **severity checks** that correct the LLM when it under-reacts to a scandal or a breakthrough. Swap `INTERVENTION_TEMPLATE` to test any scenario |

### 4. Scoring the outcome — [`src/04_post_intervention_scoring`](src/04_post_intervention_scoring/post_intervention_scoring.R)

The trained XGBoost model scores every fragile donor twice: once with their
real features and once with the simulated post-intervention perceptions. The
difference is the predicted NPS effect. The model is never retrained, so any
difference comes only from the changed perceptions.

### 5. Checking the simulation — [`src/05_fidelity_audit`](src/05_fidelity_audit)

An LLM pretending to be a donor is only useful if it behaves like the real
donor. This step runs 5 independent draws per donor and compares the simulated
trust and impact with the real fragile donors: are the distributions similar,
is there enough variety, and do tenure and capacity relate to attitudes the
same way? The results are in [Limitations](#limitations).

## Results

**What drives loyalty.** Perceived impact (gain 0.215) and trust (0.174) are
the strongest predictors, followed by tenure (0.171) and donation level
(0.123). Financial capacity is not significant once attitudes are included.
**Donor fragility is psychological, not financial.**

**The surprising part.** 94% of fragile donors are *already* aligned with
UNICEF's priorities. They don't doubt the mission; they doubt that *their*
money makes a difference. Our first, generic "here's everything UNICEF does"
message backfired in simulation (−0.25 NPS) because it made donors feel UNICEF
was busy with things they don't care about. The full story is in
[`docs/intervention_strategy.md`](docs/intervention_strategy.md).

**Intervention comparison** (final report, 176 fragile donors; only trust and
impact allowed to change, alignment held fixed):

| Scenario | Δ predicted NPS |
|---|---|
| Personalised **Proof of Impact** | **+0.86** |
| Neutral administrative (GDPR) notice | +0.06 |
| Negative disclosure (audit finds high executive bonuses) | −1.44 |

The positive, neutral and negative scenarios come out in the expected order,
and bad news hits about twice as hard as good news helps, which matches the
trust literature. An earlier run with this repository's code, where alignment
and awareness could also move, gave +1.01 for Proof of Impact.

## Limitations

- **Not causal.** The data is cross-sectional (each donor observed once). Long-
  tenure donors are the ones who already stayed, so tenure effects partly
  reflect survivorship.
- **NPS is a proxy** for leaving, not observed churn.
- **The LLM is optimistic.** A fidelity audit against the real fragile donors
  found the simulated trust (0.72 vs. 0.23) and impact (0.83 vs. 0.38) far above
  the human values (Wasserstein distance 0.49 and 0.44, above the 0.30 limit).
  Variance did not collapse, and the tenure × trust pattern was reproduced, but
  tenure × impact was not.
- So the **ranking of interventions is the useful output**; the exact numbers
  are directional at best.

## Recommendations for UNICEF Netherlands

1. **Protect the first two years.** Flag high-capacity donors at sign-up and
   start impact-focused communication early.
2. **Replace storytelling with proof.** Show each donor what *their* amount did,
   with photos, locations and a spending breakdown.
3. **Build a personal impact dashboard**, and explain clearly that it is secure
   and private.
4. **Validate with a real A/B test** (e.g. 50 fragile donors vs. a control group
   for 3 months) before rolling out.

## Running the code

The code needs the confidential data, so it will only run for people with
authorised access.

**Requirements:** R with `tidyverse`, `readxl`, `janitor`, `xgboost`, `caret`,
`car`; Python 3.10+ with `pip install -r requirements.txt`; a
[Groq](https://console.groq.com) API key.

1. Put `Merged_dataset_CD.xlsx` in the working directory and run
   `src/01_nps_model/nps_model_training.R`. Keep the R session open: step 4
   uses the trained `model_xgb` object.
2. Build the fragile segment (`fragile_group_donors.csv`, see definition above)
   and run `src/02_synthetic_donors/synthetic_donor_prompts.ipynb` to create
   `fragile_donors_with_prompts.csv`.
3. Set your key (`export GROQ_API_KEY=...`, see `.env.example`) and run one of
   the notebooks in `src/03_intervention_simulation`.
4. In the same R session as step 1, run
   `src/04_post_intervention_scoring/post_intervention_scoring.R`. Results go
   to `outputs/`, which is git-ignored.
5. Optionally, run the fidelity audit in `src/05_fidelity_audit` (see its README).

## Repository structure

```
├── README.md                       ← you are here
├── docs/
│   └── intervention_strategy.md    ← how the Proof of Impact message was designed
├── data/
│   └── README.md                   ← what data is needed and why it is not here
├── src/
│   ├── 01_nps_model/               ← NPS prediction (linear vs. XGBoost)
│   ├── 02_synthetic_donors/        ← persona prompts from real donor records
│   ├── 03_intervention_simulation/ ← LLM simulation of donor reactions
│   ├── 04_post_intervention_scoring/ ← before/after NPS with XGBoost
│   └── 05_fidelity_audit/          ← does the LLM behave like real donors?
├── requirements.txt
├── .env.example
└── .gitignore                      ← blocks every data file and secrets
```

## Key references

- Anik, L., Aknin, L. B., Norton, M. I., & Dunn, E. W. (2013). Feeling good about giving: The benefits (and costs) of self-interested charitable behavior. *Journal of Public Policy & Marketing*, 32(1), 64–75.
- Argyle, L. P., Busby, E. C., Fulda, N., Gubler, J. R., Rytting, C., & Wingate, D. (2023). Out of one, many: Using language models to simulate human samples. *Political Analysis*.
- Ascarza, E. (2018). Retention futility: Targeting high-risk customers might be ineffective. *Journal of Marketing Research*, 55(1), 80–98.
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD '16*.
- Karlan, D., & List, J. A. (2007). Does price matter in charitable giving? Evidence from a large-scale natural field experiment. *American Economic Review*, 97(5), 1774–1793.
- Morgan, R. M., & Hunt, S. D. (1994). The commitment–trust theory of relationship marketing. *Journal of Marketing*, 58(3), 20–38.
- Naskrent, J., & Siebelt, P. (2011). The influence of commitment, trust, satisfaction, and involvement on donor retention. *Nonprofit and Voluntary Sector Quarterly*, 40(4), 740–760.
- Park, J. S., et al. (2024). Generative agents: Interactive simulacra of human behavior. *Communications of the ACM*, 67(2), 70–80.
- Reichheld, F. F. (2003). The one number you need to grow. *Harvard Business Review*, 81(12), 46–54.
- Sargeant, A. (2001). Relationship fundraising: How to keep donors loyal. *Nonprofit Management & Leadership*, 12(2), 177–192.

The full list is in our final report (not included in this repository).
