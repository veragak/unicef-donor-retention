# Intervention design: from a failed message to "Proof of Impact"

This note records how we got to the Proof of Impact intervention. It covers
the iteration that the code in `src/` produces. The final report later refined
the simulation (see [the main README](../README.md#results)), so the numbers
here are the earlier, more optimistic ones.

## 1. What the model says drives NPS

XGBoost feature importance (gain) on the full donor base:

| Rank | Feature | Gain | In the fragile segment |
|---|---|---|---|
| 1 | `idx_impact` | 0.215 | low (avg 0.39) |
| 2 | `idx_trust` | 0.174 | low (avg 0.23) |
| 3 | `n_yrs_active` | 0.171 | ≤ 2 years by definition |
| 4 | `tot25_medium` | 0.123 | |
| 5 | `perc_breadth` | 0.110 | |
| 6 | `alignment_pct` | 0.100 | already high (avg 94%) |
| 7 | `idx_capacity` | 0.074 | above median by definition |
| 8 | `info_spending_more` | 0.033 | |

The two strongest levers, trust and perceived impact, are exactly where
fragile donors score lowest. Alignment, which we originally expected to be the
problem, is already close to its ceiling.

## 2. The first intervention failed

Our first message was a generic "home visit with photos and impact reports"
covering all of UNICEF's work. Simulated on the fragile donors it gave:

- trust +0.35 and impact +0.26, **but** alignment −0.81
- mean predicted NPS **−0.25**; 62% of donors got worse

By showing donors everything UNICEF does, the message made them feel UNICEF
was working on things they don't care about, even though 94% of them were
already well aligned. The alignment drop wiped out the gains in trust and
impact.

## 3. Segmenting by alignment

| Segment | Donors | Avg alignment | Avg trust | Avg impact |
|---|---|---|---|---|
| High (> 50%) | 143 (94%) | 94.4% | 0.26 | 0.40 |
| Medium (30–50%) | 6 (4%) | 41.7% | 0.17 | 0.00 |
| Low (< 30%) | 3 (2%) | 0.0% | 0.00 | 0.17 |

Almost all fragile donors already care about what UNICEF does. They don't
need to be redirected; they need evidence that their own money works.

## 4. The Proof of Impact message

Personalised per donor (amounts scale with their donation; example for €50):

> UNICEF confirms your priorities align 94% with their focus. Because you're
> already well-aligned, they show you PROOF of your €50's impact:
>
> - €17.50 funded 0.3 teachers' salaries + school supplies for 4 children
>   (photo: classroom with "Funded by donors like you" sign)
> - €14.00 vaccinated 8 children against measles
>   (vaccination records from the clinic, your donation ID visible)
> - €11.00 maintained 1 water well for 3 months
>   (GPS coordinates + date-stamped photo of the well)
> - €7.50 overhead (field staff, logistics, monitoring)
>
> You receive a dashboard link to track these programmes with monthly updates.

Design principles:

1. Confirm the existing alignment instead of trying to change it.
2. Break the donor's own amount down into concrete outputs.
3. Back every claim with verifiable proof (photos, GPS, records).
4. Be honest about overhead.
5. Offer ongoing transparency through a dashboard.

## 5. Simulated result (this code)

| Metric | Generic message | Proof of Impact |
|---|---|---|
| Mean Δ NPS | −0.25 | **+1.01** |
| Donors improved | 38% | 97.9% |
| Donors worsened | 62% | 2.1% |
| Δ trust | +0.35 | +0.47 |
| Δ impact | +0.26 | +0.44 |
| Δ alignment | −0.81 | −0.02 |

170 of 176 LLM calls succeeded; 146 donors had complete features for scoring.

## 6. What it would take to run this for real

- **Data:** link donations to specific projects (donation ID), record project
  GPS locations, collect dated photos and anonymised service records.
- **Platform:** a secure, mobile-friendly donor dashboard with a per-donor
  breakdown and monthly updates.
- **Process:** map each donor's priorities to programme areas, generate the
  proof package, send the email with the dashboard link.
- **Pilot:** 50 fragile donors for 3 months, NPS surveyed before and after,
  against a control group. Scale to the whole segment only if more than 70%
  improve.

## Caveats

These are model-on-model results: an LLM guesses how perceptions change, and
the XGBoost model turns that into a predicted NPS. Neither step has been
checked against real donor behaviour. The fidelity audit in the final report
found the LLM inflates trust and impact (see the main README), and the
refined run that held alignment fixed gave a smaller effect (+0.86). The
simulation is useful for ranking ideas worth testing, not as a forecast.
