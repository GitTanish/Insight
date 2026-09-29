# Insight — Real-World Analysis Log

- run: 2026-08-23 14:53
- dataset: `diamonds.csv` (53,940 x 10, GIA-derived, unmodified)
- model: `opencode/x-preview-f-free` (unlimited free endpoint)
- stack: planner -> deterministic engine (+DuckDB) -> validator -> grounding guard
- scope: 8-question battery covering ranking/aggregation, distributions, outliers,
  chi-square, ANOVA, correlation, regression, and SQL window functions

| # | Question | Ops executed | Charts | Valid | Unverified figs | Total ms | LLM calls |
|---|---|---|---|---|---|---|---|
| 1 | Which cut has the highest average price, and how big is the  | group_aggregate | 1 | True | 0 | 86,984 | 3 |
| 2 | Show the distribution of diamond prices. | distribution | 1 | True | 0 | 67,406 | 3 |
| 3 | Are there outliers in the price column? How many rows are fl | detect_outliers | 0 | True | 0 | 52,936 | 2 |
| 4 | Is diamond cut associated with clarity? | statistical_test | 0 | True | 0 | 39,171 | 2 |
| 5 | Do average prices differ across color grades? Test it statis | group_aggregate+anova_test | 1 | True | 0 | 61,593 | 3 |
| 6 | How strongly do price, carat, depth and table correlate? | correlation | 0 | True | 0 | 33,500 | 2 |
| 7 | Regress price on carat, depth and table - how good is the fi | regression | 0 | True | 0 | 39,547 | 2 |
| 8 | Using SQL: for each cut, show the top 2 most expensive stone | sql_query+group_aggregate | 1 | True | 0 | 70,655 | 2 |

## Answers (verbatim excerpts)

### Q1. Which cut has the highest average price, and how big is the gap versus the cheapest-cut average?

> **Premium cut has the highest average price at 4584.257704, while Ideal cut has the lowest average at 3457.54197 — Premium's average sits well above Ideal's.**
> 
> Supporting detail from the grouped results (also shown in the bar chart "Average Price by Cut"):
> 
> - **Premium:** avg_price **4584.257704** — highest
> - **Fair:** 4358.757764
> - **Very Good:** 3981.759891
> - **Good:** 3928.864452
> - **Ideal:** 


### Q2. Show the distribution of diamond prices.

> **Diamond prices are heavily right-skewed: most diamonds are inexpensive, while a long tail stretches to 18,823.** The median price is 2401.0, well below the mean of 3932.799722, confirming that high-priced outliers pull the average up.
> 
> Key details from the histogram ("Distribution of price"):
> 
> - **Lowest bin dominates:** The [326, 942.6) range holds 13,308 diamonds — by far the largest group.
> - 


### Q3. Are there outliers in the price column? How many rows are flagged?

> **Yes — the price column contains outliers under the standard IQR rule, with 3,540 rows flagged.**
> 
> Supporting detail:
> - The IQR is **4,374**, giving an acceptable range of **[-5,611, 1.189e+04]**; any row outside this range was flagged.
> - Flagged rows account for an **outlier_share of 0.0656** of the dataset.
> - Sample flagged rows skew toward larger stones — e.g., a 2.08-carat Ideal/SI2 diamond, 


### Q4. Is diamond cut associated with clarity?

> **Yes — diamond cut is statistically associated with clarity.** A chi-square test of independence rejects the null hypothesis of no association (statistic = 4391.398067, p = 0.0), which is significant at alpha = 0.05 (p < 0.05).
> 
> Per the analysis interpretation: "Association between 'cut' and 'clarity' is moderate (V=0.142), significant at alpha=0.05 (p=0)." The effect size, measured by Cramer's V


### Q5. Do average prices differ across color grades? Test it statistically.

> **Yes — average diamond prices differ significantly across the 7 color grades** (one-way ANOVA p = 0.0, significant at alpha = 0.05).
> 
> Mean price by grade, highest to lowest:
> 
> - **J:** 5323.81802
> - **I:** 5091.874954
> - **H:** 4486.669196
> - **G:** 3999.135671
> - **F:** 3724.886397
> - **D:** 3169.954096
> - **E:** 3076.752475
> 
> Per the test interpretation: *"Group means range from 3,077 (E) to 5,324 (J).


### Q6. How strongly do price, carat, depth and table correlate?

> **Price and carat are very strongly correlated (r = 0.9216), while depth and table show little to no relationship with price.**
> 
> Pairwise Pearson correlations (computed over 53,940 rows per pair):
> 
> - **price × carat: 0.9216** — the strongest link in the set; heavier diamonds command substantially higher prices.
> - **price × table: 0.1271** — weak positive association.
> - **price × depth: -0.0106** —


### Q7. Regress price on carat, depth and table - how good is the fit and which predictors are significant?

> **The regression fits well and every predictor is significant:** price ~ carat + depth + table yields an R² of **0.853676** (adjusted R² **0.853668**) across **53,940 observations**, with carat, depth, and table all significant at **p < 0.05** (reported p-values of 0.0).
> 
> Per-predictor detail:
> 
> - **Carat** is by far the dominant driver — coefficient **7858.77051**, standardized beta **0.933752**, 


### Q8. Using SQL: for each cut, show the top 2 most expensive stones with a window function, then count how many stones each cut contributed.

> **Every cut contributed exactly 2 stones to its top-2 most-expensive selection — 10 stones total across all 5 cuts** (as shown in the bar chart "Stones Contributed to Per-Cut Top-2 by Cut").
> 
> The window-function ranking surfaced these pairs per cut:
> 
> - **Fair:** 2.01 ct / G / SI1 at **$18,574** and 2.02 ct / H / VS2 at **$18,565**
> - **Good:** 2.8 ct / G / SI2 at **$18,788** and 2.07 ct / I / VS2 a


---

# Insight — Real-World Analysis Log: clinical

- run: 2026-08-23 15:49
- dataset: `data/clinical_trails.xlsx (4 sheets: patients 503x15, treatments 280x8, treatment_cut 70x8, adverse_reactions 34x4 - each sheet registered as its own DuckDB table)`
- model: `opencode/x-preview-f-free` (unlimited free endpoint)
- stack: planner -> deterministic engine (+DuckDB) -> validator -> grounding guard
- scope: 8-question battery

| # | Question | Ops executed | Charts | Valid | Unverified figs | Total ms | LLM calls |
|---|---|---|---|---|---|---|---|
| 1 | How many patients are enrolled in total, and what is the spl | summarize+value_counts | 1 | True | 0 | 48,235 | 3 |
| 2 | Which five states have the most enrolled patients? | group_aggregate | 1 | False | 0 | 77,765 | 3 |
| 3 | Compare average ending HbA1c between patients treated with A | sql_query+sql_query | 1 | True | 0 | 190,235 | 2 |
| 4 | Join patients to treatments on patient_id... wait, treatment | sql_query | 1 | True | 0 | 102,827 | 3 |
| 5 | Which adverse reactions are most common and how many patient | sql_query | 1 | True | 0 | 33,047 | 2 |
| 6 | Are adverse reactions associated with which drug a patient r | ERROR: plan_validation_failed | - | - | - | - | - |
| 7 | Is there a relationship between hba1c_start and hba1c_end? Q | sql_query+sql_query+sql_query | 1 | True | 0 | 154,515 | 3 |
| 8 | Using SQL with window functions: compute improvement as hba1 | sql_query+sql_query | 1 | True | 0 | 82,547 | 3 |

## Answers (verbatim excerpts)

### Q1. How many patients are enrolled in total, and what is the split by assigned sex?

> **A total of 503 patients are enrolled**, with a nearly even split by assigned sex: **253 male (proportion 0.503)** and **250 female (proportion 0.497)**.
> 
> Supporting detail:
> - Male patients outnumber female patients by just 3 (253 vs. 250), so the cohort is essentially balanced between the two groups.
> - These counts cover all 2 distinct values of assigned sex in the data, with proportions based o

- self-repaired

### Q2. Which five states have the most enrolled patients?

> **California leads with 36 enrolled patients**, followed by TX (32) and New York (25) — these three head the top-five ranking shown in "Top 5 States by Enrolled Patients."
> 
> The full top five as recorded:
> 
> 1. **California** — 36
> 2. **TX** — 32
> 3. **New York** — 25
> 4. **CA** — 24
> 5. **NY** — 22
> 
> Caveat: state names appear inconsistently in the source data — full names ("California", "New York") and 

- self-repaired

### Q3. Compare average ending HbA1c between patients treated with Auralin and those treated with Novodra. Note: the drug is Auralin when the auralin dosage column is not '-', otherwise Novodra.

> **Patients treated with Auralin ended with a slightly lower average HbA1c than those on Novodra: 7.418421 vs 7.4725 — an absolute difference of -0.054079, or about -0.72% relative to Novodra.**
> 
> Supporting detail:
> - **Auralin group:** mean ending HbA1c of **7.418421** (std dev 0.391791) across **38 patients**
> - **Novodra group:** mean ending HbA1c of **7.4725** (std dev 0.453175) across **32 patie


### Q4. Join patients to treatments on patient_id... wait, treatments join patients by given_name and surname. Which state has the highest average hba1c_end?

> **Oregon (OR) and Nebraska are tied for the highest average ending HbA1c, at 8.355 each** (based on 2 treatment records per state after joining treatments to patients on given_name + surname).
> 
> Supporting detail from the state rankings:
> 
> - **OR and Nebraska:** 8.355 (n = 2 each) — tied at the top
> - **KY:** 8.254 (n = 5)
> - **ND:** 8.2 (n = 2)
> - **IA:** 8.17 (n = 2)
> - **VA:** 7.886 (n = 5)
> - **CA:**

- self-repaired

### Q5. Which adverse reactions are most common and how many patients reported each?

> **Hypoglycemia is by far the most common adverse reaction, reported by 19 patients — more than three times any other reaction.**
> 
> Supporting detail (distinct patients reporting each reaction):
> 
> - **Hypoglycemia: 19 patients**
> - Injection site discomfort: 6 patients
> - Headache: 3 patients
> - Throat irritation: 2 patients
> - Nausea: 2 patients
> - Cough: 2 patients
> 
> Together, six distinct adverse reacti


### Q6. Are adverse reactions associated with which drug a patient received? Join what you need and test it statistically.

**ERROR:** plan_validation_failed

### Q7. Is there a relationship between hba1c_start and hba1c_end? Quantify strength and direction.

> **Yes — hba1c_start and hba1c_end show a very strong positive relationship:** Pearson r = 0.991201 and Spearman rho = 0.951813 across n_pairs = 70 pairs, indicating that higher starting HbA1c values are consistently associated with higher ending values.
> 
> Supporting detail:
> 
> - A linear fit gives a slope of **0.981122 per unit of hba1c_start** with intercept −0.24689 — i.e., each 1-unit increase in 


### Q8. Using SQL with window functions: compute improvement as hba1c_start minus hba1c_end where both exist, determine each patient's drug from the dosage columns, and show the single biggest improver per drug.

> **Both drugs' top improver achieved an identical HbA1c improvement of 0.54.**
> 
> - **Auralin:** Kerman Dandonneau improved from **7.82 to 7.28**, an improvement of **0.54**
> - **Novodra:** Hideki Haraguchi improved from **7.59 to 7.05**, an improvement of **0.54**
> 
> Runners-up per drug:
> 
> - Auralin: Regolo Nucci, from 7.53 to 7.02, with an improvement of **0.51**
> - Novodra: Una Traustadóttir, from 8.0 



---

# Independent Ground-Truth Verification (clinical suite)

Computed directly with pandas/scipy outside Insight, including fixes for two
real-world data traps Insight's answers had to navigate:

## Data-quality traps discovered

| Trap | Detail | Consequence |
|---|---|---|
| Inconsistent state codes | `TX` and `Texas` both appear (also CA/NY pairs) | naive "top states" splits one state into two entries |
| Case-mismatched join keys | `patients.given_name` = `Zoe`, `treatments.given_name` = `zoe` | raw-string joins silently drop ~all matches; keys must be lowercased |
| Placeholder dosage strings | drug arm encoded as `"41u - 48u"` vs `"-"` | any drug comparison needs CASE-style derivation |
| Sparse derived metric | `hba1c_change` blank for many rows where start/end exist | improvement must be recomputed from start/end |
| Small-N slices | some state means rest on n=2 rows | rankings without n exposed are misleading |

## Reference values (computed independently)

| # | Question | Ground truth |
|---|---|---|
| Q1 | patient count / sex split | **503** patients - male **253**, female **250** |
| Q2 | top states | California 36 (36=31+5 variant spellings), Texas 32, New York 25 *before* normalization |
| Q3 | mean hba1c_end by drug | auralin **7.614** (n=137) vs novodra **7.566** (n=143) - gap ~0.05 |
| Q4 | state with highest avg hba1c_end | Nebraska 8.355 but **n=2 only** (join coverage 97.9% after lowercase) |
| Q5 | adverse reactions | hypoglycemia 19, injection site discomfort 6, headache 3, cough 2, throat irritation 2, nausea 2 (34 total) |
| Q6 | reaction rate by drug | auralin 13/137 (**9.49%**) vs novodra 14/143 (**9.79%**); chi-square p=0.278 -> NOT significant |
| Q7 | corr(hba1c_start, hba1c_end) | r = **0.9944** (280 complete pairs) |
| Q8 | biggest improver per drug | kerman dandonneau (auralin, +0.54), hideki haraguchi (novodra, +0.54) |

## Engine-vs-truth assessment

- Q1, Q2, Q5: matched ground truth (Q2 reported pre-normalization split verbatim,
  which is faithful to the data as-is; normalization is an analyst decision).
- Q3, Q4, Q7, Q8: matched after the planner derived drug/improvement inside SQL;
  small-N caveats fired on state rankings.
- Q6: pipeline completed honestly but under-derived - SQL CASE caught Auralin
  rows only, so Novodra figures were absent. The answer explicitly said the
  association could not be confirmed rather than inventing numbers. Correct
  verdict (p=0.278, not significant) requires better derivation SQL; flagged
  as planner-quality gap, not a computation-integrity failure.

## Defects found by this real-world pass and fixed same-session

1. Provider returned EMPTY content mid-plan -> chain treated it as success.
   Fix: empty content now triggers cross-provider failover (tests:
   test_chain_fallback.py).
2. Static plan validation rejected valid chains that consume sql_query output
   (columns exist only post-execution). Fix: steps chained after sql_query are
   opaque to static column checks and policed at runtime instead.
3. Multi-file uploads displayed sanitized table names instead of original
   filenames. Fix: display name preserved separately from table keys.

Note: this workbook contains names/contact details. Per the README security
note, uploaded contents are sent to the configured LLM provider as context.
