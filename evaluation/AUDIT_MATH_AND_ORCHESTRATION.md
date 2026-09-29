# Insight — Audit: Math Evaluation & Orchestration

Date: 2026-08-22 · suite: 178 offline tests + 20 golden-question cases + 16 real-world questions

---

## 1. How good is the mathematics, really?

**Verdict: the numbers are trustworthy; the grader was the weak link.**

Evidence from this audit run (`mistral/mistral-small-latest`, 20 golden cases):

| Run | Result | Nature of misses |
|---|---|---|
| Full suite | 18/20 | both misses were **scoring artifacts** (see below) |
| Re-run of the 2 misses after grader fix | 2/2 | — |
| **Effective** | **20/20** | — |

The two "failures" were re-inspected by hand:

1. `sup_resolution_association` — the answer *did* conclude
   *"statistically significant (χ² = 51.07, p = 8.14e-12, Cramér's V = 0.342)"*, which
   matches pandas ground truth exactly (chat 89.73% vs email 52.84% resolution).
   The case asserted the literal substring `significant`, and an earlier phrasing
   of the same run said "strong association" instead. **Grader was vocabulary-brittle.**
2. `sup_chat_vs_email_resolution` — the rerun planned `['sql_query', 'statistical_test']`
   and returned the exact correct rates. The case only whitelisted
   `statistical_test`/`compare_subsets`, so a *valid* SQL route scored as a miss.
   **Grader was route-brittle.**

Both were fixed in `cases.json` (substring → `associat`; any-of adds `sql_query`)
and now pass. No engine change was warranted — and notably, the **grounding guard
reported `unverified_figures: []`** on both, i.e. every number in the narrative
(χ², p, V, rates) traced back to computed evidence.

### Where the maths is actually verified

| Layer | What it proves | Coverage |
|---|---|---|
| Independent recomputation (`validation/verification.py`) | every headline aggregate re-derived through a second arithmetic path | group_aggregate, top_n, value_counts, correlation |
| Golden cases (`evaluation/`) | exact expected values from independently computed ground truth (±0.5% default) | 20 cases / 5 datasets |
| Statistical routing (`test_statistics.py`) | t-test, Mann-Whitney, chi², ANOVA+η², OLS coefficients, CIs, isotonic PAV | 26 tests |
| Real-world cross-check (`REAL_WORLD_LOG.md`) | diamonds + 4-sheet clinical workbook vs pandas/scipy truth | 16 questions |
| Missing-data gauntlet (`test_invariants.py`) | every table-producing op under 30% MCAR | 14 ops parametrized |

**Known weaknesses (stated, not hidden):**

- Planning variance: the same question can route through different valid ops
  run-to-run (`sql_query` vs `statistical_test`). Numbers stay right; op-choice
  assertions need any-of groups (now reflected in the case file).
- Grounding guard covers **numbers**, not phrasing or claim-level semantics.
  It cannot tell a correct number attached to the wrong sentence.
- Free/aggregated model aliases (`*_free`, proxied `gpt-*` routes) vary in
  planning quality; correctness of output numbers is unaffected by design.

---

## 2. Orchestration status

Pipeline as executed today, per question:

```
profiler + briefing (on upload)
   ↓
plan (LLM, JSON)  ──✗──► plan_validation_failed → explicit "what went wrong" (no answer)
   ↓ ok
execute (deterministic ops / sandboxed DuckDB)  ──✗──► step error
   ↓                                            ↓
validate (sanity, small-N, MCAR, chart refs)     └──► repair once
   ↓ ok                                             (re-plan with error text)
independent recomputation ──✗──► repair once
   ↓ ok
render charts (editorial, adaptive bins, fallbacks)
   ↓
explain (LLM, computed values only)  ──✗──► deterministic fallback summary
   ↓
grounding check (every number traced to evidence) ──✗──► 1 forced rewrite
   ↓                                              └──► ⚠ N unverified figure(s)
result → SSE stream (stages) → DOCX / dbt / alert exports
```

| Capability | Status | Evidence |
|---|---|---|
| Plan validation + bounded repair | ✅ | `settings.max_repair_attempts=1`; live self-repair observed on IPL & clinical runs |
| Deterministic execution, 15 ops | ✅ | 14 ops survive 30% missingness gauntlet; sql_query sandbox blocks 10/10 hostile patterns |
| Validator (sanity, small-N, MCAR, charts) | ✅ | MCAR caveat >5% missing; small-N warnings on state rankings |
| Independent recomputation | ✅ | caught a real chained-filter bug during development |
| Answer grounding guard | ✅ | caught the fabricated "13,650" record count on real data |
| Cross-provider failover | ✅ | empty-content failover proven live (reasoning endpoint → groq) |
| Conversation state | ✅ | filters/dims/metrics persist; follow-ups become filter patches |
| Query cache (TTL + size budget) | ✅ | content-hash keyed incl. session state; eviction tests |
| Streaming UX | ✅ (this pass) | stage rail, live timer, stop/abort, persistent errors |
| BYO-key + hard-stop switch | ✅ | `INSIGHT_REQUIRE_USER_KEY=on` returns 401 without a key |
| Multi-worker / external session store | ⛔ documented | in-memory sessions by design (README + Dockerfile) |

### Open orchestration items

1. `sql_query` results are not independently recomputable (arbitrary SQL) — they
   are validated structurally (finite values, non-empty, chart refs) but not
   re-derived. Accepted trade-off; could add `EXPLAIN`-based plan checks.
2. Narrative-level claim verification (does the sentence match the number?) is
   not automated; grounding covers numbers only.
3. MAR/MNAR detection is a caveat, not a diagnostic (see `future_scope.md`).

---

## 3. Defects found and fixed in this audit

| Defect | Fix |
|---|---|
| Errors auto-hid after 1.8 s (BYO-key 401 unreadable) | persistent `setError` surface |
| No cancel path for 30–190 s queries | `AbortController` + Stop button |
| Progress shown as one overwritten line | 7-stage rail with warn states (repairing/verifying) |
| Golden grader demanded literal `significant` | substring → `associat` |
| Golden grader rejected valid `sql_query` route | any-of now includes `sql_query` |
| Typographic unicode in chart text (glyph 8209 warnings) | `_font_safe()` on titles/labels/ticks |
| `/health` 404 from host probes | alias route added |

Suite after this audit: **178 passed, ruff clean.**
