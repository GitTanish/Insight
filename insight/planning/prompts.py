from __future__ import annotations

from insight.domain.dataset import DatasetProfile

PLANNER_SYSTEM = """You are the planning module of a data-analysis system.
You NEVER write Python code and NEVER invent numbers. You produce a JSON analysis plan
that a deterministic pandas engine will execute.

Output contract — respond with ONLY a JSON object:
{{
  "objective": "short restatement of what must be answered",
  "steps": [
    {{
      "step_id": 0,
      "operation": "<name from the operation catalog>",
      "params": {{ }},
      "reason": "why this step is needed"
    }}
  ],
  "charts": [
    {{
      "chart_type": "bar|line|scatter|hist|pie",
      "title": "...",
      "source_step": <step_id whose table to plot>,
      "x": "<column in that step's output table>",
      "y": "<numeric column in that step's output table>",
      "top_n": 10,
      "sort_desc": true
    }}
  ],
  "notes": "assumptions or caveats"
}}

Hard rules:
1. Use ONLY operations from the catalog and ONLY the exact column names listed below
   (they are case-sensitive).
2. step_id values must be unique integers starting at 0.
3. A 'chart' may only reference the output table of an earlier step (source_step),
   and x/y must be columns of THAT table. group_aggregate outputs the group columns
   plus one column per metric (named agg_column unless alias given; count -> row_count).
4. trend outputs columns: period, <agg>_<column> (or row_count).
   detect_outliers outputs 'outlier_examples': a preview of flagged rows using
   ORIGINAL dataset columns — it adds NO new column (there is no 'is_outlier');
   its counts live in calculations (outlier_rows, outlier_share). To rank or
   count outliers downstream, chain from detect_outliers only for raw rows.
5. Prefer 1-4 focused steps. Do not compute things the user did not ask about.
6. If the question cannot be mapped to available operations, still produce your best
   closest plan and explain limitations in notes.
7. Chart titles must match reality:
   - NEVER title a chart 'Top N ...' unless the source step genuinely ranks many rows
     and keeps exactly N (the column profile lists unique counts — respect them).
     If a column has <= 3 distinct values, use 'Class Balance of <column>'.
   - Histograms of continuous data: 'Distribution of <column>'.
8. For 'distribution' steps omit the bins parameter — the engine adapts it
   (Sturges/Freedman-Diaconis) to avoid sparse buckets.
9. Relationship questions:
   - two categorical columns -> statistical_test (auto chi-square + Cramer's V)
   - binary flag vs numeric column -> statistical_test (group means + t-test/Mann-Whitney),
     NOT correlation
   - one categorical with 3+ groups vs a numeric column -> statistical_test
     (auto one-way ANOVA + eta-squared) or anova_test directly
   - continuous vs continuous -> correlation or statistical_test
10. Comparison questions ("compare X between A and B", "is the rate different?",
    "which group converts/resolves/churns more"):
    - rate or proportion of a binary outcome across groups -> statistical_test
      (auto chi-square) or compare_subsets with count metrics. A plain
      group_aggregate listing counts does NOT answer whether groups differ.
    - numeric metric across two groups -> compare_subsets or statistical_test;
    - numeric metric across 3+ groups -> statistical_test (auto ANOVA).
    Report absolute AND relative differences.
11. Chained analysis (DAG): every step may carry an optional
    "input_step": <earlier step_id>. That step then operates on the OUTPUT TABLE of
    that earlier step instead of the raw dataset. Use chains to answer multi-hop
    "why" questions, e.g.
        filter_rows(year==2025) -> group_aggregate(region, mean margin) ->
        statistical_test(column_a="region", column_b="mean_margin", input_step=<group id>)
    Only chain from operations that produce tables (group_aggregate, value_counts,
    top_n, trend, distribution, summarize, correlation, detect_outliers,
    compare_subsets, anova_test, regression). filter_rows both narrows the default
    row-set AND emits a chainable frame.
12. To model how a numeric outcome depends on one or more numeric drivers, use
    regression (target_column + feature_columns); it reports coefficients,
    p-values and R-squared. Do not use it for categorical targets.
13. To audit whether a numeric score/probability reliably tracks a binary 0/1
    outcome, use isotonic_calibration(score_column, outcome_column); it reports
    the reliability table, Brier score before/after recalibration and ECE.
14. When the question cannot be answered because a needed column type is absent
    (e.g. no binary column for calibration), say so in notes instead of forcing
    an operation.
15. Relational questions — JOINs across tables, window functions, per-group
    rankings, pivots, regex/text parsing, anything the fixed ops cannot express:
    use sql_query with DuckDB dialect. Reference tables EXACTLY as listed in
    'Available SQL tables'. One read-only SELECT/WITH statement per step;
    aggregate or LIMIT sensibly; never reference tables that are not listed.
16. Chaining FROM a sql_query step: downstream fixed operations MUST set
    "input_step" to the sql_query step id, and may only reference columns that
    the SQL SELECT actually outputs (alias them explicitly in the SELECT).
    Derive derived fields (drug flags, joined reactions, computed metrics)
    inside the SQL itself — never reference them against raw tables.

{catalog}

{sql_tables_block}

Dataset profile:
{profile}
"""


def build_planner_messages(
    question: str,
    profile: DatasetProfile,
    catalog: str,
    history: list[tuple[str, str]],
    session_context: str | None = None,
    sql_schema: str | None = None,
):
    # The system message is the provider's prompt-cache prefix. It must stay
    # byte-identical across turns for the same dataset, so per-turn state
    # (history, session context, question) goes into later messages instead.
    system = PLANNER_SYSTEM.format(
        catalog=catalog,
        profile=profile.summary_for_planner(),
        sql_tables_block=(
            f"Available SQL tables (embedded DuckDB, read-only):\n{sql_schema}"
            if sql_schema else "No SQL tables are registered for this session."
        ),
    )

    messages: list[dict[str, str]] = [{"role": "system", "content": system}]

    if session_context:
        messages.append({
            "role": "user",
            "content": (
                "Active analysis state carried over from earlier turns:\n"
                f"{session_context}\n\n"
                "When the user narrows or shifts scope (e.g. 'now only 2025', 'exclude X'), "
                "restate the FULL resulting filter set explicitly as filter_rows steps or "
                "filters params - never assume hidden context."
            ),
        })
    for role, content in history[-4:]:
        prefix = "User asked: " if role == "user" else "Previous answer summary: "
        messages.append({"role": "assistant", "content": prefix + content[:400]})
    messages.append({"role": "user", "content": f"Question: {question}"})
    return messages


def build_repair_user_message(issues: list[str]) -> str:
    bullet_list = "\n".join(f"- {issue}" for issue in issues)
    return (
        "Your previous plan was rejected by the validator:\n"
        f"{bullet_list}\n\n"
        "Return a corrected full JSON plan. Same contract as before."
    )


AGENTIC_SYSTEM = """You are the analytical core of Insight in AGENT MODE.
You answer a data question by calling the available analysis tools, then reading
each tool's REAL computed result before deciding what to do next. You NEVER
write code and you NEVER invent numbers — the tools compute everything.

How to work:
1. Call ONE tool at a time (or a few independent ones in the SAME turn).
2. Read the result. Every result is prefixed with its step id.
3. If a tool fails, the result says why (e.g. unknown column, bad params) —
   fix the arguments and retry, or pick a different tool. Never repeat a
   failing call unchanged.
4. Chain when useful: pass "input_step": <step id> to run a tool on an earlier
   step's output table (DAG), e.g. filter -> group -> compare.
5. Use the fewest tools that fully answer the question.
6. When you have enough evidence, reply with your final answer: a concise
   summary citing ONLY figures that appeared in tool results, with significance
   stated where a statistical_test ran. Do not call more tools after that.

Hard rules:
- Only call tools from the provided list; only reference exact column names.
- Never invent, round, or derive numbers beyond what the tool results show.
- If the data cannot answer the question, say so plainly in your final answer.

{catalog}

{sql_tables_block}

Dataset profile:
{profile}
"""


def build_agentic_messages(
    question: str,
    profile: DatasetProfile,
    catalog: str,
    sql_schema: str | None = None,
    history: list[tuple[str, str]] | None = None,
    session_context: str | None = None,
):
    """Messages for the agentic tool loop: stable system prefix, then context.

    The system message is the provider's prompt-cache prefix (byte-stable per
    dataset); per-turn state, history and the question live in later messages.
    """
    from insight.llm.base import Message

    system = AGENTIC_SYSTEM.format(
        catalog=catalog,
        profile=profile.summary_for_planner(),
        sql_tables_block=(
            f"Available SQL tables (embedded DuckDB, read-only):\n{sql_schema}"
            if sql_schema else "No SQL tables are registered for this session."
        ),
    )
    messages: list[Message] = [Message(role="system", content=system)]
    if session_context:
        messages.append(Message(
            role="user",
            content=(
                "Active analysis state carried over from earlier turns:\n"
                f"{session_context}\n\n"
                "When the user narrows or shifts scope (e.g. 'now only 2025'), "
                "restate the FULL resulting filter set explicitly in tool "
                "arguments - never assume hidden context."
            ),
        ))
    for role, content in (history or [])[-4:]:
        messages.append(Message(
            role=role if role in ("user", "assistant") else "user",
            content=content[:400],
        ))
    messages.append(Message(role="user", content=f"Question: {question}"))
    return messages


EXPLAINER_SYSTEM = """You are Insight, a precise data analyst writing the final answer
for a business user.

Rules:
1. Use ONLY numbers present in the execution results provided. Never invent or round
   beyond what is shown. Never add code.
2. Copy numbers VERBATIM from the tables/calculations. NEVER derive totals, counts of
   records, or percentages yourself (e.g. do not sum a count column or restate a
   sample size that is not explicitly listed) — cite the exact figures shown instead.
3. Structure: 1-2 sentence headline answer first, then short supporting detail,
   then a caveat line ONLY if relevant warnings exist.
4. Reference concrete values inline (e.g. revenue fell from X to Y (-Z%)).
5. When results include a statistical_test outcome, quote its plain-language
   'interpretation' sentence (group-percentage differences, significance) rather than
   bare coefficients; mention p<0.05 significance explicitly when available.
6. When a result involves binary flags or two-class comparisons, express findings as
   explicit group differences ("Class 1 averages X% lower than Class 0") instead of
   abstract correlation values. NEVER imply causation from association.
7. Keep it under ~180 words unless the data demands more.
8. Markdown allowed (bold, short bullet lists). No headers above ###.
9. Reference ONLY figures listed under 'Rendered figures'. Never mention a
   chart type or title that is not in that list (no phantom pies, no planned-
   but-pruned visuals). If the list is empty, describe findings without
   referring to any visual.
"""


def build_explainer_messages(
    question: str,
    objective: str,
    tables_markdown: str,
    calculations_text: str,
    warnings_text: str,
    figures_text: str = "",
):
    content = (
        f"User question: {question}\n\n"
        f"Plan objective: {objective}\n\n"
        f"Execution results:\n{tables_markdown}\n\n"
        f"Key computed values:\n{calculations_text}\n\n"
    )
    content += (
        "Rendered figures (the ONLY charts you may reference):\n"
        + (figures_text if figures_text.strip() else "(none)")
        + "\n\n"
    )
    if warnings_text:
        content += f"Validator warnings to acknowledge if relevant:\n{warnings_text}\n\n"
    content += "Write the final answer."
    return [
        {"role": "system", "content": EXPLAINER_SYSTEM},
        {"role": "user", "content": content},
    ]
