from __future__ import annotations

import datetime
import html
import json
import os
import traceback

import streamlit as st

from insight.domain.dataset import DatasetProfile
from insight.domain.errors import AllProvidersFailedError, InsightError
from insight.domain.query import AnalysisRequest
from insight.llm import registry as llm_registry
from insight.llm.registry import ModelInfo, available_models
from insight.orchestrator import run_analysis_sync
from insight.settings import get_settings
from utils import (
    build_report_docx,
    clear_session,
    compute_profile,
    ensure_discovery_registered,
    new_session_id,
    parse_csv,
    session_artifacts_dir,
)

QUICK_ACTIONS = [
    {"label": "Data Summary", "query": "Summarize this dataset: key statistics, distributions, and anything unusual."},
    {"label": "Visualize", "query": "Create 2-3 informative visualizations from this data."},
    {"label": "Find Patterns", "query": "Identify the top 3 most interesting patterns in this dataset."},
    {"label": "Quality Check", "query": "Assess data quality: missing values, duplicates, outliers, inconsistencies."},
]

SAMPLE_QUESTIONS = [
    "What are the column names and their types?",
    "Show summary statistics for every numeric column",
    "Which category is most frequent?",
    "Are there any outliers?",
    "How does [metric] trend over time?",
]

TIER_ORDER = {"fast": 0, "balanced": 1, "accurate": 2, "discovered": 3}


def render_app():
    _init_session_state()
    _run_discovery()
    _inject_css()
    _render_masthead()

    selection = _render_sidebar()
    if not _require_setup():
        return

    content = st.session_state.uploaded_content
    df = parse_csv(content)[0]
    profile = compute_profile(content, st.session_state.uploaded_name)
    if df is None or profile is None:
        st.error("Failed to load dataset.")
        return

    _render_overview(df, profile)
    _render_quick_actions()
    _render_history()
    _handle_input(df, profile, selection)
    _render_export()


def _init_session_state():
    defaults = {
        "session_id": new_session_id(),
        "uploaded_content": None,
        "uploaded_name": None,
        "turns": [],
        "quick_query": None,
        "force_discovery": False,
        "analysis_state": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _run_discovery():
    if not llm_registry.available_models(get_settings()) and not st.session_state.force_discovery:
        return
    with st.spinner("Scanning provider model catalogs..."):
        ensure_discovery_registered(force=st.session_state.force_discovery)
    st.session_state.force_discovery = False


def _inject_css():
    css_path = os.path.join(os.path.dirname(__file__), "style.css")
    if os.path.exists(css_path):
        with open(css_path, encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def _render_masthead():
    st.markdown(
        """
        <div class='masthead-band'>
            <span>Established 2026</span>
            <span>The Daily Insight : Data Edition</span>
            <span>Verified by Deterministic Engines</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col1, col2, col3 = st.columns([1, 4, 1])
    with col2:
        if os.path.exists("assets/logo.png"):
            st.image("assets/logo.png", use_container_width=True)
        else:
            st.markdown("<h1>THE INSIGHT</h1>", unsafe_allow_html=True)
    st.markdown("<hr class='double'>", unsafe_allow_html=True)


def _sorted_models() -> list[ModelInfo]:
    return sorted(
        available_models(),
        key=lambda m: (m.provider, TIER_ORDER.get(m.tier, 9), m.model),
    )


def _render_sidebar():
    settings = get_settings()
    with st.sidebar:
        st.header("Configuration")

        uploaded = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx", "xls"])
        if uploaded is not None:
            content = uploaded.getvalue()
            if (
                st.session_state.uploaded_name != uploaded.name
                or st.session_state.uploaded_content != content
            ):
                old_id = st.session_state.session_id
                clear_session(old_id)
                st.session_state.uploaded_content = content
                st.session_state.uploaded_name = uploaded.name
                st.rerun()
            st.success(f"{uploaded.name} loaded")
        elif st.session_state.uploaded_content is not None:
            clear_session(st.session_state.session_id)
            st.rerun()

        models = _sorted_models()
        labels = {f"{m.model_id} · {m.tier}": m.model_id for m in models}
        default_model_id = next(
            (m.model_id for m in models if m.model_id == settings.default_model_id),
            models[0].model_id if models else "",
        )
        default_label = next(
            (lbl for lbl, mid in labels.items() if mid == default_model_id),
            next(iter(labels), ""),
        )
        index = list(labels.keys()).index(default_label) if default_label in labels else 0
        choice = st.selectbox("Model", list(labels.keys()), index=index)

        b_col1, b_col2 = st.columns([1, 2])
        with b_col1:
            if st.button("Refresh models", use_container_width=True):
                st.session_state.force_discovery = True
                st.rerun()
        with b_col2:
            temperature = st.slider("Temperature", 0.0, 1.0, float(settings.temperature), 0.1)

        with st.expander("Provider status"):
            _render_provider_status()

        if st.button("Clear Conversation", use_container_width=True):
            clear_session(st.session_state.session_id)
            st.rerun()

        st.markdown("---")
        with st.expander("How Insight works"):
            st.markdown(
                "- The LLM **plans** an analysis — it never executes code\n"
                "- A **deterministic pandas engine** computes every number\n"
                "- A **validator** checks results before you see them\n"
                "- Every claim links to its calculation"
            )
        st.caption(
            "Your CSV contents are sent to the selected LLM provider as context."
        )

    return labels.get(choice, settings.default_model_id), temperature


def _render_provider_status():
    settings = get_settings()
    for spec in llm_registry.PROVIDERS.values():
        if spec.requires_key:
            env_name = spec.api_key_env or f"{spec.name.upper()}_API_KEY"
            enabled = llm_registry.provider_enabled(spec, settings)
            key_hint = env_name
        else:
            enabled = llm_registry.provider_enabled(spec, settings)
            key_hint = "OLLAMA_BASE_URL"
        dot = "🟢" if enabled else "⚪"
        count = len(
            [
                m for m in llm_registry.MODEL_CATALOG.values()
                if m.provider == spec.name
            ]
        ) + len(
            [
                m for m in llm_registry.DISCOVERED.values()
                if m.provider == spec.name
            ]
        )
        state = f"{count} models" if enabled else f"set {key_hint}"
        st.markdown(
            f"<span class='provider-row'>{dot} {spec.name.title()} "
            f"<span class='provider-note'>— {state}</span></span>",
            unsafe_allow_html=True,
        )


def _require_setup():
    if not available_models():
        st.info(
            "No LLM provider configured. Add an API key (GROQ_API_KEY, "
            "MISTRAL_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY, "
            "OPENROUTER_API_KEY...) to your `.env` — see `.env.example`."
        )
        return False
    if st.session_state.uploaded_content is None:
        _render_empty_state()
        return False
    return True


def _render_empty_state():
    st.markdown(
        """
        <div class='drop-cap hero-copy'>
            Welcome to the front page of your data. Upload a CSV in the sidebar,
            then consult our automated desk of analysts: they will profile your
            records, plan a rigorous investigation, compute every figure with
            deterministic engines, and verify the results before printing a word.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.subheader("Sample Questions")
    col1, col2 = st.columns(2)
    for i, question in enumerate(SAMPLE_QUESTIONS):
        with col1 if i % 2 == 0 else col2:
            st.markdown(f"- {question}")
    st.markdown("<hr>", unsafe_allow_html=True)


def _render_overview(df, profile: DatasetProfile):
    st.markdown("<div class='kicker'>Special Report</div>", unsafe_allow_html=True)
    st.subheader(f"I. Current State of the Records — {profile.name}")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", f"{profile.row_count:,}")
    col2.metric("Columns", profile.column_count)
    col3.metric("Missing Cells", f"{profile.missing_cells:,}")
    col4.metric("Memory", f"{profile.memory_mb:.1f} MB")

    col5, col6, col7, col8 = st.columns(4)
    roles = [c.role.value for c in profile.columns]
    col5.metric("Duplicates", f"{profile.duplicate_rows:,}")
    col6.metric("Numeric", roles.count("numeric"))
    col7.metric("Categorical", roles.count("categorical"))
    col8.metric("Datetime", roles.count("datetime"))

    fp = profile.fingerprint
    st.markdown(
        f"<div class='mono-label fingerprint'>dataset_id <b>{fp.dataset_id}</b> · "
        f"schema <b>{fp.schema_hash[:10]}</b> · content <b>{fp.content_hash[:10]}</b>"
        f" · profile v{fp.profile_version}</div>",
        unsafe_allow_html=True,
    )

    with st.expander("Column profiles"):
        rows = [
            {
                "column": c.name,
                "role": c.role.value,
                "dtype": c.dtype,
                "missing": f"{c.missing_pct:.1%}",
                "unique": c.unique_count,
                "top values": ", ".join(f"{v.value} ({v.count})" for v in c.top_values[:2]),
            }
            for c in profile.columns
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)

    _render_briefing(df, profile)

    st.markdown("<hr>", unsafe_allow_html=True)


def _render_briefing(df, profile: DatasetProfile):
    from insight.briefing import findings_to_dicts, generate_briefing

    findings = generate_briefing(df, profile)
    if not findings:
        return

    st.markdown("<div class='kicker'>Morning Edition</div>", unsafe_allow_html=True)
    st.subheader("II. Top Findings — Zero-Prompt Briefing")

    for finding in findings_to_dicts(findings):
        with st.container():
            st.markdown(
                f"<div class='finding-card'>"
                f"<span class='chip'><span class='chip-key'>{finding['category']}</span></span> "
                f"<b>{finding['title']}</b><br>"
                f"<span class='provider-note'>{finding['detail']}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
            if st.button(
                f"Investigate: {finding['suggested_query'][:48]}...",
                key=f"briefing_{finding['rank']}",
                use_container_width=True,
            ):
                st.session_state.quick_query = finding["suggested_query"]


def _render_quick_actions():
    cols = st.columns(len(QUICK_ACTIONS))
    for i, action in enumerate(QUICK_ACTIONS):
        with cols[i]:
            if st.button(action["label"], key=f"qa_{i}", use_container_width=True):
                st.session_state.quick_query = action["query"]
                st.rerun()


def _render_history():
    for turn in st.session_state.turns:
        with st.chat_message("user"):
            st.markdown(turn["question"])
        _render_assistant_turn(turn["response"])


def _badges_html(response) -> str:
    meta = response.meta
    chips: list[str] = []

    def chip(label: str, value: str) -> str:
        return (
            f"<span class='chip'><span class='chip-key'>"
            f"{html.escape(label)}</span> {html.escape(value)}</span>"
        )

    if meta.get("model"):
        chips.append(chip("model", meta["model"]))
    if meta.get("total_ms") is not None:
        chips.append(chip("total", f"{meta['total_ms']:,} ms"))
        chips.append(
            chip(
                "breakdown",
                f"plan {meta.get('planning_ms', 0):,} · exec "
                f"{meta.get('execution_ms', 0):,} · explain {meta.get('explaining_ms', 0):,} ms",
            )
        )
    if meta.get("tokens_out"):
        chips.append(chip("tokens", f"{meta['tokens_in']:,}→{meta['tokens_out']:,}"))
    if meta.get("repair_attempted"):
        chips.append("<span class='chip chip-warn'>⚠ self-repaired</span>")
    return f"<div class='chip-row'>{''.join(chips)}</div>"


def _render_assistant_turn(response):
    with st.chat_message("assistant"):
        if response.error == "plan_validation_failed":
            st.error(response.answer)
        else:
            st.markdown(response.answer)

        for plot in response.plots:
            if os.path.exists(plot.path):
                st.image(plot.path)

        st.markdown(_badges_html(response), unsafe_allow_html=True)

        if response.tables:
            with st.expander("View calculation"):
                for table in response.tables:
                    st.caption(f"step {table.step_id}: {table.name}")
                    st.dataframe(table.to_dataframe(), hide_index=True)
                    if table.truncated:
                        st.caption(f"(showing first rows of {table.total_rows:,})")

        steps = response.meta.get("steps") or []
        if steps:
            with st.expander("View plan"):
                st.caption(f"objective: {response.meta.get('objective', '')}")
                for step in steps:
                    params = {k: v for k, v in step.get("params", {}).items() if k != "reason"}
                    reason = step.get("reason")
                    st.markdown(
                        f"**{step['step_id']} · `{step['operation']}`**"
                        + (f" — {reason}" if reason else "")
                        + (" · failed" if not step.get("success") else "")
                    )
                    if params:
                        st.code(json.dumps(params, indent=2, default=str), language="json")

        warnings_list = [e.label for e in response.evidence if e.kind == "warning"]
        if warnings_list:
            st.warning("; ".join(warnings_list))


def _handle_input(df, profile, selection):
    question = st.session_state.quick_query or st.chat_input(
        "Ask a question about your data..."
    )
    st.session_state.quick_query = None
    if not question:
        return

    with st.chat_message("user"):
        st.markdown(question)

    history = []
    for turn in st.session_state.turns[-3:]:
        history.append(("user", turn["question"]))
        history.append(("assistant", turn["response"].answer[:400]))

    request = AnalysisRequest(
        question=question,
        history=history,
        model_id=selection[0],
        temperature=selection[1],
    )

    session_state_model = None
    if st.session_state.analysis_state:
        from insight.conversation.state import AnalysisSessionState

        try:
            session_state_model = AnalysisSessionState.model_validate(
                st.session_state.analysis_state
            )
        except Exception:
            session_state_model = None

    with st.chat_message("assistant"):
        try:
            with st.spinner("Investigating your data..."):
                response = run_analysis_sync(
                    df,
                    profile,
                    request,
                    artifacts_dir=session_artifacts_dir(st.session_state.session_id),
                    history=history,
                    state=session_state_model,
                )
        except AllProvidersFailedError as exc:
            details = "; ".join(str(e)[:120] for e in exc.errors[:3])
            st.error(
                f"All configured providers failed. Details: {details}\n\n"
                "Check API keys/quota, or pick another model in the sidebar."
            )
            return
        except InsightError as exc:
            st.error(f"Analysis failed: {exc.message}")
            return
        except Exception:
            st.error("Unexpected error while analyzing. See details below.")
            with st.expander("Technical details"):
                st.code(traceback.format_exc(), language="text")
            traceback.print_exc()
            return

    st.session_state.turns.append({"question": question, "response": response})
    if response.meta.get("analysis_state"):
        st.session_state.analysis_state = response.meta["analysis_state"]
    _render_assistant_turn(response)


def _render_export():
    turns = st.session_state.get("turns", [])
    with st.sidebar:
        if not turns:
            return
        dataset_name = st.session_state.get("uploaded_name") or "Dataset"
        date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        try:
            docx_bytes = build_report_docx(turns, dataset_name, date_str)
            st.download_button(
                label="Export Report (DOCX)",
                data=docx_bytes,
                file_name=f"Insight_Report_{dataset_name.replace('.csv', '')}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True,
            )
        except Exception as exc:
            st.warning(f"Export unavailable: {exc}")
