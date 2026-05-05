"""
app.py
------
Streamlit dashboard for Cross-Lingual Hallucination Drift results.

Run: streamlit run app.py
"""

import json
import glob
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Config ────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Cross-Lingual Hallucination Drift",
    page_icon="🌐",
    layout="wide",
)

LANG_LABELS = {"en": "English", "es": "Spanish", "it": "Italian", "sw": "Swahili"}
TASK_LABELS = {"truthfulqa": "TruthfulQA (Factual QA)", "xcopa": "XCOPA (Commonsense)"}
COLORS      = {"en": "#4C72B0", "es": "#DD8452", "it": "#c44e52", "sw": "#55A868"}

# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data
def load_labels():
    rows = []
    for path in glob.glob("outputs/labels/*.json"):
        with open(path, encoding="utf-8") as f:
            rows.extend(json.load(f))
    df = pd.DataFrame(rows)
    df["is_hallucinated"] = (df["label"] == "Hallucinated").astype(int)
    df["lang_label"] = df["language"].map(LANG_LABELS)
    df["task_label"] = df["task"].map(TASK_LABELS)
    return df


@st.cache_data
def compute_hr(df):
    results = []
    for (task, lang), group in df.groupby(["task", "language"]):
        total = len(group)
        hallucinated = (group["label"] == "Hallucinated").sum()
        hr = (hallucinated / total) * 100 if total > 0 else 0
        results.append({
            "task": task, "language": lang,
            "task_label": TASK_LABELS.get(task, task),
            "lang_label": LANG_LABELS.get(lang, lang),
            "total": total, "hallucinated": int(hallucinated),
            "HR": round(hr, 2),
            "avg_tokens_hall": round(
                group[group["label"] == "Hallucinated"]["token_count"].mean(), 1
            ) if hallucinated > 0 else 0,
            "avg_tokens_faith": round(
                group[group["label"] == "Faithful"]["token_count"].mean(), 1
            ) if (total - hallucinated) > 0 else 0,
        })
    hr_df = pd.DataFrame(results)
    en_rates = hr_df[hr_df["language"] == "en"].set_index("task")["HR"]
    hr_df["delta_HR"] = hr_df.apply(
        lambda r: round(r["HR"] - en_rates.get(r["task"], 0), 2), axis=1
    )
    return hr_df


@st.cache_data
def load_cause_effect():
    try:
        return pd.read_csv("results/tables/xcopa_cause_effect.csv")
    except FileNotFoundError:
        return None


@st.cache_data
def load_kappa():
    try:
        return pd.read_csv("outputs/labels/dual_judge/kappa_summary.csv")
    except FileNotFoundError:
        return None


@st.cache_data
def load_error_categories():
    try:
        long = pd.read_csv("results/tables/error_categories.csv")
        wide = pd.read_csv("results/tables/error_categories_pivot.csv")
        return long, wide
    except FileNotFoundError:
        return None, None


@st.cache_data
def load_stats():
    try:
        return pd.read_csv("results/tables/statistical_tests.csv")
    except FileNotFoundError:
        return None


df       = load_labels()
hr_df    = compute_hr(df)
ce_df    = load_cause_effect()
kappa_df = load_kappa()
err_df, err_pivot = load_error_categories()
stats_df = load_stats()

# Phi values
def get_delta(lang, task):
    row = hr_df[(hr_df["language"] == lang) & (hr_df["task"] == task)]
    return float(row["delta_HR"].values[0]) if len(row) else None

phi_es_sw = round(get_delta("es", "truthfulqa") - get_delta("sw", "xcopa"), 2) \
    if get_delta("es", "truthfulqa") is not None and get_delta("sw", "xcopa") is not None else None
phi_it = round(get_delta("it", "truthfulqa") - get_delta("it", "xcopa"), 2) \
    if get_delta("it", "truthfulqa") is not None and get_delta("it", "xcopa") is not None else None

# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.title("🌐 Hallucination Drift")
st.sidebar.caption("CS505 · Boston University")
page = st.sidebar.radio(
    "Navigate",
    ["Dashboard", "Charts", "New Analyses", "Example Browser", "Reason Analysis"],
    label_visibility="collapsed",
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Model:** Aya Expanse 8B  \n"
    "**Judge:** GPT-4o-mini  \n"
    "**Samples:** 150 / cell  \n"
    "**Cells:** en/es/it (TruthfulQA), en/it/sw (XCOPA)"
)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Dashboard
# ══════════════════════════════════════════════════════════════════════════════

if page == "Dashboard":
    st.title("Cross-Lingual Hallucination Drift")
    st.markdown(
        "Does hallucination drift depend on **task type**? "
        "Aya Expanse 8B across **English, Spanish, Italian, Swahili** on "
        "TruthfulQA (factual QA) and XCOPA (commonsense reasoning). "
        "Italian appears in both tasks as a **confound control**."
    )

    # KPI row
    st.markdown("### Key Results")
    cols = st.columns(4)
    kpis = [
        ("XCOPA EN→SW drift",       f"{get_delta('sw','xcopa'):+.2f} pp" if get_delta('sw','xcopa') else "N/A"),
        ("TruthfulQA EN→IT drift",  f"{get_delta('it','truthfulqa'):+.2f} pp" if get_delta('it','truthfulqa') else "N/A"),
        ("XCOPA EN→IT drift",       f"{get_delta('it','xcopa'):+.2f} pp" if get_delta('it','xcopa') else "N/A"),
        ("Total labeled examples",  f"{len(df):,}"),
    ]
    for col, (label, value) in zip(cols, kpis):
        col.metric(label, value)

    st.markdown("---")

    # HR table
    st.markdown("### Hallucination Rates per Cell")
    display = hr_df[["task_label", "lang_label", "total", "hallucinated", "HR", "delta_HR"]].copy()
    display.columns = ["Task", "Language", "Total", "Hallucinated", "HR (%)", "ΔHR vs EN (pp)"]
    lang_order = ["English", "Spanish", "Italian", "Swahili"]
    display["Language"] = pd.Categorical(display["Language"], categories=lang_order, ordered=True)
    display = display.sort_values(["Task", "Language"])

    def color_hr(val):
        if isinstance(val, float):
            if val > 50:  return "background-color: #f8d7da; color: #721c24"
            if val > 20:  return "background-color: #fff3cd; color: #856404"
            return "background-color: #d4edda; color: #155724"
        return ""

    st.dataframe(
        display.style.map(color_hr, subset=["HR (%)"]),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")

    # Φ scores
    st.markdown("### Drift Interaction Score (Φ)")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Cross-task aggregate Φ (es/TQA vs sw/XCOPA)**")
        st.markdown(
            f"Φ = ΔHR(es, TruthfulQA) − ΔHR(sw, XCOPA) = **{phi_es_sw:+.2f} pp**  \n"
            "χ² = 45.44, p < 0.001 — drift is strongly task-dependent."
            if phi_es_sw else "N/A"
        )
    with c2:
        st.markdown("**Within-Italian Φ (confound control)**")
        st.markdown(
            f"Φ_it = ΔHR(it, TruthfulQA) − ΔHR(it, XCOPA) = **{phi_it:+.2f} pp**  \n"
            "χ² = 16.98, p < 0.001 — task-dependent drift confirmed within same language."
            if phi_it is not None else "N/A"
        )

    st.markdown("---")

    # Token verbosity
    st.markdown("### Token Verbosity: Hallucinated vs Faithful Responses")
    st.caption("Swahili XCOPA hallucinations are ~2× longer — incoherent rambling, not short wrong answers.")
    tok_rows = []
    for _, row in hr_df.iterrows():
        tok_rows.append({
            "Cell": f"{row['lang_label']}\n({row['task_label'].split('(')[0].strip()})",
            "Hallucinated": row["avg_tokens_hall"],
            "Faithful": row["avg_tokens_faith"],
        })
    tok_df = pd.DataFrame(tok_rows)
    tok_melt = tok_df.melt(id_vars="Cell", var_name="Label", value_name="Avg Tokens")
    fig_tok = px.bar(
        tok_melt, x="Cell", y="Avg Tokens", color="Label",
        barmode="group", color_discrete_map={"Hallucinated": "#e74c3c", "Faithful": "#2ecc71"},
        height=380,
    )
    fig_tok.update_layout(margin=dict(t=10, b=10), legend_title_text="")
    st.plotly_chart(fig_tok, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Charts
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Charts":
    st.title("Charts")

    st.markdown("### Hallucination Rate by Language and Task")
    fig1 = px.bar(
        hr_df, x="lang_label", y="HR", color="language",
        facet_col="task_label", text="HR",
        color_discrete_map=COLORS,
        labels={"lang_label": "Language", "HR": "Hallucination Rate (%)", "task_label": "Task"},
        height=440,
        category_orders={"lang_label": ["English", "Spanish", "Italian", "Swahili"]},
    )
    fig1.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig1.update_layout(showlegend=False, margin=dict(t=40, b=10))
    fig1.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("---")

    st.markdown("### Cross-Lingual Drift (ΔHR vs English)")
    non_en = hr_df[hr_df["language"] != "en"].copy()
    fig2 = px.bar(
        non_en, x="lang_label", y="delta_HR", color="task_label",
        barmode="group", text="delta_HR",
        color_discrete_sequence=["#4C72B0", "#DD8452"],
        labels={"lang_label": "Language", "delta_HR": "ΔHR vs English (pp)", "task_label": "Task"},
        height=440,
        category_orders={"lang_label": ["Spanish", "Italian", "Swahili"]},
    )
    fig2.update_traces(texttemplate="%{text:+.1f}", textposition="outside")
    fig2.add_hline(y=0, line_dash="dash", line_color="black", line_width=1)
    fig2.update_layout(margin=dict(t=10, b=10), legend_title_text="Task")
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    st.markdown("### Hallucination Rate Heatmap")
    pivot = hr_df.pivot(index="lang_label", columns="task_label", values="HR")
    pivot = pivot.reindex(["English", "Spanish", "Italian", "Swahili"])
    fig3 = px.imshow(
        pivot, text_auto=".1f", color_continuous_scale="RdYlGn_r",
        labels={"color": "HR (%)"},
        height=340,
    )
    fig3.update_layout(margin=dict(t=10, b=10))
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")

    st.markdown("### Statistical Test Results")
    if stats_df is not None:
        display_stats = stats_df[["test", "comparison", "statistic", "p_value", "significant"]].copy()
        display_stats.columns = ["Test", "Comparison", "Statistic", "p-value", "Significant"]

        def color_sig(val):
            if val == "YES": return "background-color: #f8d7da; color: #721c24; font-weight: bold"
            return "background-color: #d4edda; color: #155724"

        st.dataframe(
            display_stats.style.map(color_sig, subset=["Significant"]),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Run src/06_statistical_tests.py to generate this table.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — New Analyses
# ══════════════════════════════════════════════════════════════════════════════

elif page == "New Analyses":
    st.title("New Analyses")
    st.markdown("Italian confound control · XCOPA cause/effect · Dual-judge agreement · Error categories")

    tab1, tab2, tab3 = st.tabs(["XCOPA Cause vs Effect", "Dual-Judge Agreement", "Error Categories"])

    # ── Tab 1: Cause vs Effect ────────────────────────────────────────────────
    with tab1:
        st.markdown("### XCOPA: Cause vs Effect Hallucination Rates")
        st.markdown(
            "Does causal direction (cause vs effect) affect hallucination? "
            "Swahili shows near-total failure in both directions."
        )
        if ce_df is not None:
            ce_df["lang_label"] = ce_df["language"].map(LANG_LABELS)
            fig_ce = px.bar(
                ce_df, x="lang_label", y="HR", color="question_type",
                barmode="group", text="HR",
                color_discrete_map={"cause": "#5b7fbf", "effect": "#bf7f5b"},
                labels={"lang_label": "Language", "HR": "Hallucination Rate (%)", "question_type": "Type"},
                height=420,
                category_orders={"lang_label": ["English", "Italian", "Swahili"]},
            )
            fig_ce.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig_ce.update_layout(margin=dict(t=10, b=10), legend_title_text="Question Type")
            st.plotly_chart(fig_ce, use_container_width=True)

            st.dataframe(
                ce_df[["lang_label", "question_type", "total", "hallucinated", "HR"]]
                .rename(columns={"lang_label": "Language", "question_type": "Type",
                                  "total": "N", "hallucinated": "Hallucinated", "HR": "HR (%)"})
                .reset_index(drop=True),
                use_container_width=True,
                hide_index=True,
            )
            st.caption(
                "No significant cause vs effect difference within Italian or Swahili "
                "(both Swahili directions are near-total hallucination)."
            )
        else:
            st.info("Run src/08_xcopa_cause_effect.py to generate this data.")

    # ── Tab 2: Dual-Judge Agreement ───────────────────────────────────────────
    with tab2:
        st.markdown("### Dual-Judge Agreement (Cohen's κ)")
        st.markdown(
            "GPT-4o-mini (primary judge) vs Claude claude-sonnet-4-5 (second judge) vs human annotators "
            "on 50 Italian samples per cell."
        )
        if kappa_df is not None:
            k = kappa_df.copy()
            fig_k = go.Figure()
            comparisons = ["GPT vs Claude", "GPT vs Human", "Claude vs Human"]
            cols_k = ["kappa_gpt_vs_claude", "kappa_gpt_vs_human", "kappa_claude_vs_human"]
            colors_k = ["#4C72B0", "#DD8452", "#55A868"]
            for comp, col, color in zip(comparisons, cols_k, colors_k):
                fig_k.add_trace(go.Bar(
                    name=comp, x=k["cell"], y=k[col],
                    text=[f"{v:.3f}" for v in k[col]],
                    textposition="outside",
                    marker_color=color,
                ))
            fig_k.add_hline(y=0.6, line_dash="dot", line_color="gray",
                            annotation_text="Moderate agreement threshold (κ=0.6)")
            fig_k.update_layout(
                barmode="group", height=420,
                xaxis_title="Cell", yaxis_title="Cohen's κ",
                yaxis=dict(range=[-0.1, 1.0]),
                legend_title_text="Comparison",
                margin=dict(t=10, b=10),
            )
            st.plotly_chart(fig_k, use_container_width=True)

            st.dataframe(
                k.rename(columns={
                    "cell": "Cell", "n_pairs": "N",
                    "n_human_filled": "Human Labels",
                    "raw_agree_gpt_vs_claude": "Raw Agree (GPT/Claude)",
                    "kappa_gpt_vs_claude": "κ GPT/Claude",
                    "kappa_gpt_vs_human": "κ GPT/Human",
                    "kappa_claude_vs_human": "κ Claude/Human",
                }),
                use_container_width=True,
                hide_index=True,
            )
            st.caption(
                "TruthfulQA/Italian: moderate agreement (κ=0.662). "
                "XCOPA/Italian: near-chance agreement (κ=0.065) — "
                "commonsense outputs are genuinely ambiguous even for strong LLM judges."
            )
        else:
            st.info("Run src/09_dual_judge_italian.py to generate this data.")

    # ── Tab 3: Error Categories ───────────────────────────────────────────────
    with tab3:
        st.markdown("### Error Categorization")
        st.markdown("What types of hallucinations does the model produce?")
        if err_df is not None:
            plot_df = err_df.copy()
            plot_df["lang_label"] = plot_df["language"].map(LANG_LABELS)
            plot_df["task_label"] = plot_df["task"].map(TASK_LABELS)
            plot_df["cell"] = plot_df["lang_label"] + " / " + plot_df["task_label"].str.split("(").str[0].str.strip()
            plot_df["category"] = plot_df["category"].replace({
                "fabricated": "Fabricated",
                "incoherent": "Incoherent",
                "other": "Other",
                "wrong_answer": "Wrong Answer",
            })

            fig_err = px.bar(
                plot_df, x="cell", y="count", color="category",
                barmode="stack", text="count",
                color_discrete_map={
                    "Fabricated": "#e74c3c",
                    "Incoherent": "#e67e22",
                    "Wrong Answer": "#3498db",
                    "Other": "#95a5a6",
                },
                labels={"cell": "Cell", "count": "Count", "category": "Error Type"},
                height=440,
            )
            fig_err.update_traces(textposition="inside", texttemplate="%{text}")
            fig_err.update_layout(margin=dict(t=10, b=80), legend_title_text="Error Type",
                                  xaxis_tickangle=-30)
            st.plotly_chart(fig_err, use_container_width=True)

            if err_pivot is not None:
                piv = err_pivot.copy()
                piv["lang_label"] = piv["language"].map(LANG_LABELS)
                piv["task_label"] = piv["task"].map(TASK_LABELS)
                piv = piv[["task_label", "lang_label", "fabricated", "incoherent", "wrong_answer", "other", "TOTAL"]]
                piv.columns = ["Task", "Language", "Fabricated", "Incoherent", "Wrong Answer", "Other", "Total"]
                st.dataframe(piv.reset_index(drop=True), use_container_width=True, hide_index=True)
            st.caption("Swahili XCOPA: 40/148 hallucinations (27%) are incoherent — the model rambles rather than giving a wrong answer.")
        else:
            st.info("Run src/10_error_analysis.py to generate this data.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Example Browser
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Example Browser":
    st.title("Example Browser")
    st.caption("Browse individual model responses and judge labels.")

    col1, col2, col3 = st.columns(3)
    task_filter  = col1.selectbox("Task",     ["All"] + list(TASK_LABELS.values()))
    lang_filter  = col2.selectbox("Language", ["All"] + list(LANG_LABELS.values()))
    label_filter = col3.selectbox("Label",    ["All", "Hallucinated", "Faithful"])

    filtered = df.copy()
    if task_filter  != "All": filtered = filtered[filtered["task_label"]  == task_filter]
    if lang_filter  != "All": filtered = filtered[filtered["lang_label"]  == lang_filter]
    if label_filter != "All": filtered = filtered[filtered["label"]       == label_filter]

    st.markdown(f"**{len(filtered)} examples** match your filters.")

    table_cols = ["task_label", "lang_label", "label", "token_count", "question"]
    display = filtered[table_cols].copy()
    display.columns = ["Task", "Language", "Label", "Tokens", "Question"]
    display["Question"] = display["Question"].astype(str).str[:80] + "…"

    selected_idx = st.dataframe(
        display.reset_index(drop=True),
        use_container_width=True,
        hide_index=False,
        selection_mode="single-row",
        on_select="rerun",
    )

    if selected_idx and selected_idx.get("selection", {}).get("rows"):
        row_idx = selected_idx["selection"]["rows"][0]
        row = filtered.iloc[row_idx]
        st.markdown("---")
        st.markdown(
            f"**Task:** {row['task_label']} &nbsp;|&nbsp; "
            f"**Language:** {row['lang_label']} &nbsp;|&nbsp; "
            f"**Label:** {'🔴 Hallucinated' if row['label'] == 'Hallucinated' else '🟢 Faithful'} &nbsp;|&nbsp; "
            f"**Tokens:** {row['token_count']}"
        )
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Question / Prompt**")
            q = row.get("question", row.get("premise", "—"))
            st.info(str(q))
        with c2:
            st.markdown("**Model Response**")
            st.info(str(row.get("response", "—")))
        st.markdown("**Judge Reason**")
        st.warning(str(row.get("reason", "—")))
    else:
        st.caption("Click a row to see the full response and judge reason.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — Reason Analysis
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Reason Analysis":
    st.title("Reason Analysis")
    st.caption("What does GPT-4o-mini say when it flags a hallucination?")

    hall_df = df[df["label"] == "Hallucinated"].copy()

    CATEGORIES = {
        "Incoherent / off-topic":    ["incoherent", "irrelevant", "off-topic", "unrelated", "confusing", "unclear", "nonsensical"],
        "Wrong answer":              ["incorrect", "wrong", "false", "inaccurate", "mistaken", "not correct", "does not match"],
        "Incomplete / no answer":    ["does not address", "does not answer", "incomplete", "no clear", "fails to"],
        "Fabricated / made-up":      ["fabricat", "made up", "invented", "does not exist", "fictional"],
        "Code-switching / language": ["language", "english", "swahili", "spanish", "italian", "switch", "translat"],
    }

    def categorize(reason):
        reason_lower = str(reason).lower()
        matched = [cat for cat, kws in CATEGORIES.items() if any(kw in reason_lower for kw in kws)]
        return matched[0] if matched else "Other"

    hall_df["category"] = hall_df["reason"].apply(categorize)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### Hallucination Categories (all cells)")
        cat_counts = hall_df["category"].value_counts().reset_index()
        cat_counts.columns = ["Category", "Count"]
        fig_cat = px.bar(
            cat_counts, x="Count", y="Category", orientation="h",
            color="Count", color_continuous_scale="Reds",
            height=350,
        )
        fig_cat.update_layout(margin=dict(t=10, b=10), coloraxis_showscale=False,
                              yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_cat, use_container_width=True)

    with col2:
        st.markdown("### Categories by Cell")
        all_cells = [f"{r['task_label']} / {r['lang_label']}"
                     for _, r in hall_df[["task_label", "lang_label"]].drop_duplicates().iterrows()]
        task_lang_filter = st.selectbox("Filter cell", ["All"] + sorted(all_cells))
        subset = hall_df.copy()
        if task_lang_filter != "All":
            tl, ll = task_lang_filter.split(" / ")
            subset = subset[(subset["task_label"] == tl) & (subset["lang_label"] == ll)]
        cat_sub = subset["category"].value_counts().reset_index()
        cat_sub.columns = ["Category", "Count"]
        fig_sub = px.pie(cat_sub, names="Category", values="Count", hole=0.4, height=350)
        fig_sub.update_layout(margin=dict(t=10, b=10))
        st.plotly_chart(fig_sub, use_container_width=True)

    st.markdown("---")
    st.markdown("### Categories Across All Cells")
    heat_data = hall_df.groupby(["lang_label", "category"]).size().reset_index(name="count")
    heat_pivot = heat_data.pivot(index="lang_label", columns="category", values="count").fillna(0)
    fig_heat = px.imshow(
        heat_pivot, text_auto=True, color_continuous_scale="Oranges",
        labels={"color": "Count"}, height=300,
    )
    fig_heat.update_layout(margin=dict(t=10, b=10))
    st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("---")
    st.markdown("### Browse Judge Reasons")
    cat_select = st.selectbox("Category", ["All"] + list(CATEGORIES.keys()) + ["Other"])
    browse = hall_df if cat_select == "All" else hall_df[hall_df["category"] == cat_select]
    browse = browse[["task_label", "lang_label", "question", "reason"]].copy()
    browse.columns = ["Task", "Language", "Question", "Judge Reason"]
    browse["Question"] = browse["Question"].astype(str).str[:60] + "…"
    st.dataframe(browse.reset_index(drop=True), use_container_width=True, hide_index=True)
