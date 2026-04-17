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
import streamlit as st

# ── Config ────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Cross-Lingual Hallucination Drift",
    page_icon="🌐",
    layout="wide",
)

LANG_LABELS = {"en": "English", "es": "Spanish", "sw": "Swahili"}
TASK_LABELS = {"truthfulqa": "TruthfulQA (Factual QA)", "xcopa": "XCOPA (Commonsense)"}
COLORS      = {"en": "#4C72B0", "es": "#DD8452", "sw": "#55A868"}

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
            "task_label": TASK_LABELS[task], "lang_label": LANG_LABELS[lang],
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


df     = load_labels()
hr_df  = compute_hr(df)

# Cross-task aggregate Φ
tqa_drift  = hr_df[(hr_df["language"] == "es") & (hr_df["task"] == "truthfulqa")]["delta_HR"].values
xcopa_drift = hr_df[(hr_df["language"] == "sw") & (hr_df["task"] == "xcopa")]["delta_HR"].values
phi = round(float(tqa_drift[0]) - float(xcopa_drift[0]), 2) if len(tqa_drift) and len(xcopa_drift) else None

# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.title("🌐 Hallucination Drift")
st.sidebar.caption("CS505 · Boston University")
page = st.sidebar.radio(
    "Navigate",
    ["Dashboard", "Charts", "Example Browser", "Reason Analysis"],
    label_visibility="collapsed",
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Model:** Aya Expanse 8B  \n**Judge:** GPT-4o-mini  \n**Samples:** 150 / split"
)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Dashboard
# ══════════════════════════════════════════════════════════════════════════════

if page == "Dashboard":
    st.title("Cross-Lingual Hallucination Drift")
    st.markdown(
        "Does hallucination drift depend on **task type**? "
        "Comparing Aya Expanse 8B across factual QA (TruthfulQA) and commonsense reasoning (XCOPA)."
    )

    # Top KPI row
    st.markdown("### Key Results")
    cols = st.columns(4)
    kpis = [
        ("TruthfulQA EN→ES drift", f"{hr_df[(hr_df['language']=='es') & (hr_df['task']=='truthfulqa')]['delta_HR'].values[0]:+.2f} pp"),
        ("XCOPA EN→SW drift",      f"{hr_df[(hr_df['language']=='sw') & (hr_df['task']=='xcopa')]['delta_HR'].values[0]:+.2f} pp"),
        ("Aggregate Φ",            f"{phi:+.2f} pp" if phi is not None else "N/A"),
        ("Total labeled examples", f"{len(df):,}"),
    ]
    for col, (label, value) in zip(cols, kpis):
        col.metric(label, value)

    st.markdown("---")

    # HR table
    st.markdown("### Hallucination Rates per Cell")
    display = hr_df[["task_label", "lang_label", "total", "hallucinated", "HR", "delta_HR"]].copy()
    display.columns = ["Task", "Language", "Total", "Hallucinated", "HR (%)", "ΔHR vs EN (pp)"]
    display = display.sort_values(["Task", "Language"])

    def color_hr(val):
        if isinstance(val, float):
            if val > 50:   return "background-color: #f8d7da; color: #721c24"
            if val > 20:   return "background-color: #fff3cd; color: #856404"
            return "background-color: #d4edda; color: #155724"
        return ""

    st.dataframe(
        display.style.map(color_hr, subset=["HR (%)"]),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")

    # Token verbosity finding
    st.markdown("### Token Verbosity: Hallucinated vs Faithful Responses")
    st.caption("When the model hallucinates in Swahili XCOPA, it generates nearly 2× more tokens — a sign of confusion/rambling.")
    tok_rows = []
    for _, row in hr_df.iterrows():
        tok_rows.append({"Task/Language": f"{row['task_label']}\n({row['lang_label']})",
                         "Hallucinated": row["avg_tokens_hall"],
                         "Faithful": row["avg_tokens_faith"]})
    tok_df = pd.DataFrame(tok_rows)
    tok_melt = tok_df.melt(id_vars="Task/Language", var_name="Label", value_name="Avg Tokens")
    fig_tok = px.bar(
        tok_melt, x="Task/Language", y="Avg Tokens", color="Label",
        barmode="group", color_discrete_map={"Hallucinated": "#e74c3c", "Faithful": "#2ecc71"},
        height=350,
    )
    fig_tok.update_layout(margin=dict(t=10, b=10), legend_title_text="")
    st.plotly_chart(fig_tok, use_container_width=True)

    # Φ explanation
    st.markdown("---")
    st.markdown("### Drift Interaction Score (Φ)")
    st.markdown(
        f"""
        **Φ = ΔHR(es, TruthfulQA) − ΔHR(sw, XCOPA) = {phi:+.2f} pp**

        A large negative Φ means XCOPA drift dominates TruthfulQA drift — the model's hallucination
        problem is far worse on commonsense reasoning in Swahili than on factual QA in Spanish.
        This supports the hypothesis that **cross-lingual drift is task-dependent**.
        """
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Charts
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Charts":
    st.title("Charts")

    # Chart 1: HR by language/task
    st.markdown("### Hallucination Rate by Language and Task")
    fig1 = px.bar(
        hr_df, x="lang_label", y="HR", color="language",
        facet_col="task_label", text="HR",
        color_discrete_map=COLORS,
        labels={"lang_label": "Language", "HR": "Hallucination Rate (%)", "task_label": "Task"},
        height=420,
    )
    fig1.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig1.update_layout(showlegend=False, margin=dict(t=40, b=10))
    fig1.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("---")

    # Chart 2: ΔHR drift
    st.markdown("### Cross-Lingual Drift (ΔHR vs English)")
    non_en = hr_df[hr_df["language"] != "en"].copy()
    fig2 = px.bar(
        non_en, x="lang_label", y="delta_HR", color="task_label",
        barmode="group", text="delta_HR",
        color_discrete_sequence=["#4C72B0", "#DD8452"],
        labels={"lang_label": "Language", "delta_HR": "ΔHR vs English (pp)", "task_label": "Task"},
        height=420,
    )
    fig2.update_traces(texttemplate="%{text:+.1f}", textposition="outside")
    fig2.add_hline(y=0, line_dash="dash", line_color="black", line_width=1)
    fig2.update_layout(margin=dict(t=10, b=10), legend_title_text="Task")
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    # Chart 3: HR heatmap
    st.markdown("### Hallucination Rate Heatmap")
    pivot = hr_df.pivot(index="lang_label", columns="task_label", values="HR")
    fig3 = px.imshow(
        pivot, text_auto=".1f", color_continuous_scale="RdYlGn_r",
        labels={"color": "HR (%)"},
        height=300,
    )
    fig3.update_layout(margin=dict(t=10, b=10))
    st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Example Browser
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

    # Table view
    table_cols = ["task_label", "lang_label", "label", "token_count", "question"]
    display = filtered[table_cols].copy()
    display.columns = ["Task", "Language", "Label", "Tokens", "Question"]
    display["Question"] = display["Question"].str[:80] + "…"

    selected_idx = st.dataframe(
        display.reset_index(drop=True),
        use_container_width=True,
        hide_index=False,
        selection_mode="single-row",
        on_select="rerun",
    )

    # Detail panel
    if selected_idx and selected_idx.get("selection", {}).get("rows"):
        row_idx = selected_idx["selection"]["rows"][0]
        row = filtered.iloc[row_idx]
        st.markdown("---")
        st.markdown(f"**Task:** {row['task_label']} &nbsp;|&nbsp; **Language:** {row['lang_label']} &nbsp;|&nbsp; **Label:** {'🔴 Hallucinated' if row['label'] == 'Hallucinated' else '🟢 Faithful'}")
        st.markdown(f"**Tokens:** {row['token_count']}")

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
# PAGE 4 — Reason Analysis
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Reason Analysis":
    st.title("Reason Analysis")
    st.caption("What does GPT-4o-mini say when it flags a hallucination?")

    hall_df = df[df["label"] == "Hallucinated"].copy()

    # Keyword categories
    CATEGORIES = {
        "Incoherent / off-topic":     ["incoherent", "irrelevant", "off-topic", "unrelated", "confusing", "unclear", "nonsensical"],
        "Wrong answer":               ["incorrect", "wrong", "false", "inaccurate", "mistaken", "not correct", "does not match"],
        "Incomplete / no answer":     ["does not address", "does not answer", "incomplete", "no clear", "fails to"],
        "Fabricated / made-up":       ["fabricat", "made up", "invented", "does not exist", "fictional"],
        "Code-switching / language":  ["language", "english", "swahili", "spanish", "switch", "translat"],
    }

    def categorize(reason):
        reason_lower = str(reason).lower()
        matched = [cat for cat, kws in CATEGORIES.items() if any(kw in reason_lower for kw in kws)]
        return matched[0] if matched else "Other"

    hall_df["category"] = hall_df["reason"].apply(categorize)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### Hallucination Categories (all splits)")
        cat_counts = hall_df["category"].value_counts().reset_index()
        cat_counts.columns = ["Category", "Count"]
        fig_cat = px.bar(
            cat_counts, x="Count", y="Category", orientation="h",
            color="Count", color_continuous_scale="Reds",
            height=350,
        )
        fig_cat.update_layout(margin=dict(t=10, b=10), coloraxis_showscale=False, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_cat, use_container_width=True)

    with col2:
        st.markdown("### Categories by Task & Language")
        task_lang_filter = st.selectbox(
            "Filter split",
            ["All"] + [f"{r['task_label']} / {r['lang_label']}" for _, r in
                       hall_df[["task_label","lang_label"]].drop_duplicates().iterrows()],
        )
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
    st.markdown("### Browse Judge Reasons")
    cat_select = st.selectbox("Category", ["All"] + list(CATEGORIES.keys()) + ["Other"])
    browse = hall_df if cat_select == "All" else hall_df[hall_df["category"] == cat_select]
    browse = browse[["task_label", "lang_label", "question", "reason"]].copy()
    browse.columns = ["Task", "Language", "Question", "Judge Reason"]
    browse["Question"] = browse["Question"].astype(str).str[:60] + "…"
    st.dataframe(browse.reset_index(drop=True), use_container_width=True, hide_index=True)
