"""
DataLens — Dataset Recommendation System
Uses native Streamlit components to avoid HTML rendering issues.
"""

import streamlit as st
import pandas as pd
import os

from scanner import scan_folder
from profiler import profile_dataset
from matcher import score_all, find_similar_datasets

st.set_page_config(page_title="DataLens", page_icon="📂",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
[data-testid="collapsedControl"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
  --bg:       #080c14;
  --surface:  #0e1420;
  --surface2: #141d2e;
  --border:   #1e2d45;
  --border2:  #263550;
  --text:     #e2e8f4;
  --muted:    #5a7090;
  --accent:   #3b82f6;
  --accent2:  #60a5fa;
  --green:    #22c55e;
  --yellow:   #f59e0b;
  --red:      #ef4444;
  --purple:   #a78bfa;
}

html, body, [class*="css"] {
  font-family: 'Space Grotesk', sans-serif !important;
  background: var(--bg) !important;
  color: var(--text) !important;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2rem 4rem !important; max-width: 1400px; }

section[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] small,
section[data-testid="stSidebar"] div { color: var(--text) !important; }

.stTextInput input, .stTextArea textarea {
  background: var(--surface2) !important;
  border: 1px solid var(--border2) !important;
  color: var(--text) !important;
  border-radius: 8px !important;
  font-family: 'Space Grotesk', sans-serif !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 2px rgba(59,130,246,0.15) !important;
}
.stButton > button {
  background: var(--accent) !important;
  color: #fff !important;
  border: none !important;
  border-radius: 8px !important;
  font-family: 'Space Grotesk', sans-serif !important;
  font-weight: 600 !important;
  transition: all 0.2s !important;
}
.stButton > button:hover {
  background: var(--accent2) !important;
  transform: translateY(-1px);
  box-shadow: 0 4px 20px rgba(59,130,246,0.35) !important;
}

/* Metric cards */
[data-testid="metric-container"] {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
  padding: 1rem 1.2rem !important;
}
[data-testid="metric-container"] label {
  font-size: 0.7rem !important;
  text-transform: uppercase !important;
  letter-spacing: 0.1em !important;
  color: var(--muted) !important;
}
[data-testid="stMetricValue"] {
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 1.8rem !important;
  color: var(--accent2) !important;
}

/* Expander */
.streamlit-expanderHeader {
  background: var(--surface2) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  color: var(--muted) !important;
  font-size: 0.82rem !important;
}
.streamlit-expanderContent {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-top: none !important;
}
.stDataFrame { border-radius: 8px; overflow: hidden; }

/* Divider */
hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }

/* Horizontal rule for cards */
.card-rule { border: none; border-top: 1px solid var(--border); margin: 0.6rem 0; }

/* Score bar */
.bar-wrap { background: var(--surface2); border-radius: 99px; height: 5px; margin-top: 6px; }
.bar-fill  { height: 5px; border-radius: 99px; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
for key in ["results", "dupes", "query_used"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔎 DataLens")
    st.caption("Dataset Recommendation System")
    st.divider()

    st.markdown("**📁 Dataset Folder**")
    folder_path = st.text_input("folder", label_visibility="collapsed",
                                placeholder="C:\\Users\\you\\datasets")

    st.markdown("**🔍 Your Query**")
    user_query = st.text_area("query", label_visibility="collapsed", height=120,
                              placeholder="e.g. I want to detect unusual money movements between accounts")

    st.divider()
    run_btn = st.button("🔍  Find Matching Datasets", use_container_width=True, type="primary")

    st.divider()
    st.caption("**Supported formats**")
    st.caption("📄 CSV &nbsp;|&nbsp; 📊 Excel &nbsp;|&nbsp; 🗄️ SQL")
    st.divider()
    st.caption("🤖 **AI-powered** — Groq LLM understands your intent, not just keywords.")


# ── Page title ────────────────────────────────────────────────────────────────
st.markdown("# Dataset Recommendation")
st.caption("Describe what you need — DataLens ranks every dataset by column-level relevance to your query.")
st.divider()

# ── Run ───────────────────────────────────────────────────────────────────────
if run_btn:
    if not folder_path or not os.path.isdir(folder_path):
        st.error("⚠️ Enter a valid folder path in the sidebar.")
        st.stop()
    if not user_query or len(user_query.strip()) < 3:
        st.error("⚠️ Describe what you're looking for in the query box.")
        st.stop()

    with st.spinner("Scanning folder…"):
        raw_datasets = scan_folder(folder_path)

    if not raw_datasets:
        st.warning("No supported files found (CSV / Excel / SQL).")
        st.stop()

    valid  = [d for d in raw_datasets if d["df"] is not None]
    failed = [d for d in raw_datasets if d["df"] is None]

    if failed:
        with st.expander(f"⚠️ {len(failed)} file(s) could not be read"):
            for f in failed:
                st.code(f"{f['filename']}: {f.get('error', 'unknown error')}")

    if not valid:
        st.error("No readable datasets found.")
        st.stop()

    with st.spinner(f"Profiling {len(valid)} datasets…"):
        profiles = [profile_dataset(d["filename"], d["df"]) for d in valid]

    with st.spinner("Matching with AI — understanding your intent…"):
        results = score_all(profiles, user_query.strip())
        dupes   = find_similar_datasets(profiles)

    st.session_state.results    = results
    st.session_state.dupes      = dupes
    st.session_state.query_used = user_query.strip()

# ── Display ───────────────────────────────────────────────────────────────────
if st.session_state.results:
    results    = st.session_state.results
    dupes      = st.session_state.dupes
    query_used = st.session_state.query_used

    n_rec     = sum(1 for r in results if r["recommended"])
    top_score = results[0]["score"] if results else 0
    n_dupes   = len(dupes)

    # Query echo
    st.info(f'🔍 Query: "{query_used}"')
    st.success("🤖 Groq LLM active — results reflect semantic understanding, not just keyword matching.")

    # Stat metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Datasets Scanned", len(results))
    c2.metric("Relevant Matches", n_rec)
    c3.metric("Near-Duplicates",  n_dupes)
    c4.metric("Top Score",        f"{top_score:.0f}/100")

    st.divider()

    col_main, col_side = st.columns([3, 1], gap="large")

    # ── Main results column ───────────────────────────────────────────────────
    with col_main:
        st.markdown("#### Ranked Results")
        show_all = st.toggle("Show low-relevance datasets too", value=False)
        st.markdown("")

        displayed = 0
        for r in results:
            if not show_all and not r["recommended"]:
                continue
            displayed += 1

            score   = r["score"]
            fname   = r["filename"]
            src     = r.get("source", "file")
            rec     = r["recommended"]

            # Score colour
            if score >= 40:
                score_color = "🟢"
                bar_color   = "#22c55e"
            elif score >= 20:
                score_color = "🟡"
                bar_color   = "#f59e0b"
            else:
                score_color = "⚫"
                bar_color   = "#5a7090"

            # Source label
            if src == "sql":
                src_label = "🗄️ SQL table"
            elif fname.lower().endswith((".xlsx", ".xls")):
                src_label = "📊 Excel"
            else:
                src_label = "📄 CSV"

            rec_icon = "✅" if rec else "➖"

            # Duplicate warning
            dup_warning = ""
            for dp in dupes:
                if fname in (dp["dataset_a"], dp["dataset_b"]):
                    other = dp["dataset_b"] if fname == dp["dataset_a"] else dp["dataset_a"]
                    dup_warning = f"⚠️ {dp['similarity']}% similar to `{other}`"
                    break

            # ── Card using native Streamlit inside a container ────────────────
            with st.container(border=True):
                h1, h2 = st.columns([3, 1])
                with h1:
                    st.markdown(f"**{rec_icon} &nbsp; {fname}**")
                    st.caption(src_label)
                with h2:
                    st.markdown(
                        f"<p style='text-align:right;font-family:JetBrains Mono,monospace;"
                        f"font-size:1.4rem;font-weight:700;margin:0;color:"
                        f"{'#22c55e' if score>=40 else '#f59e0b' if score>=20 else '#5a7090'};'>"
                        f"{score:.1f}<span style='font-size:0.6em;opacity:0.6;'>/100</span></p>",
                        unsafe_allow_html=True
                    )

                # Meta row
                m1, m2, m3 = st.columns(3)
                m1.caption(f"🗂 {r['num_cols']} columns")
                m2.caption(f"TF-IDF: {r['tfidf_score']:.1f}")
                m3.caption(f"Keyword: {r['overlap_score']:.1f}")

                # Matched columns
                matched = r.get("matched_columns", [])
                if matched:
                    st.caption("**Matched columns:** " + "  `" + "`   `".join(matched[:8]) + "`")

                # Dup warning
                if dup_warning:
                    st.warning(dup_warning, icon="⚠️")

                # LLM info
                if r.get("llm_score") is not None:
                    st.caption(f"🤖 LLM score: {r['llm_score']:.0f}/100")
                if r.get("llm_reason") and not str(r.get("llm_reason", "")).startswith("LLM unavailable"):
                    st.caption(f"💬 _{r['llm_reason']}_")

                # Progress bar
                bar_w = min(int(score * 2), 100)
                st.markdown(
                    f"<div class='bar-wrap'><div class='bar-fill' "
                    f"style='width:{bar_w}%;background:{bar_color};'></div></div>",
                    unsafe_allow_html=True
                )

                # Column details expander
                with st.expander("🔬 View all columns"):
                    df_cols = pd.DataFrame(r["columns"])[
                        ["name", "semantic_type", "dtype", "null_pct", "unique_count"]
                    ].rename(columns={
                        "name": "Column", "semantic_type": "Semantic Type",
                        "dtype": "Type", "null_pct": "Null %", "unique_count": "Unique"
                    })
                    st.dataframe(df_cols, use_container_width=True, hide_index=True)

        if displayed == 0:
            st.info("No strong matches found. Toggle 'Show low-relevance datasets' or rephrase your query.")

    # ── Side column ───────────────────────────────────────────────────────────
    with col_side:
        st.markdown("#### All Datasets")
        for r in results:
            score = r["score"]
            icon  = "🗄️" if r.get("source") == "sql" else "📄"
            name  = r["filename"]
            short = (name[:22] + "…") if len(name) > 22 else name

            if score >= 40:   clr = "green"
            elif score >= 20: clr = "orange"
            else:             clr = "grey"

            sc1, sc2 = st.columns([3, 1])
            sc1.caption(f"{icon} {short}")
            sc2.markdown(
                f"<p style='text-align:right;font-family:JetBrains Mono,monospace;"
                f"font-size:0.8rem;font-weight:700;color:"
                f"{'#22c55e' if clr=='green' else '#f59e0b' if clr=='orange' else '#5a7090'};margin:0;'>"
                f"{score:.0f}</p>",
                unsafe_allow_html=True
            )

        if dupes:
            st.divider()
            st.markdown("#### ⚠️ Duplicates")
            for dp in dupes:
                with st.container(border=True):
                    st.caption(f"**{dp['similarity']}% similar**")
                    st.caption(dp["dataset_a"])
                    st.caption(f"↔ {dp['dataset_b']}")

    # ── Export ────────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("#### Export Results")

    report_rows = [{
        "Dataset":         r["filename"],
        "Source":          r.get("source", "file").upper(),
        "Score":           r["score"],
        "TF-IDF":          r["tfidf_score"],
        "Keyword":         r["overlap_score"],
        "LLM Score":       r.get("llm_score") or "",
        "LLM Reason":      r.get("llm_reason") or "",
        "Relevant":        "Yes" if r["recommended"] else "No",
        "Matched Columns": ", ".join(r.get("matched_columns", [])),
        "Total Columns":   r["num_cols"],
        "Query":           query_used,
    } for r in results]

    report_df = pd.DataFrame(report_rows)
    dl_col, tbl_col = st.columns([1, 3])
    with dl_col:
        st.download_button(
            "⬇️ Download CSV Report",
            data=report_df.to_csv(index=False).encode(),
            file_name="datalens_results.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with tbl_col:
        with st.expander("📋 View full results table"):
            st.dataframe(report_df, use_container_width=True, hide_index=True)

# ── Empty state ───────────────────────────────────────────────────────────────
else:
    st.markdown("<br><br>", unsafe_allow_html=True)
    ec1, ec2, ec3 = st.columns([1, 2, 1])
    with ec2:
        st.markdown("## 🔎")
        st.markdown("### Ready to scan")
        st.markdown("""
Enter your **folder path** and describe what you need in the sidebar.

DataLens will rank every dataset — CSV, Excel, or SQL table — by how well
its **column names** match your intent.

Powered by **Groq LLM** — understands vague queries like *"unusual money movement"*
or *"predict who might leave"* without needing exact keyword matches.
        """)
        st.markdown("")
        st.info("💡 **Tip:** SQL files are split automatically — each table is scored separately.")
