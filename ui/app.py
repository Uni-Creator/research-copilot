"""Streamlit front end for the research copilot.   streamlit run ui/app.py

Runs the pipeline in-process (one container, no API needed) and streams each agent step live.
Set COPILOT_UI_PASSWORD to gate a public deployment: every run spends LLM tokens.
"""

from __future__ import annotations

import hmac
import os
import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent)
)  # `streamlit run` puts ui/ on the path, not the repo root

import streamlit as st  # noqa: E402

from copilot.config import Settings  # noqa: E402
from copilot.factory import CopilotFactory  # noqa: E402
from copilot.graph.state import SearchFilters  # noqa: E402
from copilot.progress import format_event_md  # noqa: E402
from copilot.render import dumps_state, render_html, render_markdown  # noqa: E402

STAGES = {
    "Full review (plan, read, write, verify)": "critic",
    "No verification (plan, read, write)": "write",
    "Read papers only": "read",
    "Search only (fast, cheap)": "retrieve",
}
VERDICT_ICON = {"supported": "✅ supported", "partial": "🟡 partial", "unsupported": "❌ unsupported"}

st.set_page_config(page_title="Research Copilot", page_icon="📚", layout="wide")


@st.cache_resource
def get_factory() -> CopilotFactory:
    return CopilotFactory(Settings.from_env())


def gate() -> None:
    password = os.environ.get("COPILOT_UI_PASSWORD")
    if password and not st.session_state.get("authed"):
        entered = st.text_input("Password", type="password", key="password")
        if entered and hmac.compare_digest(entered.encode(), password.encode()):
            st.session_state["authed"] = True
            st.rerun()
        st.stop()


def run_pipeline(factory, question: str, filters: SearchFilters, until: str, fulltext: bool) -> None:
    st.session_state.pop("result", None)
    with st.status("Agents working…", expanded=True) as status:
        try:
            # each agent step (plan, searches, papers chosen, reading, fact-checks...) is shown as it happens
            final = factory(until, fulltext).run_live(
                question, filters, on_event=lambda ev: st.markdown(format_event_md(ev))
            )
            st.session_state["result"] = final
            status.update(label="Done", state="complete", expanded=False)
        except Exception as exc:
            status.update(label="Failed", state="error")
            st.error(f"{type(exc).__name__}: {exc}")


def show_result(result: dict) -> None:
    papers, notes = result.get("papers", {}), result.get("notes", {})
    draft, crit, usage = result.get("draft"), result.get("critique"), result.get("usage")

    c = st.columns(5)
    c[0].metric("Papers shortlisted", len(result.get("shortlist", [])))
    c[1].metric("Papers read", len(notes))
    c[2].metric("Claims verified", len(crit.checks) if crit else "-")
    c[3].metric("Support rate", f"{crit.support_rate:.0%}" if crit and crit.support_rate is not None else "-")
    c[4].metric("Tokens", f"{usage.total_tokens:,}" if usage else "0")
    if crit and not crit.passed:
        st.warning(
            "The Critic's quality bar was not fully met; check the Verification tab before relying on this draft."
        )
    if errors := result.get("errors"):
        with st.expander(f"{len(errors)} warning(s)"):
            for e in errors:
                st.write(f"- {e}")

    report_tab, papers_tab, verify_tab, trace_tab, dl_tab = st.tabs(
        ["Report", "Papers", "Verification", "Trace", "Download"]
    )

    with report_tab:
        if draft:
            st.markdown(render_markdown(result))
        else:
            st.info("No report: the pipeline stopped before the writer. See the Papers tab.")
    with papers_tab:
        rows = []
        for rank, pid in enumerate(result.get("shortlist", []), 1):
            p, n = papers[pid], notes.get(pid)
            rows.append(
                {
                    "#": rank,
                    "title": p.title,
                    "year": p.year,
                    "citations": p.citation_count,
                    "score": p.score,
                    "method": n.method if n else "",
                    "datasets": ", ".join(n.datasets) if n else "",
                    "link": p.url,
                }
            )
        st.dataframe(
            rows, use_container_width=True, hide_index=True, column_config={"link": st.column_config.LinkColumn("link")}
        )
    with verify_tab:
        if crit and crit.checks:
            st.caption("Each cited sentence is checked against only the passage(s) it cites.")
            st.dataframe(
                [
                    {
                        "verdict": VERDICT_ICON[k.verdict],
                        "claim": k.claim,
                        "reason": k.reason,
                        "evidence": ", ".join(k.chunk_ids),
                    }
                    for k in crit.checks
                ],
                use_container_width=True,
                hide_index=True,
            )
            if crit.missing_topics:
                st.write("Gaps flagged: " + "; ".join(crit.missing_topics))
        else:
            st.info("No verification data for this run.")
    with trace_tab:
        st.dataframe(
            [
                {"time": e.ts[11:19], "agent": e.agent, "kind": e.kind, "event": e.message}
                for e in sorted(result.get("trace", []), key=lambda e: e.ts)
            ],
            use_container_width=True,
            hide_index=True,
        )
    with dl_tab:
        md = render_markdown(result)
        st.download_button("Report (Markdown)", md, "report.md", "text/markdown", key="dl_md")
        st.download_button(
            "Report (HTML)", render_html(md, result["question"]), "report.html", "text/html", key="dl_html"
        )
        st.download_button("Full run (JSON)", dumps_state(result), "run.json", "application/json", key="dl_json")


def main() -> None:
    st.title("📚 Research Copilot")
    st.caption(
        "A team of agents plans the search, reads the papers, writes a cited review, "
        "and fact-checks every citation against the source text."
    )
    gate()

    override = st.session_state.get("factory_override")  # test hook: inject a pre-built factory
    factory = override or get_factory()
    if override is None and not os.environ.get("ANTHROPIC_API_KEY"):
        st.warning("ANTHROPIC_API_KEY is not set; runs will fail at the first LLM call.")

    with st.sidebar:
        st.header("Options")
        mode = st.selectbox("Pipeline", list(STAGES), key="mode")
        y_from = st.number_input("Year from (0 = any)", 0, 2100, 0, key="year_from")
        y_to = st.number_input("Year to (0 = any)", 0, 2100, 0, key="year_to")
        cap = int(os.environ.get("COPILOT_UI_MAX_PAPERS", 30))
        max_papers = st.slider("Papers to read", 3, cap, min(12, cap), key="max_papers")
        fulltext = st.checkbox(
            "Read full PDFs", True, key="fulltext", help="Off = abstracts only: much faster and cheaper, but shallower."
        )

    question = st.text_area(
        "Research question",
        key="question",
        height=90,
        placeholder="e.g. What are the main approaches to gloss-free sign language production?",
    )
    if st.button("Run", type="primary", key="run"):
        if len(question.strip()) < 10:
            st.error("Please enter a research question (at least 10 characters).")
        else:
            run_pipeline(
                factory,
                question.strip(),
                SearchFilters(year_from=y_from or None, year_to=y_to or None, max_papers=max_papers),
                STAGES[mode],
                fulltext,
            )

    if "result" in st.session_state:
        show_result(st.session_state["result"])


main()
