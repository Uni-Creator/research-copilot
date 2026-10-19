from pathlib import Path

import pytest

from copilot.factory import CopilotFactory
from tests.test_full_graph import SETTINGS, overrides


def test_factory_shares_store_and_applies_fulltext_flag():
    kw, *_ = overrides()
    factory = CopilotFactory(SETTINGS, **kw)
    a, b = factory("critic", True), factory("write", True)
    assert a.store is b.store and a.until == "critic" and b.until == "write"  # one store for the process
    assert factory("retrieve", True).store is None  # retrieval needs no store

    kw2, _, _, fetched = overrides()
    factory2 = CopilotFactory(SETTINGS, **kw2)
    final = factory2("read", fulltext=False).run("How is sign language produced?")
    assert fetched == [] and set(final["notes"]) == set(final["shortlist"])  # abstracts only: no PDF fetched


def test_ui_end_to_end_with_streamlit_apptest():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    kw, *_ = overrides()
    at = AppTest.from_file(str(Path(__file__).parent.parent / "ui" / "app.py"), default_timeout=60)
    at.session_state["factory_override"] = CopilotFactory(SETTINGS, **kw)
    at.run()
    assert not at.exception

    at.button(key="run").click().run()  # too-short question is rejected
    assert any("at least 10 characters" in e.value for e in at.error)

    at.text_area(key="question").input("How is sign language produced?")
    at.button(key="run").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error

    labels = [m.label for m in at.metric]
    assert "Papers shortlisted" in labels and "Support rate" in labels
    support = next(m for m in at.metric if m.label == "Support rate")
    assert support.value == "100%"
    assert any("## References" in m.value or "References" in m.value for m in at.markdown)
    assert at.session_state["result"]["draft"].version == 2
