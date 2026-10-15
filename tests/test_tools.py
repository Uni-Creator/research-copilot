from copilot.graph.state import SearchFilters
from copilot.tools.arxiv import build_arxiv_query, parse_arxiv_feed
from copilot.tools.ranking import rank_papers
from copilot.tools.semantic_scholar import parse_s2_item, year_param
from tests.conftest import mk

FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2004.14874v2</id>
    <published>2020-04-30T10:00:00Z</published>
    <title>Progressive Transformers for
      End-to-End Sign Language Production</title>
    <summary>  We propose a model.  </summary>
    <author><name>Ben Saunders</name></author><author><name>Necati Cihan Camgoz</name></author>
    <link href="http://arxiv.org/abs/2004.14874v2" rel="alternate"/>
    <link title="pdf" href="http://arxiv.org/pdf/2004.14874v2" rel="related"/>
    <arxiv:doi>10.1007/978-3-030-58621-8_40</arxiv:doi>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/cs/0112017v1</id>
    <published>2001-12-01T00:00:00Z</published>
    <title>Old style id</title><summary>x</summary>
  </entry>
</feed>"""


def test_parse_arxiv_feed():
    a, b = parse_arxiv_feed(FEED, "sign language production")
    assert a.arxiv_id == "2004.14874" and a.year == 2020
    assert a.title == "Progressive Transformers for End-to-End Sign Language Production"
    assert a.authors == ["Ben Saunders", "Necati Cihan Camgoz"] and a.doi.startswith("10.1007")
    assert a.abstract == "We propose a model." and a.found_by == ["sign language production"]
    assert b.arxiv_id == "cs/0112017" and b.pdf_url == "https://arxiv.org/pdf/cs/0112017"


def test_build_arxiv_query():
    q = build_arxiv_query("the sign language production approaches", SearchFilters())
    assert q == "all:sign AND all:language AND all:production"
    assert build_arxiv_query("sign language", SearchFilters(), "or") == "all:sign OR all:language"
    dated = build_arxiv_query("sign language", SearchFilters(year_from=2021, year_to=2024))
    assert "submittedDate:[202101010000 TO 202412312359]" in dated


def test_s2_parsing():
    p = parse_s2_item({
        "paperId": "abc", "title": "T", "abstract": None, "year": 2022, "citationCount": 5,
        "authors": [{"name": "A"}, {"name": None}], "externalIds": {"ArXiv": "2201.1", "DOI": "10.1/x"},
        "openAccessPdf": None, "venue": "",
    }, "q")
    assert p.paper_id == "arxiv:2201.1" and p.abstract == "" and p.authors == ["A"]
    assert p.pdf_url == "https://arxiv.org/pdf/2201.1" and p.venue is None
    assert parse_s2_item({"title": "  "}) is None
    assert year_param(SearchFilters(year_from=2020)) == "2020-"
    assert year_param(SearchFilters(year_to=2024)) == "-2024"
    assert year_param(SearchFilters()) is None


def test_ranking_prefers_multi_query_relevant_cited_papers():
    q = "gloss-free sign language production"
    strong = mk("Gloss-free sign language production", year=2024, cites=100, abstract="sign language production")
    weak = mk("Unrelated vision paper", year=2015, cites=0)
    ranked = rank_papers([weak, strong], [0.01, 0.05], q, now_year=2025)
    assert [p.title for p in ranked][0] == strong.title
    assert ranked[0].score > ranked[1].score and 0 <= ranked[1].score <= 1
    assert rank_papers([], [], q) == []
