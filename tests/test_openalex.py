from __future__ import annotations

from copilot.tools.openalex import parse_openalex_work, reconstruct_abstract, year_filter
from copilot.graph.state import SearchFilters


def test_reconstruct_abstract_uses_word_positions() -> None:
    assert reconstruct_abstract({"sign": [0], "language": [1], "generation": [2]}) == "sign language generation"


def test_reconstruct_abstract_empty() -> None:
    assert reconstruct_abstract(None) == ""
    assert reconstruct_abstract({}) == ""


def test_year_filter_range_and_bounds() -> None:
    assert year_filter(SearchFilters(year_from=2020, year_to=2024)) == (
        "from_publication_date:2020-01-01,to_publication_date:2024-12-31"
    )
    assert year_filter(SearchFilters(year_from=2020)) == "from_publication_date:2020-01-01"
    assert year_filter(SearchFilters(year_to=2024)) == "to_publication_date:2024-12-31"
    assert year_filter(SearchFilters()) is None


def test_parse_openalex_work_maps_metadata() -> None:
    work = {
        "id": "https://openalex.org/W123",
        "doi": "https://doi.org/10.1234/example",
        "title": "Sign Generation",
        "publication_year": 2024,
        "authorships": [{"author": {"display_name": "Ada Researcher"}}],
        "abstract_inverted_index": {"Sign": [0], "generation": [1]},
        "cited_by_count": 12,
        "primary_location": {
            "source": {"display_name": "Example Journal"},
            "landing_page_url": "https://example.org/paper",
            "pdf_url": "https://example.org/paper.pdf",
        },
        "best_oa_location": None,
        "locations": [],
        "ids": {"openalex": "https://openalex.org/W123", "doi": "https://doi.org/10.1234/example"},
    }
    paper = parse_openalex_work(work, "sign generation")
    assert paper is not None
    assert paper.title == "Sign Generation"
    assert paper.doi == "10.1234/example"
    assert paper.year == 2024
    assert paper.authors == ["Ada Researcher"]
    assert paper.abstract == "Sign generation"
    assert paper.citation_count == 12
    assert paper.sources == ["openalex"]
    assert paper.pdf_url == "https://example.org/paper.pdf"
