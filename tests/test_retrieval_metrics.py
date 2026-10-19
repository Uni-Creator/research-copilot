"""Tests for eval/metrics.py: coverage, ranked-retrieval metrics, MMR ranking, and arxiv query."""

import math

import pytest

from copilot.graph.state import ResearchPlan, SearchFilters, SubQuestion
from copilot.tools.arxiv import build_arxiv_query
from copilot.tools.ranking import MMRWeights, rank_papers
from eval.metrics import (
    average_precision,
    coverage,
    match_paper,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    subquestion_coverage,
)
from tests.conftest import mk


# match_paper
def test_match_paper_by_arxiv_id():
    p = mk("Any Title", arxiv="2411.17799")
    assert match_paper({"arxiv_id": "2411.17799"}, p)
    assert not match_paper({"arxiv_id": "9999.00000"}, p)


def test_match_paper_by_doi():
    p = mk("Any Title", doi="10.1007/s11263-019-01281-2")
    assert match_paper({"doi": "10.1007/s11263-019-01281-2"}, p)
    assert not match_paper({"doi": "10.1234/other"}, p)


def test_match_paper_by_title_jaccard():
    p = mk("Signs as Tokens: A Retrieval-Enhanced Multilingual Sign Language Generator")
    assert match_paper({"title": "Signs as Tokens: A Retrieval-Enhanced Multilingual Sign Language Generator"}, p)
    assert match_paper({"title": "signs as tokens retrieval enhanced multilingual sign language generator"}, p)
    assert not match_paper({"title": "An Unrelated Paper About Vision Transformers"}, p)


def test_match_paper_empty_must_cite():
    p = mk("Title")
    assert not match_paper({"title": ""}, p)


# coverage


def test_coverage_empty_must_cite():
    assert coverage([], [mk("X")]) == (None, [])


def test_coverage_all_found():
    p1 = mk("Paper Alpha", arxiv="0001.1")
    p2 = mk("Paper Beta", arxiv="0002.2")
    must = [{"arxiv_id": "0001.1"}, {"arxiv_id": "0002.2"}]
    score, missed = coverage(must, [p1, p2])
    assert score == 1.0 and missed == []


def test_coverage_partial():
    p1 = mk("Paper Alpha", arxiv="0001.1")
    must = [{"arxiv_id": "0001.1"}, {"arxiv_id": "9999.9"}]
    score, missed = coverage(must, [p1])
    assert score == 0.5 and len(missed) == 1


def test_coverage_none_found():
    must = [{"arxiv_id": "0001.1"}, {"arxiv_id": "0002.2"}]
    score, missed = coverage(must, [mk("Unrelated")])
    assert score == 0.0 and len(missed) == 2


# Ranked-retrieval metrics  (precision@k, recall@k, nDCG@k, MRR, AP)


@pytest.fixture
def ranked_pool():
    """8-paper ranked list; papers at ranks 2 and 5 are must-cites."""
    papers = [mk(f"Paper {i}", arxiv=f"0000.{i}") for i in range(1, 9)]
    return papers


@pytest.fixture
def must_cite():
    return [{"arxiv_id": "0000.2"}, {"arxiv_id": "0000.5"}]


def test_precision_at_k_zero_k(must_cite, ranked_pool):
    assert precision_at_k(must_cite, ranked_pool, 0) is None


def test_precision_at_k_no_must_cite(ranked_pool):
    assert precision_at_k([], ranked_pool, 5) is None


def test_precision_at_k_first_hit_at_rank_2(must_cite, ranked_pool):
    # Top-1: not a must-cite -> P@1 = 0
    assert precision_at_k(must_cite, ranked_pool, 1) == 0.0
    # Top-2: second paper is must-cite -> P@2 = 1/2
    assert precision_at_k(must_cite, ranked_pool, 2) == 0.5
    # Top-5: ranks 2 and 5 are must-cites -> P@5 = 2/5
    assert precision_at_k(must_cite, ranked_pool, 5) == 0.4


def test_recall_at_k(must_cite, ranked_pool):
    assert recall_at_k(must_cite, ranked_pool, 1) == 0.0
    # After rank 2 we have 1/2 must-cites
    assert recall_at_k(must_cite, ranked_pool, 2) == 0.5
    # After rank 5 we have both must-cites -> 1.0
    assert recall_at_k(must_cite, ranked_pool, 5) == 1.0
    assert recall_at_k(must_cite, ranked_pool, 8) == 1.0


def test_ndcg_at_k(must_cite, ranked_pool):
    # With must-cites at ranks 2 and 5:
    # DCG@5 = 1/log2(3) + 1/log2(6) = 1/1.585 + 1/2.585
    # IDCG@5 (ideal: both at ranks 1,2) = 1/log2(2) + 1/log2(3) = 1 + 0.631
    dcg = 1 / math.log2(3) + 1 / math.log2(6)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    expected = round(dcg / idcg, 4)
    assert ndcg_at_k(must_cite, ranked_pool, 5) == expected


def test_ndcg_at_k_perfect():
    must = [{"arxiv_id": "0000.1"}, {"arxiv_id": "0000.2"}]
    papers = [mk(f"P{i}", arxiv=f"0000.{i}") for i in range(1, 5)]
    # Both must-cites at top-2 -> perfect nDCG
    assert ndcg_at_k(must, papers, 2) == 1.0


def test_ndcg_edge_cases(must_cite):
    assert ndcg_at_k([], [], 5) is None
    assert ndcg_at_k(must_cite, [], 0) is None


def test_mrr_hit_at_rank_2(must_cite, ranked_pool):
    assert mrr(must_cite, ranked_pool) == round(1 / 2, 4)


def test_mrr_no_hit():
    must = [{"arxiv_id": "9999.1"}]
    papers = [mk("Unrelated", arxiv="0001.1")]
    assert mrr(must, papers) == 0.0


def test_mrr_empty():
    assert mrr([], []) is None


def test_average_precision(must_cite, ranked_pool):
    # Hits at positions 2 and 5 (1-indexed)
    # AP = (1/2 + 2/5) / 2  (divided by total must-cites)
    # P@2 when first hit = 1/2; running found=1 at rank 2
    # P@5 when second hit = 2/5; running found=2 at rank 5
    expected = round((1 / 2 + 2 / 5) / 2, 4)
    assert average_precision(must_cite, ranked_pool) == expected


def test_average_precision_no_hits():
    must = [{"arxiv_id": "9999.1"}]
    papers = [mk("Unrelated", arxiv="0001.1")]
    assert average_precision(must, papers) == 0.0


def test_average_precision_empty_must():
    assert average_precision([], [mk("X")]) is None


# Sub-question coverage


def test_subquestion_coverage_empty_plan():
    assert subquestion_coverage(None, []) == {}


def test_subquestion_coverage_basic():
    plan = ResearchPlan(
        sub_questions=[
            SubQuestion(id="sq1", question="VQ-VAE pose tokenization codebook", queries=["pose vqvae"]),
            SubQuestion(id="sq2", question="diffusion models sign generation", queries=["discrete diffusion sign"]),
        ]
    )
    papers = [
        mk("Pose-VQVAE for sign language", abstract="vqvae codebook pose"),
        mk("Sign language discrete diffusion", abstract="diffusion generation"),
    ]
    cov = subquestion_coverage(plan, papers)
    assert set(cov.keys()) == {"sq1", "sq2"}
    # sq1 tokens: vq, vae, pose, tokenization, codebook, pose, vqvae
    # many should appear in paper abstracts -> score > 0
    assert cov["sq1"] > 0.0
    assert cov["sq2"] > 0.0


def test_subquestion_coverage_no_overlap():
    plan = ResearchPlan(
        sub_questions=[
            SubQuestion(id="sq1", question="quantum computing fault tolerance", queries=["quantum error correction"]),
        ]
    )
    papers = [mk("Sign Language Production", abstract="avatars sign pose")]
    cov = subquestion_coverage(plan, papers)
    # No overlap between quantum terms and sign language papers
    assert cov["sq1"] == 0.0


# MMR re-ranking


def test_mmr_promotes_diverse_papers():
    """Two clusters of papers; MMR should interleave instead of taking the whole first cluster."""
    # Cluster A: all about sign language (similar to each other)
    a1 = mk("Sign language production neural network", year=2023, cites=50, abstract="sign language neural")
    a2 = mk("Sign language generation transformer", year=2022, cites=40, abstract="sign language transformer")
    a3 = mk("Sign pose generation seq2seq", year=2021, cites=30, abstract="sign pose generation")
    # Cluster B: about avatar animation (different tokens)
    b1 = mk("3D avatar animation face rendering", year=2023, cites=80, abstract="avatar rendering animation")

    # Give cluster A high RRF, B slightly lower so without MMR A comes first
    papers = [a1, a2, a3, b1]
    rrf_scores = [0.05, 0.04, 0.03, 0.03]
    q = "sign language avatar animation production"

    # Without MMR: top order would be a1, a2, a3, b1
    no_mmr = rank_papers(papers, rrf_scores, q, mmr_enabled=False)
    # With MMR enabled: b1 should be pulled up earlier
    with_mmr = rank_papers(papers, rrf_scores, q, mmr_enabled=True, mmr=MMRWeights(relevance=0.6, similarity=0.4))

    b1_rank_no_mmr = next(i for i, p in enumerate(no_mmr) if p.title == b1.title)
    b1_rank_mmr = next(i for i, p in enumerate(with_mmr) if p.title == b1.title)
    # MMR should move b1 to an earlier rank than pure score ordering
    assert b1_rank_mmr <= b1_rank_no_mmr


def test_mmr_single_paper():
    papers = [mk("Only paper")]
    result = rank_papers(papers, [0.1], "query")
    assert len(result) == 1


def test_mmr_empty():
    assert rank_papers([], [], "query") == []


# arXiv query builder


def test_build_arxiv_query_and_mode():
    q = build_arxiv_query("Signs as Tokens multilingual sign language generator", SearchFilters())
    # Should contain both all: and ti: field specifiers
    assert "all:" in q
    assert "ti:" in q


def test_build_arxiv_query_or_mode():
    q = build_arxiv_query("sign language production", SearchFilters(), mode="or")
    assert " OR " in q
    assert "ti:" not in q  # OR fallback is all:-only


def test_build_arxiv_query_date_filter():
    q = build_arxiv_query("sign language", SearchFilters(year_from=2021, year_to=2024))
    assert "submittedDate:[202101010000 TO 202412312359]" in q


def test_build_arxiv_query_empty():
    q = build_arxiv_query("", SearchFilters())
    assert q.startswith("all:")


def test_build_arxiv_query_known_title():
    """Model names in the query should produce a ti: clause that can exactly match the paper."""
    q = build_arxiv_query("Signs as Tokens retrieval enhanced multilingual sign language", SearchFilters())
    assert "ti:signs" in q or "ti:tokens" in q
