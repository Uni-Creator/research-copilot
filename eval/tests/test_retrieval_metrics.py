import unittest

from eval.retrieval_metrics import (
    graded_relevance_metrics,
    judged_aspect_coverage,
    normalize_arxiv,
    normalize_doi,
    normalize_title,
    paper_matches_entry,
)


class RetrievalMetricsTests(unittest.TestCase):

    def test_normalize_title(self):
        self.assertEqual(normalize_title("Signs as Tokens: A Retrieval-Enhanced Model!"), (
            "signs as tokens a retrieval enhanced model"
        ))
        self.assertEqual(normalize_title(None), "")
        self.assertEqual(normalize_title("  TEXT   to Sign "), "text to sign")

    def test_identifier_normalization(self):
        self.assertEqual(normalize_arxiv("https://arxiv.org/abs/2411.17799v2"), "2411.17799")
        self.assertEqual(normalize_arxiv("arXiv:2411.17799"), "2411.17799")
        self.assertEqual(normalize_doi("https://doi.org/10.1000/ABC"), "10.1000/abc")

    def test_identifier_match_wins_over_title(self):
        paper = {"title": "A changed metadata title", "arxiv_id": "2411.17799v3"}
        self.assertTrue(paper_matches_entry(
            {"title": "Signs as Tokens", "arxiv_id": "https://arxiv.org/abs/2411.17799"}, paper
        ))

    def test_unjudged_papers_are_not_silently_labeled_irrelevant(self):
        ranked = [
            {"title": "Paper A", "doi": "10.1/a"},
            {"title": "Unjudged Paper", "doi": "10.1/x"},
            {"title": "Paper B", "doi": "10.1/b"},
        ]
        judgments = [
            {"title": "Paper A", "doi": "10.1/a", "grade": 2},
            {"title": "Paper B", "doi": "10.1/b", "grade": 0},
        ]
        metrics = graded_relevance_metrics(judgments, ranked, ks=(3,))
        self.assertEqual(metrics["relevance_judgment_count"], 2)
        self.assertEqual(metrics["judged_results_at_3"], 2)
        self.assertAlmostEqual(metrics["judgment_coverage_at_3"], 2 / 3, places=4)
        self.assertEqual(metrics["judged_precision_at_3"], 0.5)
        self.assertEqual(metrics["judged_recall_at_3"], 1.0)

    def test_empty_judgments_report_unavailable_not_zero(self):
        metrics = graded_relevance_metrics([], [{"title": "Paper A"}], ks=(5,))
        self.assertIsNone(metrics["judged_precision_at_5"])
        self.assertIsNone(metrics["judged_recall_at_5"])
        self.assertIsNone(metrics["judged_ndcg_at_5"])
        self.assertIsNone(metrics["judged_mrr"])

    def test_aspect_coverage_requires_explicit_tags(self):
        aspects = [{"id": "tokens"}, {"id": "avatars"}]
        judgments = [{"title": "Paper A", "doi": "10.1/a", "grade": 2, "aspects": ["tokens"]}]
        result = judged_aspect_coverage(aspects, judgments, [{"title": "Paper A", "doi": "10.1/a"}])
        self.assertEqual(result["aspect_coverage"], {"tokens": 1.0, "avatars": None})
        self.assertEqual(result["judged_aspect_coverage"], 1.0)

    def test_aspect_not_covered_if_relevant_paper_is_not_shortlisted(self):
        aspects = [{"id": "tokens"}]
        judgments = [{"title": "Paper A", "doi": "10.1/a", "grade": 2, "aspects": ["tokens"]}]
        result = judged_aspect_coverage(aspects, judgments, [{"title": "Other Paper"}])
        self.assertEqual(result["aspect_coverage"], {"tokens": 0.0})
        self.assertEqual(result["judged_aspect_coverage"], 0.0)


    def test_must_cite_found_by_arxiv_id(self):
        must = {"title": "Unknown Title", "arxiv_id": "2411.17799"}
        paper = {"title": "Different Metadata Title", "arxiv_id": "https://arxiv.org/abs/2411.17799v1"}
        self.assertTrue(paper_matches_entry(must, paper))

    def test_must_cite_found_by_doi(self):
        must = {"title": "Unknown Title", "doi": "10.1007/s11263-019-01281-2"}
        paper = {"title": "Different Metadata Title", "doi": "https://doi.org/10.1007/s11263-019-01281-2"}
        self.assertTrue(paper_matches_entry(must, paper))

    def test_title_differs_only_by_punctuation_or_capitalization(self):
        must = {"title": "Text2Sign: Towards Sign Language Production, Using NMT & GANs!"}
        paper = {"title": "text2sign towards sign language production using nmt gans"}
        self.assertTrue(paper_matches_entry(must, paper))

    def test_unrelated_title_not_matched(self):
        must = {"title": "Signs as Tokens: A Multilingual Sign Language Generator"}
        paper = {"title": "Quantum Error Correction with Surface Codes"}
        self.assertFalse(paper_matches_entry(must, paper))

    def test_relevant_paper_appears_only_in_pool(self):
        from copilot.graph.state import Paper
        from eval.metrics import coverage
        must_cite = [{"title": "Signs as Tokens", "arxiv_id": "2411.17799"}]
        paper_must = Paper(title="Signs as Tokens", arxiv_id="2411.17799")
        paper_other = Paper(title="Other Unrelated Work", arxiv_id="2001.00001")

        pool = [paper_must, paper_other]
        shortlist = [paper_other]

        cov_pool, missed_pool = coverage(must_cite, pool)
        cov_short, missed_short = coverage(must_cite, shortlist)

        self.assertEqual(cov_pool, 1.0)
        self.assertEqual(missed_pool, [])
        self.assertEqual(cov_short, 0.0)
        self.assertEqual(missed_short, ["Signs as Tokens"])

    def test_citation_exists_but_does_not_support_claim(self):
        from copilot.agents.critic import split_claims, support_rate
        from copilot.graph.state import ClaimCheck

        # Reference exists and is valid
        cid = "arxiv:2411.17799#abs"
        chunk_store = {
            cid: "Signs as Tokens proposes discrete motion representations for multilingual sign generation."
        }
        draft_markdown = "The architecture achieves 120 FPS on Raspberry Pi devices [cite:arxiv:2411.17799#abs]."
        claims = split_claims(draft_markdown)
        self.assertEqual(len(claims), 1)

        # Mock judge returning unsupported because chunk says nothing about Raspberry Pi or FPS
        class MockJudgeLLM:
            pass

        # Simulated claim check
        check = ClaimCheck(
            claim=claims[0][0],
            chunk_ids=claims[0][1],
            verdict="unsupported",
            reason="insufficient evidence: passage does not state 120 FPS or Raspberry Pi",
        )
        rate = support_rate([check])
        self.assertEqual(rate, 0.0)
        # Reference validity is 1.0 because the chunk exists
        ref_validity = 1.0 if cid in chunk_store else 0.0
        self.assertEqual(ref_validity, 1.0)

    def test_copilot_max_papers_env_resolution(self):
        import os

        from copilot.config import Settings
        old_val = os.environ.get("COPILOT_MAX_PAPERS")
        try:
            os.environ["COPILOT_MAX_PAPERS"] = "10"
            s = Settings.from_env()
            self.assertEqual(s.max_papers, 10)
        finally:
            if old_val is not None:
                os.environ["COPILOT_MAX_PAPERS"] = old_val
            else:
                os.environ.pop("COPILOT_MAX_PAPERS", None)


if __name__ == "__main__":
    unittest.main()

