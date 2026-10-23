import json
import tempfile
import unittest
from pathlib import Path

from eval.pool_judgments import build_annotation_pool


class PoolJudgmentTests(unittest.TestCase):
    def test_pools_candidates_across_variants_without_assigning_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = root / "artifacts"
            artifacts.mkdir()
            common = {
                "topic": {"id": "topic-a"},
                "state": {
                    "papers": {
                        "arxiv:1": {"title": "Paper One", "arxiv_id": "1", "abstract": "About tokens", "score": 0.8},
                        "doi:2": {"title": "Paper Two", "doi": "10.1/two", "score": 0.7},
                    },
                    "shortlist": ["arxiv:1"],
                },
            }
            for variant in ("baseline", "multi"):
                artifact = dict(common, variant=variant)
                (artifacts / f"topic-a__{variant}.json").write_text(json.dumps(artifact))
            dataset = {"name": "test", "topics": [{"id": "topic-a", "question": "Question?", "topic_aspects": [{"id": "tokens"}]}]}
            pool = build_annotation_pool(artifacts, dataset)
            self.assertEqual(len(pool["topics"]), 1)
            self.assertEqual(len(pool["topics"][0]["candidates"]), 2)
            paper = next(p for p in pool["topics"][0]["candidates"] if p["title"] == "Paper One")
            self.assertIsNone(paper["grade"])
            self.assertEqual(paper["pool_rank_by_variant"], {"baseline": 1, "multi": 1})
            self.assertEqual(paper["shortlist_rank_by_variant"], {"baseline": 1, "multi": 1})


if __name__ == "__main__":
    unittest.main()
