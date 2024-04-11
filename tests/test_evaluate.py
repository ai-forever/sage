import unittest

from sage.evaluation import Scorer


class TestEvaluationKit(unittest.TestCase):

    scorer = Scorer()

    sources = ["спел Кейс ее .", "спел Кейс ее ."]
    corrections = ["спелл кейс её !", "спелл кейс её !"]
    answers = ["спел кейс её .", "спелл Кейс ее !"]

    def test_scorer_errant_only(self):
        metrics = self.scorer.score(self.sources, self.corrections, self.answers, metrics=["errant"])
        expected_metrics = {
            "CASE_Precision": 100.0, "CASE_Recall": 50.0, "CASE_F1": 66.67,
            "YO_Precision": 100.0, "YO_Recall": 50.0, "YO_F1": 66.67,
            "SPELL_Precision": 100.0, "SPELL_Recall": 50.0, "SPELL_F1": 66.67,
            "PUNCT_Precision": 100.0, "PUNCT_Recall": 50.0, "PUNCT_F1": 66.67
        }
        self.assertDictEqual(metrics, expected_metrics)

    def test_scorer_ruspelleval_only(self):
        metrics = self.scorer.score(self.sources, self.corrections, self.answers, metrics=["ruspelleval"])
        self.assertDictEqual(metrics, {"Precision": 100.0, "Recall": 50.0, "F1": 66.67})

    def test_scorer_errant_ruspelleval(self):
        metrics = self.scorer.score(self.sources, self.corrections, self.answers, metrics=["errant", "ruspelleval"])
        expected_metrics = {
            "CASE_Precision": 100.0, "CASE_Recall": 50.0, "CASE_F1": 66.67,
            "YO_Precision": 100.0, "YO_Recall": 50.0, "YO_F1": 66.67,
            "SPELL_Precision": 100.0, "SPELL_Recall": 50.0, "SPELL_F1": 66.67,
            "PUNCT_Precision": 100.0, "PUNCT_Recall": 50.0, "PUNCT_F1": 66.67,
            "Precision": 100.0, "Recall": 50.0, "F1": 66.67
        }
        self.assertDictEqual(metrics, expected_metrics)

    def test_empty_errant(self):
        scorer = Scorer(False)
        self.assertRaises(
            AttributeError,
            scorer.score,
            **{"sources": self.sources, "corrections": self.corrections, "answers": self.answers, "metrics": ["errant"]}
        )


if __name__ == "__main__":
    unittest.main()
