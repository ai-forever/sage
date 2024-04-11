import unittest

from sage.utils import load_available_dataset_from_hf, DatasetsAvailable
from sage.evaluation.ruspelleval import evaluation


class TestRuSpellEval(unittest.TestCase):

    def test_ruspelleval_edge_cases(self):
        ruspell_sources, ruspell_corrections = load_available_dataset_from_hf(
            DatasetsAvailable.RUSpellRU.name, for_labeler=True, split="test")
        coincide_metrics = evaluation(ruspell_sources, ruspell_corrections, ruspell_corrections)
        self.assertEqual(coincide_metrics, {"Precision": 100.0, "Recall": 100.0, "F1": 100.0})
        loose_metrics = evaluation(
            ruspell_sources, ruspell_corrections, ruspell_sources[:-1] + [ruspell_corrections[-1]])
        self.assertEqual(loose_metrics, {"Precision": 100.0, "Recall": 0.05, "F1": 0.1})

    def test_ruspelleval_general_case(self):
        source = " ".join(["фотка", "классная", "кстате", "хоть", "и", "не", "по", "теме"])
        correction = " ".join(["фотка", "классная", "кстати", "хоть", "и", "не", "по", "теме"])
        answer = " ".join(["фотка", "классная", "кстати", "хотя", "не", "по", "теме"])
        case1_metrics = evaluation([source], [correction], [answer])
        self.assertEqual(case1_metrics, {"Precision": 50.0, "Recall": 100.0, "F1": 66.67})


if __name__ == "__main__":
    unittest.main()
