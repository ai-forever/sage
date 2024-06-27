import os
import unittest
import numpy as np

from sage.spelling_corruption.sbsc.labeler import process_mistypings
from sage.utils.data_load_utils import load_available_dataset_from_hf
from sage.utils.utils import draw_and_save_errors_distributions_comparison_charts
from sage.spelling_corruption.sbsc.sbsc import StatisticBasedSpellingCorruption

SEED = 42


class SbscCorruptorTests(unittest.TestCase):
    def test_custom_stats(self):
        typos_count = [3]
        stats = {
            "insertion": {
                "abs": [0], "rel": [np.random.uniform(high=0.1)]
            },
            "deletion": {
                "abs": [2], "rel": [np.random.uniform(low=0.2, high=0.3)]
            },
            "substitution": {
                "abs": [1], "rel": [np.random.uniform(low=0.4, high=0.5)]
            },
            "transposition": {
                "abs": [3], "rel": [np.random.uniform(low=0.6, high=0.7)]
            },
            "extra_separator": {
                "abs": [4], "rel": [np.random.uniform(low=0.8, high=0.9)]
            },
            "missing_separator": {
                "abs": [5], "rel": [np.random.uniform(low=0.9, high=1.)]
            },
        }
        corruptor = StatisticBasedSpellingCorruption(
            lang="rus",
            typos_count=typos_count,
            stats=stats,
            confusion_matrix={" ": {" ": 1}},
            skip_if_position_not_found=True,
            random_seed=SEED
        )
        res = corruptor.batch_corrupt(["я пошел домой"] * 3)
        self.assertEqual(len(res), 3)

    def test_from_file_txt(self):
        corruptor = StatisticBasedSpellingCorruption(
            lang="rus",
            skip_if_position_not_found=True,
            reference_dataset_name_or_path=os.path.join(os.getcwd(), "data", "sanity_check_samples", "corruptor_tests"),
            random_seed=SEED
        )
        res = corruptor.batch_corrupt(["я пошел домой"] * 3)
        self.assertEqual(len(res), 3)

    def test_from_file_csv(self):
        corruptor = StatisticBasedSpellingCorruption(
            lang="rus",
            skip_if_position_not_found=True,
            reference_dataset_name_or_path=os.path.join(
                os.getcwd(), "data", "sanity_check_samples", "corruptor_tests", "csv"),
            random_seed=SEED
        )
        res = corruptor.batch_corrupt(["я пошел домой"] * 3)
        self.assertEqual(len(res), 3)

    def test_partial_custom_stats(self):
        typos_count = [3]
        stats = {
            "insertion": {
                "abs": [0], "rel": [np.random.uniform(high=0.1)]
            },
            "deletion": {
                "abs": [2], "rel": [np.random.uniform(low=0.2, high=0.3)]
            },
            "substitution": {
                "abs": [1], "rel": [np.random.uniform(low=0.4, high=0.5)]
            },
            "transposition": {
                "abs": [3], "rel": [np.random.uniform(low=0.6, high=0.7)]
            },
            "extra_separator": {
                "abs": [4], "rel": [np.random.uniform(low=0.8, high=0.9)]
            },
            "missing_separator": {
                "abs": [5], "rel": [np.random.uniform(low=0.9, high=1.)]
            },
        }
        corruptor = StatisticBasedSpellingCorruption(
            lang="rus",
            typos_count=None,
            stats=stats,
            confusion_matrix={" ": {" ": 1}},
            skip_if_position_not_found=True,
            reference_dataset_name_or_path=os.path.join(os.getcwd(), "data", "sanity_check_samples", "corruptor_tests"),
            random_seed=SEED
        )
        res = corruptor.batch_corrupt(["я пошел домой"] * 3)
        self.assertEqual(len(res), 3)

        corruptor = StatisticBasedSpellingCorruption(
            lang="rus",
            typos_count=typos_count,
            stats=None,
            confusion_matrix={" ": {" ": 1}},
            skip_if_position_not_found=True,
            reference_dataset_name_or_path=os.path.join(os.getcwd(), "data", "sanity_check_samples", "corruptor_tests"),
            random_seed=SEED
        )
        res = corruptor.batch_corrupt(["я пошел домой"] * 3)
        self.assertEqual(len(res), 3)

        corruptor = StatisticBasedSpellingCorruption(
            lang="rus",
            typos_count=typos_count,
            stats=stats,
            confusion_matrix=None,
            skip_if_position_not_found=True,
            reference_dataset_name_or_path=os.path.join(os.getcwd(), "data", "sanity_check_samples", "corruptor_tests"),
            random_seed=SEED
        )
        res = corruptor.batch_corrupt(["я пошел домой"] * 3)
        self.assertEqual(len(res), 3)

        corruptor = StatisticBasedSpellingCorruption(
            lang="rus",
            typos_count=None,
            stats=None,
            confusion_matrix={" ": {" ": 1}},
            skip_if_position_not_found=True,
            reference_dataset_name_or_path=os.path.join(os.getcwd(), "data", "sanity_check_samples", "corruptor_tests"),
            random_seed=SEED
        )
        res = corruptor.batch_corrupt(["я пошел домой"] * 3)
        self.assertEqual(len(res), 3)

        corruptor = StatisticBasedSpellingCorruption(
            lang="rus",
            typos_count=None,
            stats=stats,
            confusion_matrix=None,
            skip_if_position_not_found=True,
            reference_dataset_name_or_path=os.path.join(os.getcwd(), "data", "sanity_check_samples", "corruptor_tests"),
            random_seed=SEED
        )
        res = corruptor.batch_corrupt(["я пошел домой"] * 3)
        self.assertEqual(len(res), 3)

        corruptor = StatisticBasedSpellingCorruption(
            lang="rus",
            typos_count=typos_count,
            stats=None,
            confusion_matrix=None,
            skip_if_position_not_found=True,
            reference_dataset_name_or_path=os.path.join(os.getcwd(), "data", "sanity_check_samples", "corruptor_tests"),
            random_seed=SEED
        )
        res = corruptor.batch_corrupt(["я пошел домой"] * 3)
        self.assertEqual(len(res), 3)

    def test_wrong_config(self):
        typos_count = [3]
        stats = {
            "insertion": {
                "abs": [0], "rel": [np.random.uniform(high=0.1)]
            },
            "deletion": {
                "abs": [2], "rel": [np.random.uniform(low=0.2, high=0.3)]
            },
            "substitution": {
                "abs": [1], "rel": [np.random.uniform(low=0.4, high=0.5)]
            },
            "transposition": {
                "abs": [3], "rel": [np.random.uniform(low=0.6, high=0.7)]
            },
            "extra_separator": {
                "abs": [4], "rel": [np.random.uniform(low=0.8, high=0.9)]
            },
            "missing_separator": {
                "abs": [5], "rel": [np.random.uniform(low=0.9, high=1.)]
            },
        }

        # Empty everything
        with self.assertRaises(RuntimeError):
            corruptor = StatisticBasedSpellingCorruption(
                lang="rus",
                skip_if_position_not_found=True,
                random_seed=SEED
            )

        # Empty cases
        with self.assertRaises(RuntimeError):
            corruptor = StatisticBasedSpellingCorruption(
                lang="rus",
                typos_count=None,
                stats=stats,
                confusion_matrix={" ": {" ": 1}},
                skip_if_position_not_found=True,
                reference_dataset_name_or_path=None,
                random_seed=SEED
            )

        # Empty cases
        with self.assertRaises(RuntimeError):
            corruptor = StatisticBasedSpellingCorruption(
                lang="rus",
                typos_count=typos_count,
                stats=None,
                confusion_matrix={" ": {" ": 1}},
                skip_if_position_not_found=True,
                reference_dataset_name_or_path=None,
                random_seed=SEED
            )

        # Empty cases
        with self.assertRaises(RuntimeError):
            corruptor = StatisticBasedSpellingCorruption(
                lang="rus",
                typos_count=typos_count,
                stats=stats,
                confusion_matrix=None,
                skip_if_position_not_found=True,
                reference_dataset_name_or_path=None,
                random_seed=SEED
            )

        # Empty cases
        with self.assertRaises(RuntimeError):
            corruptor = StatisticBasedSpellingCorruption(
                lang="rus",
                typos_count=None,
                stats=None,
                confusion_matrix={" ": {" ": 1}},
                skip_if_position_not_found=True,
                reference_dataset_name_or_path=None,
                random_seed=SEED
            )

        # Empty cases
        with self.assertRaises(RuntimeError):
            corruptor = StatisticBasedSpellingCorruption(
                lang="rus",
                typos_count=None,
                stats=stats,
                confusion_matrix=None,
                skip_if_position_not_found=True,
                reference_dataset_name_or_path=None,
                random_seed=SEED
            )

        # Empty cases
        with self.assertRaises(RuntimeError):
            corruptor = StatisticBasedSpellingCorruption(
                lang="rus",
                typos_count=typos_count,
                stats=None,
                confusion_matrix=None,
                skip_if_position_not_found=True,
                reference_dataset_name_or_path=None,
                random_seed=SEED
            )

        # Wrong directory
        with self.assertRaises(ValueError):
            corruptor = StatisticBasedSpellingCorruption(
                lang="rus",
                skip_if_position_not_found=True,
                reference_dataset_name_or_path=os.path.join(os.getcwd(), "wrong_path"),
                random_seed=SEED
            )

        # Files with wrong names
        with self.assertRaises(RuntimeError):
            corruptor = StatisticBasedSpellingCorruption(
                lang="rus",
                skip_if_position_not_found=True,
                reference_dataset_name_or_path=os.path.join(
                    os.getcwd(), "data", "sanity_check_samples", "corruptor_tests", "wrong_names"),
                random_seed=SEED
            )

        # Broken txt files
        with self.assertRaises(RuntimeError):
            corruptor = StatisticBasedSpellingCorruption(
                lang="rus",
                skip_if_position_not_found=True,
                reference_dataset_name_or_path=os.path.join(
                    os.getcwd(), "data", "sanity_check_samples", "corruptor_tests", "broken_text_files"),
                random_seed=SEED
            )

        # Broken csv files
        with self.assertRaises(RuntimeError):
            corruptor = StatisticBasedSpellingCorruption(
                lang="rus",
                skip_if_position_not_found=True,
                reference_dataset_name_or_path=os.path.join(
                    os.getcwd(), "data", "sanity_check_samples", "corruptor_tests", "broken_csv_file_opening"),
                random_seed=SEED
            )

        # Broken csv files
        with self.assertRaises(RuntimeError):
            corruptor = StatisticBasedSpellingCorruption(
                lang="rus",
                skip_if_position_not_found=True,
                reference_dataset_name_or_path=os.path.join(
                    os.getcwd(), "data", "sanity_check_samples", "corruptor_tests", "broken_csv_file_columns"),
                random_seed=SEED
            )

        # Broken csv files
        with self.assertRaises(ValueError):
            corruptor = StatisticBasedSpellingCorruption(
                lang="rus",
                skip_if_position_not_found=True,
                reference_dataset_name_or_path=os.path.join(
                    os.getcwd(), "data", "sanity_check_samples", "corruptor_tests", "broken_csv_file_nans"),
                random_seed=SEED
            )

        # Wrong lang code
        with self.assertRaises(ValueError):
            corruptor = StatisticBasedSpellingCorruption(
                lang="Ru",
                skip_if_position_not_found=True,
                reference_dataset_name_or_path="RUSpellRU",
                random_seed=SEED
            )

    def test_corrupt(self):
        corruptor = StatisticBasedSpellingCorruption(
            lang="rus",
            skip_if_position_not_found=True,
            reference_dataset_name_or_path=os.path.join(os.getcwd(), "data", "sanity_check_samples", "corruptor_tests"),
            random_seed=SEED
        )
        res = corruptor.corrupt("я пошел домой")
        self.assertEqual(str, type(res))

    def test_from_hf_ruspellru(self):
        corruptor = StatisticBasedSpellingCorruption(
            lang="rus",
            reference_dataset_name_or_path="RUSpellRU",
            random_seed=SEED
        )
        res = corruptor.corrupt("я пошел домой")
        self.assertEqual(str, type(res))

        res = corruptor.batch_corrupt(["я пошел домой"] * 3)
        self.assertEqual(3, len(res))

        sources, corrections = load_available_dataset_from_hf("RUSpellRU", for_labeler=True, split="train")
        reference_stats, _, reference_typos_cnt = process_mistypings(sources, corrections)
        spoiled_sentences = corruptor.batch_corrupt(corrections)
        actual_stats, _, actual_typos_cnt = process_mistypings(spoiled_sentences, corrections)

        draw_and_save_errors_distributions_comparison_charts(
            actual_typos_cnt, reference_typos_cnt, actual_stats, reference_stats, "ruspellru.jpg")

    def test_from_hf_multidomain_gold(self):
        corruptor = StatisticBasedSpellingCorruption(
            lang="rus",
            reference_dataset_name_or_path="MultidomainGold",
            random_seed=SEED
        )
        res = corruptor.corrupt("я пошел домой")
        self.assertEqual(str, type(res))

        res = corruptor.batch_corrupt(["я пошел домой"] * 3)
        self.assertEqual(3, len(res))

        sources, corrections = load_available_dataset_from_hf("MultidomainGold", for_labeler=True, split="train")
        reference_stats, _, reference_typos_cnt = process_mistypings(sources, corrections)
        spoiled_sentences = corruptor.batch_corrupt(corrections)
        actual_stats, _, actual_typos_cnt = process_mistypings(spoiled_sentences, corrections)

        draw_and_save_errors_distributions_comparison_charts(
            actual_typos_cnt, reference_typos_cnt, actual_stats, reference_stats, "gold.jpg")

    def test_from_hf_github_typo_corpus(self):
        corruptor = StatisticBasedSpellingCorruption(
            lang="rus",
            reference_dataset_name_or_path="GitHubTypoCorpusRu",
            reference_dataset_split="test",
            random_seed=SEED
        )
        res = corruptor.corrupt("я пошел домой")
        self.assertEqual(str, type(res))

        res = corruptor.batch_corrupt(["я пошел домой"] * 3)
        self.assertEqual(3, len(res))

        sources, corrections = load_available_dataset_from_hf("GitHubTypoCorpusRu", for_labeler=True, split="test")
        reference_stats, _, reference_typos_cnt = process_mistypings(sources, corrections)
        spoiled_sentences = corruptor.batch_corrupt(corrections)
        actual_stats, _, actual_typos_cnt = process_mistypings(spoiled_sentences, corrections)

        draw_and_save_errors_distributions_comparison_charts(
            actual_typos_cnt, reference_typos_cnt, actual_stats, reference_stats, "github.jpg")

    def test_from_hf_med_spellchecker(self):
        corruptor = StatisticBasedSpellingCorruption(
            lang="rus",
            reference_dataset_name_or_path="MedSpellchecker",
            reference_dataset_split="test",
            random_seed=SEED
        )
        res = corruptor.corrupt("я пошел домой")
        self.assertEqual(str, type(res))

        res = corruptor.batch_corrupt(["я пошел домой"] * 3)
        self.assertEqual(3, len(res))

        sources, corrections = load_available_dataset_from_hf("MedSpellchecker", for_labeler=True, split="test")
        reference_stats, _, reference_typos_cnt = process_mistypings(sources, corrections)
        spoiled_sentences = corruptor.batch_corrupt(corrections)
        actual_stats, _, actual_typos_cnt = process_mistypings(spoiled_sentences, corrections)

        draw_and_save_errors_distributions_comparison_charts(
            actual_typos_cnt, reference_typos_cnt, actual_stats, reference_stats, "med.jpg")


if __name__ == '__main__':
    unittest.main()
