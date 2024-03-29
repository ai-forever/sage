import os
import unittest
import numpy as np

from sage.spelling_corruption.sbsc.labeler import process_mistypings
from sage.spelling_corruption import CharAugConfig, WordAugConfig, SBSCConfig
from sage.utils.data_load_utils import load_available_dataset_from_hf
from sage.utils.utils import draw_and_save_errors_distributions_comparison_charts
from sage.spelling_corruption import WordAugCorruptor, CharAugCorruptor, SBSCCorruptor

SEED = 0


class CorruptorApiTests(unittest.TestCase):
    sentence = "я пошел домой"
    sentences = [sentence] * 3
    n_tests = 10
    sources, corrections = load_available_dataset_from_hf("RUSpellRU", for_labeler=True, split="train")
    ruspellru_stats, ruspellru_confusion_matrix, ruspellru_typos_cnt = process_mistypings(sources, corrections)

    def _draw_generated_distributions(self, corruptor, file_name):
        spoiled_sentences = corruptor.batch_corrupt(self.corrections, seed=SEED)
        ours_ruspellru_stats, ours_ruspellru_confusion_matrix, ours_ruspellru_typos_cnt = \
            process_mistypings(spoiled_sentences, self.corrections)
        draw_and_save_errors_distributions_comparison_charts(
            actual_typos_cnt=ours_ruspellru_typos_cnt,
            reference_typos_cnt=self.ruspellru_typos_cnt,
            actual_stats=ours_ruspellru_stats,
            reference_stats=self.ruspellru_stats,
            path_to_save=file_name
        )

    def _test_generation_correct(self, corruptor):
        res = corruptor.corrupt(self.sentence)
        self.assertEqual(type(res), str)

        res = corruptor.batch_corrupt(self.sentences)
        self.assertEqual(3, len(res))

    def test_word_augmenter(self):
        default_config = WordAugConfig()
        corruptor = WordAugCorruptor.from_default_config()
        self.assertEqual(corruptor.engine.min_aug, default_config.min_aug)
        self.assertEqual(corruptor.engine.max_aug, default_config.max_aug)
        self.assertEqual(corruptor.engine.unit_prob, default_config.unit_prob)
        self._test_generation_correct(corruptor)

        config = WordAugConfig(min_aug=2, max_aug=6, unit_prob=0.1)
        corruptor = WordAugCorruptor.from_config(config)
        self.assertEqual(corruptor.engine.min_aug, config.min_aug)
        self.assertEqual(corruptor.engine.max_aug, config.max_aug)
        self.assertEqual(corruptor.engine.unit_prob, config.unit_prob)
        self._test_generation_correct(corruptor)

    def test_char_augmenter(self):
        default_config = CharAugConfig()
        corruptor = CharAugCorruptor.from_default_config()
        self.assertEqual(corruptor.engine.min_aug, default_config.min_aug)
        self.assertEqual(corruptor.engine.max_aug, default_config.max_aug)
        self.assertEqual(corruptor.engine.unit_prob, default_config.unit_prob)
        self.assertEqual(corruptor.engine.mult_num, default_config.mult_num)
        self._test_generation_correct(corruptor)

        config = CharAugConfig(min_aug=2, max_aug=6, unit_prob=0.1, mult_num=3)
        corruptor = CharAugCorruptor.from_config(config)
        self.assertEqual(corruptor.engine.min_aug, config.min_aug)
        self.assertEqual(corruptor.engine.max_aug, config.max_aug)
        self.assertEqual(corruptor.engine.unit_prob, config.unit_prob)
        self.assertEqual(corruptor.engine.mult_num, config.mult_num)
        self._test_generation_correct(corruptor)

    def test_sbsc_corruptor(self):
        # From custom stats
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
        config = SBSCConfig(
            typos_count=typos_count,
            stats=stats,
            confusion_matrix={" ": {" ": 1}},
        )
        corruptor = SBSCCorruptor.from_config(config)
        self._test_generation_correct(corruptor)

        # From txt files
        config = SBSCConfig(
            reference_dataset_name_or_path=os.path.join(os.getcwd(), "data", "sanity_check_samples", "corruptor_tests")
        )
        corruptor = SBSCCorruptor.from_config(config)
        self._test_generation_correct(corruptor)

        # From csv file
        config = SBSCConfig(
            reference_dataset_name_or_path=os.path.join(
                os.getcwd(), "data", "sanity_check_samples", "corruptor_tests", "csv")
        )
        corruptor = SBSCCorruptor.from_config(config)
        self._test_generation_correct(corruptor)

        # From partial custom stats
        config = SBSCConfig(
            typos_count=None,
            stats=stats,
            confusion_matrix={" ": {" ": 1}},
            skip_if_position_not_found=True,
            reference_dataset_name_or_path=os.path.join(os.getcwd(), "data", "sanity_check_samples", "corruptor_tests"),
        )
        corruptor = SBSCCorruptor.from_config(config)
        self._test_generation_correct(corruptor)

        config = SBSCConfig(
            typos_count=typos_count,
            stats=None,
            confusion_matrix={" ": {" ": 1}},
            skip_if_position_not_found=True,
            reference_dataset_name_or_path=os.path.join(os.getcwd(), "data", "sanity_check_samples", "corruptor_tests"),
        )
        corruptor = SBSCCorruptor.from_config(config)
        self._test_generation_correct(corruptor)

        config = SBSCConfig(
            typos_count=typos_count,
            stats=stats,
            confusion_matrix=None,
            skip_if_position_not_found=True,
            reference_dataset_name_or_path=os.path.join(os.getcwd(), "data", "sanity_check_samples", "corruptor_tests"),
        )
        corruptor = SBSCCorruptor.from_config(config)
        self._test_generation_correct(corruptor)

        config = SBSCConfig(
            typos_count=None,
            stats=None,
            confusion_matrix={" ": {" ": 1}},
            skip_if_position_not_found=True,
            reference_dataset_name_or_path=os.path.join(os.getcwd(), "data", "sanity_check_samples", "corruptor_tests"),
        )
        corruptor = SBSCCorruptor.from_config(config)
        self._test_generation_correct(corruptor)

        config = SBSCConfig(
            typos_count=None,
            stats=stats,
            confusion_matrix=None,
            skip_if_position_not_found=True,
            reference_dataset_name_or_path=os.path.join(os.getcwd(), "data", "sanity_check_samples", "corruptor_tests")
        )
        corruptor = SBSCCorruptor.from_config(config)
        self._test_generation_correct(corruptor)

        config = SBSCConfig(
            typos_count=typos_count,
            stats=None,
            confusion_matrix=None,
            skip_if_position_not_found=True,
            reference_dataset_name_or_path=os.path.join(os.getcwd(), "data", "sanity_check_samples", "corruptor_tests")
        )
        corruptor = SBSCCorruptor.from_config(config)
        self._test_generation_correct(corruptor)

    def test_sbsc_corruptor_batch_corrupt(self):
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
        config = SBSCConfig(
            typos_count=typos_count,
            stats=stats,
            confusion_matrix={" ": {" ": 1}},
        )
        corruptor = SBSCCorruptor.from_config(config)
        self._draw_generated_distributions(corruptor, "sbsc_random_stats.jpg")

        config = SBSCConfig(
            reference_dataset_name_or_path=os.path.join(os.getcwd(), "data", "sanity_check_samples", "corruptor_tests"),
        )
        corruptor = SBSCCorruptor.from_config(config)
        self._draw_generated_distributions(corruptor, "sbsc_stats_from_txt.jpg")

        config = SBSCConfig(
            reference_dataset_name_or_path=os.path.join(
                os.getcwd(), "data", "sanity_check_samples", "corruptor_tests", "csv"),
        )
        corruptor = SBSCCorruptor.from_config(config)
        self._draw_generated_distributions(corruptor, "sbsc_stats_from_csv.jpg")

        config = SBSCConfig(
            typos_count=None,
            stats=stats,
            confusion_matrix={" ": {" ": 1}},
            skip_if_position_not_found=True,
            reference_dataset_name_or_path=os.path.join(os.getcwd(), "data", "sanity_check_samples", "corruptor_tests"),
        )
        corruptor = SBSCCorruptor.from_config(config)
        self._draw_generated_distributions(corruptor, "sbsc_custom_stats_typos_from_txt.jpg")

        config = SBSCConfig(
            reference_dataset_name_or_path="RUSpellRU",
            reference_dataset_split="train"
        )
        corruptor = SBSCCorruptor.from_config(config)
        self._draw_generated_distributions(corruptor, "sbsc_from_dataset.jpg")

        # English version
        en_config = SBSCConfig(
            lang="en",
            reference_dataset_name_or_path=os.path.join(os.getcwd(), "data", "example_data", "bea60k", "subsample")
        )
        corruptor = SBSCCorruptor.from_config(en_config)

        with open(os.path.join(os.getcwd(), "data", "example_data", "bea60k", "subsample", "sources.txt")) as src:
            sources = src.read().split("\n")
        with open(os.path.join(os.getcwd(), "data", "example_data", "bea60k", "subsample", "corrections.txt")) as corr:
            corrections = corr.read().split("\n")

        bea_stats, bea_confusion_matrix, bea_typos_cnt = process_mistypings(sources, corrections)
        spoiled_sentences = corruptor.batch_corrupt(corrections)
        ours_bea_stats, ours_bea_confusion_matrix, ours_bea_typos_cnt = process_mistypings(spoiled_sentences, corrections)
        draw_and_save_errors_distributions_comparison_charts(
            actual_typos_cnt=ours_bea_typos_cnt,
            reference_typos_cnt=bea_typos_cnt,
            actual_stats=ours_bea_stats,
            reference_stats=bea_stats,
            path_to_save="bea60k.jpg"
        )

        with self.assertRaises(RuntimeError):
            corruptor = SBSCCorruptor.from_config(
                config=SBSCConfig(
                    reference_dataset_name_or_path=None,
                )
            )

    def test_word_aug_batch_corrupt(self):
        config = WordAugConfig(min_aug=2, max_aug=6, unit_prob=0.1)
        corruptor = WordAugCorruptor.from_config(config)
        self._draw_generated_distributions(corruptor, "word_aug.jpg")

        config = CharAugConfig(min_aug=2, max_aug=6, unit_prob=0.1, mult_num=3)
        corruptor = CharAugCorruptor.from_config(config)
        self._draw_generated_distributions(corruptor, "char_aug.jpg")


if __name__ == '__main__':
    unittest.main()
