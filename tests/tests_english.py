import re
import math
import logging
import unittest
import string
import numpy as np

from sage.spelling_corruption.sbsc.labeler import process_mistypings
from sage.spelling_corruption.sbsc.model import Model

logger = logging.getLogger("tests")
logger.setLevel(logging.INFO)

SEED = 102
seed = np.random.default_rng(SEED)


class AugmenterTests(unittest.TestCase):

    pattern = string.punctuation.replace("-", "")

    def test_labeler_empty(self):
        logger.info("\nTesting labeler empty string...\n")
        expected_global_stats = {
            "insertion": {
                "abs": [],
                "rel": []
            },
            "deletion": {
                "abs": [],
                "rel": []
            },
            "substitution": {
                "abs": [],
                "rel": []
            },
            "transposition": {
                "abs": [],
                "rel": []
            },
            "missing_separator": {
                "abs": [],
                "rel": []
            },
            "extra_separator": {
                "abs": [],
                "rel": []
            },
        }
        expected_global_cm = {}
        expected_mistyping_cnt = []
        actual_global_stats, actual_global_cm, actual_mistypings_cnt = process_mistypings([], [])
        self.assertEqual(expected_global_stats, actual_global_stats)
        self.assertEqual(expected_global_cm, actual_global_cm)
        self.assertEqual(expected_mistyping_cnt, actual_mistypings_cnt)

    def test_labeler_insertion_in_beginning(self):
        logger.info("\nTesting labeler insertion in beginning...\n")
        src1 = "I" * 100 + "In our Acadamy we are not allowed to smoke ."
        corr1 = "In our Acadamy we are not allowed to smoke ."
        l = len(re.sub(r"[{}]".format(self.pattern), "", src1.lower()))

        expected_global_stats = {
            "insertion": {
                "abs": list(range(100)),
                "rel": []
            },
            "deletion": {
                "abs": [],
                "rel": []
            },
            "substitution": {
                "abs": [],
                "rel": []
            },
            "transposition": {
                "abs": [],
                "rel": []
            },
            "missing_separator": {
                "abs": [],
                "rel": []
            },
            "extra_separator": {
                "abs": [],
                "rel": []
            },
        }
        expected_global_cm = {}
        expected_mistyping_cnt = [100]

        for k, v in expected_global_stats.items():
            expected_global_stats[k]["abs"] = sorted(v["abs"])
            expected_global_stats[k]["rel"] = sorted([elem / l for elem in v["abs"]])
        expected_mistyping_cnt = sorted(expected_mistyping_cnt)

        actual_global_stats, actual_global_cm, actual_mistypings_cnt = process_mistypings([src1], [corr1])
        for k, v in actual_global_stats.items():
            actual_global_stats[k]["abs"] = sorted(v["abs"])
            actual_global_stats[k]["rel"] = sorted(v["rel"])
        actual_mistypings_cnt = sorted(actual_mistypings_cnt)

        self.assertEqual(expected_global_stats, actual_global_stats,
                         "expected stats: {} \nactual stats: {}".format(expected_global_stats, actual_global_stats))
        self.assertEqual(expected_mistyping_cnt, actual_mistypings_cnt,
                         "expected counts: {} \nactual counts: {}".format(expected_mistyping_cnt,
                                                                          actual_mistypings_cnt))
        self.assertEqual(expected_global_cm, actual_global_cm,
                         "expected cm: {} \nactual cm: {}".format(expected_global_cm, actual_global_cm))

    def test_labeler_deletion_in_beginning(self):
        logger.info("\nTesting labeler deletion in beginning...\n")
        src1 = "In our Acadamy we are not allowed to smoke ."
        corr1 = "I" * 100 + "In our Acadamy we are not allowed to smoke ."
        l = len(re.sub(r"[{}]".format(self.pattern), "", src1.lower()))

        expected_global_stats = {
            "insertion": {
                "abs": [],
                "rel": []
            },
            "deletion": {
                "abs": [0] * 100,
                "rel": []
            },
            "substitution": {
                "abs": [],
                "rel": []
            },
            "transposition": {
                "abs": [],
                "rel": []
            },
            "missing_separator": {
                "abs": [],
                "rel": []
            },
            "extra_separator": {
                "abs": [],
                "rel": []
            },
        }
        expected_global_cm = {}
        expected_mistyping_cnt = [100]

        for k, v in expected_global_stats.items():
            expected_global_stats[k]["abs"] = sorted(v["abs"])
            expected_global_stats[k]["rel"] = sorted([elem / l for elem in v["abs"]])
        expected_mistyping_cnt = sorted(expected_mistyping_cnt)

        actual_global_stats, actual_global_cm, actual_mistypings_cnt = process_mistypings([src1], [corr1])
        for k, v in actual_global_stats.items():
            actual_global_stats[k]["abs"] = sorted(v["abs"])
            actual_global_stats[k]["rel"] = sorted(v["rel"])
        actual_mistypings_cnt = sorted(actual_mistypings_cnt)

        self.assertEqual(expected_global_stats, actual_global_stats,
                         "expected stats: {} \nactual stats: {}".format(expected_global_stats, actual_global_stats))
        self.assertEqual(expected_mistyping_cnt, actual_mistypings_cnt,
                         "expected counts: {} \nactual counts: {}".format(expected_mistyping_cnt,
                                                                          actual_mistypings_cnt))
        self.assertEqual(expected_global_cm, actual_global_cm,
                         "expected cm: {} \nactual cm: {}".format(expected_global_cm, actual_global_cm))

    def test_labeler_single(self):
        logger.info("\nTesting labeler single example...\n")
        src1 = " ".join(["Th e queuses are not the only prolem . If you go by car there is",
                 "the problem of parkengf , fi you goby bus th ere are also queues",
                 "ase you have to carry a lot ofs carrer bag carrier bsg s during your",
                 "journy and yo u wo n't always be lucky sxnough to fi nd a seat ."])
        corr1 = " ".join(["The queues are not the only problem . If you go by car there is",
                 "the problem of parking , if you go by bus there are also queues",
                 "and you have to carry a lot of carrier bag carrier bags during your",
                 "journey and you wo n't always be lucky enough to find a seat ."])
        l = len(re.sub(r"[{}]".format(self.pattern), "", src1.lower()))

        expected_global_stats = {
            "insertion": {
                "abs": [9, 86, 158, 235],
                "rel": []
            },
            "deletion": {
                "abs": [33, 164, 202],
                "rel": []
            },
            "substitution": {
                "abs": [83, 129, 130, 180, 236],
                "rel": []
            },
            "transposition": {
                "abs": [89],
                "rel": []
            },
            "missing_separator": {
                "abs": [98],
                "rel": []
            },
            "extra_separator": {
                "abs": [2, 107, 182, 210, 248],
                "rel": []
            },
        }
        expected_global_cm = {
            "i": {"e": 1},
            "n": {"s": 1},
            "d": {"e": 1},
            "a": {"s": 1},
            "e": {"x": 1},
        }
        expected_mistyping_cnt = [19]

        for k, v in expected_global_stats.items():
            expected_global_stats[k]["abs"] = sorted(v["abs"])
            expected_global_stats[k]["rel"] = sorted([elem / l for elem in v["abs"]])
        expected_mistyping_cnt = sorted(expected_mistyping_cnt)

        actual_global_stats, actual_global_cm, actual_mistypings_cnt = process_mistypings([src1], [corr1])
        for k, v in actual_global_stats.items():
            actual_global_stats[k]["abs"] = sorted(v["abs"])
            actual_global_stats[k]["rel"] = sorted(v["rel"])
        actual_mistypings_cnt = sorted(actual_mistypings_cnt)

        self.assertEqual(expected_global_stats, actual_global_stats,
                         "expected stats: {} \nactual stats: {}".format(expected_global_stats, actual_global_stats))
        self.assertEqual(expected_mistyping_cnt, actual_mistypings_cnt,
                         "expected counts: {} \nactual counts: {}".format(expected_mistyping_cnt, actual_mistypings_cnt))
        self.assertEqual(expected_global_cm, actual_global_cm,
                         "expected cm: {} \nactual cm: {}".format(expected_global_cm, actual_global_cm))

    def test_labeler_batch(self):
        logger.info("\nTesting labeler batch...\n")
        src1 = " ".join(["Th e queuses are not the only prolem . If you go by car there is",
                         "the problem of parkengf , fi you goby bus th ere are also queues",
                         "ase you have to carry a lot ofs carrer bag carrier bsg s during your",
                         "journy and yo u wo n't always be lucky sxnough to fi nd a seat ."])
        corr1 = " ".join(["The queues are not the only problem . If you go by car there is",
                          "the problem of parking , if you go by bus there are also queues",
                          "and you have to carry a lot of carrier bag carrier bags during your",
                          "journey and you wo n't always be lucky enough to find a seat ."])
        l1 = len(re.sub(r"[{}]".format(self.pattern), "", src1.lower()))

        src2 = " ".join(["I readlly wa nted to gs to the Engeish school but I have stopped",
                         "cecause te lessons wes re too expene sives ."])
        corr2 = " ".join(["I really wanted to go to the English school but I have stopped",
                         "because the lessons were too expensive ."])
        l2 = len(re.sub(r"[{}]".format(self.pattern), "", src2.lower()))

        expected_global_stats = {
            "insertion": {
                "abs": [9, 86, 158, 235],
                "rel": []
            },
            "deletion": {
                "abs": [33, 164, 202],
                "rel": []
            },
            "substitution": {
                "abs": [83, 129, 130, 180, 236],
                "rel": []
            },
            "transposition": {
                "abs": [89],
                "rel": []
            },
            "missing_separator": {
                "abs": [98],
                "rel": []
            },
            "extra_separator": {
                "abs": [2, 107, 182, 210, 248],
                "rel": []
            },
        }
        expected_global_stats_2 = {
            "insertion": {
                "abs": [5, 86, 100, 106],
                "rel": []
            },
            "deletion": {
                "abs": [74],
                "rel": []
            },
            "substitution": {
                "abs": [22, 34, 65],
                "rel": []
            },
            "transposition": {
                "abs": [],
                "rel": []
            },
            "missing_separator": {
                "abs": [],
                "rel": []
            },
            "extra_separator": {
                "abs": [12, 87, 101],
                "rel": []
            },
        }
        expected_global_cm = {
            "i": {"e": 1},
            "n": {"s": 1},
            "d": {"e": 1},
            "a": {"s": 1},
            "e": {"x": 1},
            "o": {"s": 1},
            "l": {"e": 1},
            "b": {"c": 1},
        }
        expected_mistyping_cnt = [19, 11]

        for k in expected_global_stats.keys():
            expected_global_stats[k]["rel"] = [elem / l1 for elem in expected_global_stats[k]["abs"]]
            expected_global_stats[k]["abs"].extend(expected_global_stats_2[k]["abs"])
            expected_global_stats[k]["rel"].extend([elem / l2 for elem in expected_global_stats_2[k]["abs"]])

            expected_global_stats[k]["abs"] = sorted(expected_global_stats[k]["abs"])
            expected_global_stats[k]["rel"] = sorted(expected_global_stats[k]["rel"])

        expected_mistyping_cnt = sorted(expected_mistyping_cnt)

        actual_global_stats, actual_global_cm, actual_mistypings_cnt = process_mistypings([src1, src2], [corr1, corr2])
        for k, v in actual_global_stats.items():
            actual_global_stats[k]["abs"] = sorted(v["abs"])
            actual_global_stats[k]["rel"] = sorted(v["rel"])
        actual_mistypings_cnt = sorted(actual_mistypings_cnt)

        self.assertEqual(expected_global_stats, actual_global_stats,
                         "expected stats: {} \nactual stats: {}".format(expected_global_stats, actual_global_stats))
        self.assertEqual(expected_mistyping_cnt, actual_mistypings_cnt,
                         "expected counts: {} \nactual counts: {}".format(expected_mistyping_cnt,
                                                                          actual_mistypings_cnt))
        self.assertEqual(expected_global_cm, actual_global_cm,
                         "expected cm: {} \nactual cm: {}".format(expected_global_cm, actual_global_cm))

    def test_mistypings(self):
        test_functions = []

        def register_function(foo):
            test_functions.append(foo)

        @register_function
        def test_insertion(skip_mode):
            logger.info("\nTesting insertion with skip mode {}...\n".format(skip_mode))
            k = 10
            sentence = "." * k
            typos_count = [2 * k]
            stats = {"insertion": {"abs": list(range(10)), "rel": [0.05 + 0.1 * i for i in range(10)]}}
            params = {
                "typos_count": typos_count,
                "stats": stats,
                "confusion_matrix": {" ": {" ": 1}},
                "skip_if_position_not_found": skip_mode,
                "debug_mode": True,
                "lang": "en",
            }
            m = Model(**params)
            m.transform(sentence, seed)
            for pos, pre_positions, post_positions in \
                    zip(m.stats["pos"], m.stats["used_positions_pre"], m.stats["used_positions_after"]):
                self.assertFalse(pos in pre_positions, "{} in {}".format(pos, pre_positions))
                expected_post_positions = [elem if elem < pos else elem + 1 for elem in pre_positions]
                expected_post_positions.append(pos)
                self.assertEqual(
                    sorted(expected_post_positions),
                    sorted(post_positions), "{} vs {}".format(expected_post_positions, post_positions)
                )

        @register_function
        def test_deletion(skip_mode):
            logger.info("\nTesting deletion with skip mode {}...\n".format(skip_mode))
            sentence1 = " D ".join(self.pattern)
            sentence2 = " . ".join(self.pattern)
            typos_count = [2 * len(sentence1)]
            stats = {"deletion": {"abs": list(range(10)), "rel": [0.05 + 0.1 * i for i in range(10)]}}
            params = {
                "typos_count": typos_count,
                "stats": stats,
                "confusion_matrix": {" ": {" ": 1}},
                "skip_if_position_not_found": skip_mode,
                "debug_mode": True,
                "lang": "en",
            }
            m = Model(**params)
            actual_transform1 = m.transform(sentence1, seed)
            actual_transform2 = m.transform(sentence2, seed)

            expected_sentence1 = sentence1.replace("D", "")
            self.assertEqual(
                actual_transform1, expected_sentence1, "{} vs {}".format(actual_transform1, expected_sentence1))
            self.assertEqual(actual_transform2, sentence2, "{} vs {}".format(actual_transform2, sentence2))

            for pos, pre_positions, post_positions in \
                    zip(m.stats["pos"], m.stats["used_positions_pre"], m.stats["used_positions_after"]):
                self.assertFalse(pos in pre_positions, "{} in {}".format(pos, pre_positions))
                expected_post_positions = [elem if elem < pos else elem - 1 for elem in pre_positions]
                expected_post_positions.append(pos)
                self.assertEqual(
                    sorted(expected_post_positions),
                    sorted(post_positions), "{} vs {}".format(expected_post_positions, post_positions)
                )

        @register_function
        def test_substitution(skip_mode):
            logger.info("\nTesting substitution with skip mode {}...\n".format(skip_mode))
            conditions = string.punctuation + string.digits + "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
            sentence1 = " ".join(conditions)
            sentence2 = " L ".join(conditions)
            sentence3 = " l ".join(conditions)
            typos_count = [2 * len(sentence1)]
            stats = {"substitution": {"abs": list(range(10)), "rel": [0.05 + 0.1 * i for i in range(10)]}}
            params = {
                "typos_count": typos_count,
                "stats": stats,
                "confusion_matrix": {"l": {"x": 1}},
                "skip_if_position_not_found": skip_mode,
                "debug_mode": True,
                "lang": "en",
            }
            m = Model(**params)
            actual_transform1 = m.transform(sentence1, seed)
            actual_transform2 = m.transform(sentence2, seed)
            actual_transform3 = m.transform(sentence3, seed)

            expected_transform2 = sentence2.replace("L", "X")
            expected_transform3 = sentence3.replace("l", "x")
            self.assertEqual(actual_transform1, sentence1, "{} vs {}".format(actual_transform1, sentence1))
            self.assertEqual(
                actual_transform2, expected_transform2, "{} vs {}".format(actual_transform2, expected_transform2))
            self.assertEqual(
                actual_transform3, expected_transform3, "{} vs {}".format(actual_transform3, expected_transform3))

            for pos, pre_positions, post_positions in \
                    zip(m.stats["pos"], m.stats["used_positions_pre"], m.stats["used_positions_after"]):
                self.assertFalse(pos in pre_positions, "{} in {}".format(pos, pre_positions))
                expected_post_positions = pre_positions + [pos]
                self.assertEqual(
                    sorted(expected_post_positions),
                    sorted(post_positions), "{} vs {}".format(expected_post_positions, post_positions)
                )

        @register_function
        def test_transposition(skip_mode):
            logger.info("\nTesting transposition with skip mode {}...\n".format(skip_mode))
            sentence1 = "d".join(string.punctuation)
            sentence2 = "dd".join(string.punctuation)
            sentence3 = " abb ".join(string.punctuation)
            d = {elem: pos for pos, elem in enumerate(sentence3) if elem in string.punctuation}

            typos_count = [2 * len(sentence3)]
            stats = {"transposition": {"abs": list(range(10)), "rel": [0.05 + 0.1 * i for i in range(10)]}}
            params = {
                "typos_count": typos_count,
                "stats": stats,
                "confusion_matrix": {" ": {" ": 1}},
                "skip_if_position_not_found": skip_mode,
                "debug_mode": True,
                "lang": "en",
            }
            m = Model(**params)
            actual_transform1 = m.transform(sentence1, seed)
            actual_transform2 = m.transform(sentence2, seed)
            actual_transform3 = m.transform(sentence3, seed)
            d_transform = {elem: pos for pos, elem in enumerate(actual_transform3) if elem in string.punctuation}

            self.assertEqual(actual_transform1, sentence1, "{} vs {}".format(actual_transform1, sentence1))
            self.assertEqual(actual_transform2, sentence2, "{} vs {}".format(actual_transform2, sentence2))
            self.assertDictEqual(d, d_transform, "dicts are not the same")

            count = 0
            for i in range(len(actual_transform3) - 1):
                if actual_transform3[i] == actual_transform3[i + 1]:
                    count += 1
                    break
            self.assertEqual(count, 0, "count = {}".format(count))

            for pos, pre_positions, post_positions in \
                    zip(m.stats["pos"], m.stats["used_positions_pre"], m.stats["used_positions_after"]):
                self.assertFalse(pos in pre_positions, "{} in {}".format(pos, pre_positions))
                self.assertFalse(pos + 1 in pre_positions, "{} in {}".format(pos + 1, pre_positions))
                expected_post_positions = pre_positions + [pos, pos + 1]
                self.assertEqual(
                    sorted(expected_post_positions),
                    sorted(post_positions), "{} vs {}".format(expected_post_positions, post_positions)
                )

        @register_function
        def test_extra_separator(skip_mode):
            logger.info("\nTesting extra separator with skip mode {}...\n".format(skip_mode))
            sentence1 = " d ".join(string.punctuation)
            sentence2 = " dd ".join(string.punctuation)
            typos_count = [2 * len(sentence2)]
            stats = {"extra_separator": {"abs": list(range(10)), "rel": [0.05 + 0.1 * i for i in range(10)]}}
            params = {
                "typos_count": typos_count,
                "stats": stats,
                "confusion_matrix": {" ": {" ": 1}},
                "skip_if_position_not_found": skip_mode,
                "debug_mode": True,
                "lang": "en",
            }
            m = Model(**params)
            actual_transform1 = m.transform(sentence1, seed)
            actual_transform2 = m.transform(sentence2, seed)

            expected_transform2 = sentence2.replace("dd", "d d")
            self.assertEqual(actual_transform1, sentence1, "{} vs {}".format(actual_transform1, sentence1))
            self.assertEqual(
                actual_transform2, expected_transform2, "{} vs {}".format(actual_transform2, expected_transform2))

            for pos, pre_positions, post_positions in \
                    zip(m.stats["pos"], m.stats["used_positions_pre"], m.stats["used_positions_after"]):
                self.assertFalse(pos in pre_positions, "{} in {}".format(pos, pre_positions))
                self.assertNotEqual(pos, 0)
                expected_post_positions = [elem if elem < pos else elem + 1 for elem in pre_positions]
                expected_post_positions.append(pos)
                self.assertEqual(
                    sorted(expected_post_positions),
                    sorted(post_positions), "{} vs {}".format(expected_post_positions, post_positions)
                )

        @register_function
        def test_missing_separator(skip_mode):
            logger.info("\nTesting missing separator with skip mode {}...\n".format(skip_mode))
            sentence1 = "d" * 20
            sentence2 = " ".join(list(sentence1))

            typos_count = [2 * len(sentence2)]
            stats = {"missing_separator": {"abs": list(range(10)), "rel": [0.05 + 0.1 * i for i in range(10)]}}
            params = {
                "typos_count": typos_count,
                "stats": stats,
                "confusion_matrix": {" ": {" ": 1}},
                "skip_if_position_not_found": skip_mode,
                "debug_mode": True,
                "lang": "en",
            }
            m = Model(**params)
            actual_transform1 = m.transform(sentence1, seed)
            actual_transform2 = m.transform(sentence2, seed)

            self.assertEqual(actual_transform1, sentence1, "{} vs {}".format(actual_transform1, sentence1))
            self.assertEqual(actual_transform2, sentence1, "{} vs {}".format(actual_transform2, sentence1))
            for pos, pre_positions, post_positions in \
                    zip(m.stats["pos"], m.stats["used_positions_pre"], m.stats["used_positions_after"]):
                self.assertFalse(pos in pre_positions, "{} in {}".format(pos, pre_positions))
                expected_post_positions = [elem if elem < pos else elem - 1 for elem in pre_positions]
                expected_post_positions.append(pos)
                self.assertEqual(
                    sorted(expected_post_positions),
                    sorted(post_positions), "{} vs {}".format(expected_post_positions, post_positions)
                )

        for test_function in test_functions:
            test_function(skip_mode=True)
            test_function(skip_mode=False)

    def test_transform_empty_string(self):
        logger.info("\nTesting transform empty string...\n")
        sentence = ""
        typos_count = [6]
        stats = {
            "insertion": {
                "abs": [0], "rel": [np.random.uniform(high=0.1)]
            },
            "deletion": {
                "abs": [2], "rel": [np.random.uniform(high=0.1)]
            },
            "substitution": {
                "abs": [1], "rel": [np.random.uniform(high=0.1)]
            },
            "transposition": {
                "abs": [3], "rel": [np.random.uniform(high=0.1)]
            },
            "extra_separator": {
                "abs": [4], "rel": [np.random.uniform(high=0.1)]
            },
            "missing_separator": {
                "abs": [5], "rel": [np.random.uniform(high=0.1)]
            },
        }
        params = {
            "typos_count": typos_count,
            "stats": stats,
            "confusion_matrix": {" ": {" ": 1}},
            "skip_if_position_not_found": True,
            "debug_mode": True,
            "lang": "en",
        }
        m = Model(**params)
        result = m.transform(sentence, seed)
        self.assertEqual(result, "")

    def test_transform_single(self):
        logger.info("\nTesting transform single bucket...\n")
        sentence = "We can either rain check or we can make plans."
        typos_count = [1]
        stats = {
            "insertion": {
                "abs": [0], "rel": [np.random.uniform(low=0.1)]
            },
            "deletion": {
                "abs": [2], "rel": [np.random.uniform(low=0.1, high=0.2)]
            },
            "substitution": {
                "abs": [1], "rel": [np.random.uniform(low=0.2, high=0.3)]
            },
            "transposition": {
                "abs": [3], "rel": [np.random.uniform(low=0.3, high=0.4)]
            },
            "extra_separator": {
                "abs": [4], "rel": [np.random.uniform(low=0.5, high=0.6)]
            },
            "missing_separator": {
                "abs": [5], "rel": [np.random.uniform(low=0.6, high=0.7)]
            },
        }
        params = {
            "typos_count": typos_count,
            "stats": stats,
            "confusion_matrix": {" ": {" ": 1}},
            "skip_if_position_not_found": True,
            "debug_mode": True,
            "lang": "en",
        }
        m = Model(**params)

        bins = [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
        error2interval = {}
        for k, v in stats.items():
            error2interval[k] = np.digitize(v["rel"][0], bins)

        for _ in range(len(stats) * 3):
            result = m.transform(sentence, seed)
            global_stats, global_cm, mistypings_cnt = process_mistypings([result], [sentence])

            for k, v in global_stats.items():
                for pos in v["rel"]:
                    actual_interval = np.digitize(pos, bins)
                    expected_interval = error2interval[k]
                    self.assertLessEqual(actual_interval, expected_interval + 1)
                    self.assertGreaterEqual(actual_interval, expected_interval - 1)

    def test_transform_short_string(self):
        logger.info("\nTesting transform short string...\n")
        typos_count = [3]
        stats = {
            "insertion": {
                "abs": [0], "rel": [np.random.uniform(low=0.1)]
            },
            "deletion": {
                "abs": [2], "rel": [np.random.uniform(low=0.1, high=0.2)]
            },
            "substitution": {
                "abs": [1], "rel": [np.random.uniform(low=0.2, high=0.3)]
            },
            "transposition": {
                "abs": [3], "rel": [np.random.uniform(low=0.3, high=0.4)]
            },
            "extra_separator": {
                "abs": [4], "rel": [np.random.uniform(low=0.5, high=0.6)]
            },
            "missing_separator": {
                "abs": [5], "rel": [np.random.uniform(low=0.6, high=0.7)]
            },
        }
        params = {
            "typos_count": typos_count,
            "stats": stats,
            "confusion_matrix": {" ": {" ": 1}},
            "skip_if_position_not_found": True,
            "debug_mode": True,
            "lang": "en",
        }
        m = Model(**params)
        bins = [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
        error2interval = {}
        for k, v in stats.items():
            error2interval[k] = np.digitize(v["rel"][0], bins)
        for i in range(15):
            sentence = "f" * i
            result = m.transform(sentence, seed)
            global_stats, global_cm, mistypings_cnt = process_mistypings([result], [sentence])
            if sum(mistypings_cnt) == 0:
                self.assertEqual(sentence, result)
            else:
                self.assertNotEqual(sentence, result)

    def test_factorization_scheme(self):
        logger.info("\nTesting factorization scheme...\n")
        typos_count = [1]
        stats = {
            "missing_separator": {
                "abs": [1], "rel": [0.1]
            },
        }
        params = {
            "typos_count": typos_count,
            "stats": stats,
            "confusion_matrix": {" ": {" ": 1}},
            "skip_if_position_not_found": True,
            "debug_mode": True,
            "lang": "en",
        }
        m = Model(**params)
        for sample_length in range(10, 1000):
            res = []
            for interval_idx in range(1, 11):
                left, right = m._bins[interval_idx - 1], m._bins[interval_idx]
                most_left = math.ceil(sample_length * left)
                most_right = math.ceil(sample_length * right)
                for elem in range(most_left, most_right):
                    self.assertGreaterEqual(elem / sample_length, left)
                    self.assertLessEqual(elem / sample_length, right)
                    res.append(elem)
            self.assertEqual(res, list(range(sample_length)))

    def test_wrong_stats_format(self):
        logger.info("\nTesting empty stats...\n")

        # Wrong lang code
        typos_count = [1]
        stats = {
            "insertion": {
                "abs": [0], "rel": [np.random.uniform(low=0.1)]
            },
            "deletion": {
                "abs": [2], "rel": [np.random.uniform(low=0.1, high=0.2)]
            },
            "substitution": {
                "abs": [1], "rel": [np.random.uniform(low=0.2, high=0.3)]
            },
            "transposition": {
                "abs": [3], "rel": [np.random.uniform(low=0.3, high=0.4)]
            },
            "extra_separator": {
                "abs": [4], "rel": [np.random.uniform(low=0.5, high=0.6)]
            },
            "missing_separator": {
                "abs": [5], "rel": [np.random.uniform(low=0.6, high=0.7)]
            },
        }
        params = {
            "typos_count": typos_count,
            "stats": stats,
            "confusion_matrix": {" ": {" ": 1}},
            "skip_if_position_not_found": True,
            "debug_mode": True,
            "lang": "Brit"
        }
        self.assertRaises(ValueError, Model, **params)

        # All empty
        params = {
            "typos_count": [],
            "stats": {},
            "confusion_matrix": {},
            "skip_if_position_not_found": True,
            "debug_mode": True,
            "lang": "en",
        }
        self.assertRaises(ValueError, Model, **params)

        # Stats and cm empty
        params = {
            "typos_count": [1],
            "stats": {},
            "confusion_matrix": {},
            "skip_if_position_not_found": True,
            "debug_mode": True,
            "lang": "en",
        }
        self.assertRaises(ValueError, Model, **params)

        # CM is empty
        params = {
            "typos_count": [1],
            "stats": {
                "substitution": {
                    "abs": [1], "rel": [0.1]
                },
            },
            "confusion_matrix": {},
            "skip_if_position_not_found": True,
            "debug_mode": True,
            "lang": "en",
        }
        self.assertRaises(ValueError, Model, **params)

        # Stats are actually empty
        stats = {
            "insertion": {
                "abs": [], "rel": []
            },
            "deletion": {
                "abs": [], "rel": []
            },
            "substitution": {
                "abs": [], "rel": []
            },
            "transposition": {
                "abs": [], "rel": []
            },
            "extra_separator": {
                "abs": [], "rel": []
            },
            "missing_separator": {
                "abs": [], "rel": []
            },
        }
        params = {
            "typos_count": [1],
            "stats": stats,
            "confusion_matrix": {" ": {" ": 1}},
            "skip_if_position_not_found": True,
            "debug_mode": True,
            "lang": "en",
        }
        self.assertRaises(ValueError, Model, **params)

        # Typos counts are empty
        params = {
            "typos_count": [],
            "stats": {
                "substitution": {
                    "abs": [1], "rel": [0.1]
                },
            },
            "confusion_matrix": {" ": {" ": 1}},
            "skip_if_position_not_found": True,
            "debug_mode": True,
            "lang": "en",
        }
        self.assertRaises(ValueError, Model, **params)

        # CM is not empty
        params = {
            "typos_count": [],
            "stats": {},
            "confusion_matrix": {" ": {" ": 1}},
            "skip_if_position_not_found": True,
            "debug_mode": True,
            "lang": "en",
        }
        self.assertRaises(ValueError, Model, **params)

        # Stats are not empty
        params = {
            "typos_count": [],
            "stats": {
                "deletion": {
                    "abs": [1], "rel": [0.1]
                },
            },
            "confusion_matrix": {},
            "skip_if_position_not_found": True,
            "debug_mode": True,
            "lang": "en",
        }
        self.assertRaises(ValueError, Model, **params)

        # CM is legally empty
        params = {
            "typos_count": [1],
            "stats": {
                "deletion": {
                    "abs": [1], "rel": [0.1]
                },
            },
            "confusion_matrix": {},
            "skip_if_position_not_found": True,
            "debug_mode": True,
            "lang": "en",
        }
        model = Model(**params)

        # Stats are wrong
        params = {
            "typos_count": [1],
            "stats": {
                "deletion": {
                    "abs": [-1, 2, 3], "rel": [0.1, 0.9, 0.5]
                },
                "insertion": {
                    "abs": [1, 2, 3], "rel": [-0.1, 0.9, 0.5]
                },
            },
            "confusion_matrix": {},
            "skip_if_position_not_found": True,
            "debug_mode": True,
            "lang": "en",
        }
        self.assertRaises(ValueError, Model, **params)

        # Counts are wrong
        params = {
            "typos_count": [-1, 2, 3],
            "stats": {
                "missing_separator": {
                    "abs": [1, 2, 3], "rel": [0.1, 0.9, 0.5]
                },
            },
            "confusion_matrix": {},
            "skip_if_position_not_found": True,
            "debug_mode": True,
            "lang": "en",
        }
        self.assertRaises(ValueError, Model, **params)

        # Stats are wrong (key)
        stats = {
            "insertion": {
                "abs": [], "rel": []
            },
            "deletion": {
                "abs": [], "rel": []
            },
            "sbstitution": {
                "abs": [], "rel": []
            },
            "transposition": {
                "abs": [], "rel": []
            },
            "extra_separator": {
                "abs": [], "rel": []
            },
            "missing_separator": {
                "abs": [], "rel": []
            },
        }
        params = {
            "typos_count": [1, 2, 3],
            "stats": stats,
            "confusion_matrix": {" ": {" ": 1}},
            "skip_if_position_not_found": True,
            "debug_mode": True,
            "lang": "en",
        }
        self.assertRaises(ValueError, Model, **params)

        # CM is wrong 1
        params = {
            "typos_count": [1, 2, 3],
            "stats": {
                "missing_separator": {
                    "abs": [1, 2, 3], "rel": [0.1, 0.9, 0.5]
                },
            },
            "confusion_matrix": {"adff": {"s": 1}},
            "skip_if_position_not_found": True,
            "debug_mode": True,
            "lang": "en",
        }
        self.assertRaises(ValueError, Model, **params)

        # CM is wrong 2
        params = {
            "typos_count": [1, 2, 3],
            "stats": {
                "missing_separator": {
                    "abs": [1, 2, 3], "rel": [0.1, 0.9, 0.5]
                },
            },
            "confusion_matrix": {"f": {"asdf": 1}},
            "skip_if_position_not_found": True,
            "debug_mode": True,
            "lang": "en",
        }
        self.assertRaises(ValueError, Model, **params)

        # CM is wrong 3
        params = {
            "typos_count": [1, 2, 3],
            "stats": {
                "missing_separator": {
                    "abs": [1, 2, 3], "rel": [0.1, 0.9, 0.5]
                },
            },
            "confusion_matrix": {"a": {"b": -1}},
            "skip_if_position_not_found": True,
            "debug_mode": True,
            "lang": "en",
        }
        self.assertRaises(ValueError, Model, **params)

    def test_transform_several_strings(self):
        logger.info("\nTesting transform several strings...\n")
        sentence1 = ""
        sentence2 = "adfdf"
        sentence3 = "We can either rain check or we can make plans."
        sentence4 = " ".join(["Th e queuses are not the only prolem . If you go by car there is",
                             "the problem of parkengf , fi you goby bus th ere are also queues",
                             "ase you have to carry a lot ofs carrer bag carrier bsg s during your",
                             "journy and yo u wo n't always be lucky sxnough to fi nd a seat ."])
        sentences = [sentence1, sentence2, sentence3, sentence4]
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
        params = {
            "typos_count": typos_count,
            "stats": stats,
            "confusion_matrix": {" ": {" ": 1}},
            "skip_if_position_not_found": True,
            "debug_mode": True,
            "lang": "en",
        }
        m = Model(**params)
        results = []
        for sentence in sentences:
            results.append(m.transform(sentence, seed))

        self.assertEqual(results[0], "")
        global_stats, global_cm, mistypings_cnt = process_mistypings(results[1:], sentences[1:])

        bins = [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
        error2interval = {}
        for k, v in stats.items():
            error2interval[k] = np.digitize(v["rel"][0], bins)

        for k, v in global_stats.items():
            for pos in v["rel"]:
                actual_interval = np.digitize(pos, bins)
                expected_interval = error2interval[k]
                self.assertLessEqual(actual_interval, expected_interval + 1)
                self.assertGreaterEqual(actual_interval, expected_interval - 1)


if __name__ == '__main__':
    unittest.main()
