"""
This module provides the main functionality to make statistical mistypings 
that is embodied in Model class. 

"""

import math
from functools import reduce
from typing import List, Dict, Optional, Union

import numpy as np

from .base_classes import Fabric, Distribution
from .labeler import TyposTypes
from ...utils.lang_utils import SUBSTITUTION_OPTIONS, AVAILABLE_LANG_CODES


class Model:
    """Statistical model parametrized by fetched distributions.

    Given parallel corpus, number of typos per sentence, types of error and their
    corresponding positions and substitution statistics are first gathered.
    Raw statistics are then fed to `Model` and normalized to appropriate discrete
    distributions. `Model` is parametrized by these distributions, and is used
    to corrupt text in a statistic-based manner.

    Attributes:
        debug_mode (bool): used for tests purposes;
        stats (Dict[str, List[int]]): used for tests purposes;
        lang (str): language of original text;
        skip_if_position_not_found (bool): whether to skip typo, when appropriate position cannot be found;

    Usage:
        from labeler import process_mistypings

        sources, corrections = load_data(...)
        typos_cnt, cm, stats = process_mistypings(sources, corrections)
        model = Model(typos_cnt, stats, cm, True, "ru")
        print(model.transform(clean_sentence))
    """
    names = [typo_type.name for typo_type in TyposTypes]

    def __init__(
        self, typos_count: List[int], 
        stats: Dict[str, Dict[str, List[float]]], 
        confusion_matrix: Dict[str, Dict[str, int]],
        skip_if_position_not_found: bool,
        lang: str,
        debug_mode: bool = False,
    ):
        # For debugging purposes only
        self.debug_mode = debug_mode
        self.stats = {
            "used_positions_pre": [],
            "used_positions_after": [],
            "pos": [],
        }

        self.validate_inputs(stats, confusion_matrix, typos_count, lang)

        self.lang = lang.strip("_ ").lower()
        self.skip_if_position_not_found = skip_if_position_not_found

        # Number of mistypings per sentence
        self._register_distribution("number_of_errors_per_sent", typos_count, False)
        
        # Type of mistypings
        typos_cnt = {typo: len(v["abs"]) for typo, v in stats.items()}
        typos = reduce(lambda x, y: x + y, [[k] * v for k, v in typos_cnt.items()])
        self._register_distribution("type_of_typo", typos)
        
        # Relative positions of mistypings
        self._bins = [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
        for typo, v in stats.items():
            # To avoid 1.s being thrown in 11th bucket
            rel_positions = [pos if pos < 1. else pos - 0.00001 for pos in v["rel"]]
            
            buckets = np.digitize(rel_positions, self._bins)
            self._register_distribution(typo + "_positions", buckets)
            
        # Substitutions (confusion matrix)
        for ch, candidates in confusion_matrix.items():
            counts = reduce(lambda x, y: x + y, [[k] * v for k, v in candidates.items()])
            self._register_distribution("substitutions_for_{}".format(ord(ch)), counts)

    @classmethod
    def validate_inputs(cls, stats: Dict[str, Dict[str, List[float]]], confusion_matrix: Dict[str, Dict[str, int]],
                        typos_counts: List[int], lang: str):
        lang = lang.strip("_ ").lower()
        if lang not in AVAILABLE_LANG_CODES:
            raise ValueError(
                "Wrong language code: {}. Available codes are {}".format(lang, " ".join(AVAILABLE_LANG_CODES)))
        if len(stats) == 0:
            raise ValueError("Stats are empty, you should provide some")
        total_pos_num = 0
        for k, v in stats.items():
            if k not in cls.names:
                raise ValueError("You provided stats in wrong format, the key {} is not expected".format(k))
            if len(v["abs"]) != len(v["rel"]):
                raise ValueError("Your inputs' lengths in stats (abs / rel) do not match for {}".format(k))
            illegal_positions = [i for i, elem in enumerate(v["abs"]) if elem < 0]
            if len(illegal_positions) != 0:
                raise ValueError("Provide non-negative values for absolute positions for {} at positions {}".format(
                    k, illegal_positions))
            illegal_positions = [i for i, elem in enumerate(v["rel"]) if elem < 0 or elem > 1]
            if len(illegal_positions) != 0:
                raise ValueError("Provide values between 0 and 1 for relative positions for {} at positions {}".format(
                    k, illegal_positions))
            total_pos_num += len(v["abs"])
        if total_pos_num == 0:
            raise ValueError("Provide some actual statistics")
        if len(typos_counts) == 0:
            raise ValueError("Typos counts are empty, you should provide some")
        if min(typos_counts) < 0:
            raise ValueError("Provide non-negative number of errors")
        if len(confusion_matrix) == 0 and "substitution" in stats and len(stats["substitution"]["abs"]) > 0:
            raise ValueError("Confusion matrix is empty, but substitution is in stats")
        for k, v in confusion_matrix.items():
            if len(k) != 1:
                raise ValueError("Wrong format of key {} in confusion matrix".format(k))
            for sub, count in v.items():
                if len(sub) != 1:
                    raise ValueError("Wrong format of substitution {} in confusion matrix".format(sub))
                if count < 0:
                    raise ValueError("Provide non-negative value for count for key {} and substitution {}".format(
                        k, sub))
        
    def _register_distribution(
            self, distribution: str, evidences: Union[List[int], np.array], exclude_zero: Optional[bool] = False):
        if hasattr(self, distribution):
            raise ValueError("You already defined that distribution {}".format(distribution))
        d = Distribution(evidences, exclude_zero)
        setattr(self, distribution, d)

    def _factorization_scheme(self, interval_idx: int, sequence_length: int) -> [int, int]:
        """Calculates exact absolute edge positions in a sentence, considering relative positions in a sentence.

        Args:
            interval_idx (int):
                interval id ranging from 1 to 10, representing equal non-overlaping semi-open
                intervals in [0,1];
            sequence_length (int): number of characters in a sentence;
        """
        left, right = self._bins[interval_idx - 1], self._bins[interval_idx]
        most_left = math.ceil(sequence_length * left)
        most_right = math.ceil(sequence_length * right)
        return most_left, most_right
    
    def transform(self, sentence: str, rng: np.random.default_rng):
        """Spelling corruption procedure.

        The algorithm follows consequtive steps:
            1. Sample number of errors;
            2. For each error sample its type and corresponding interval in a sentence;
            3. Calculate absolute start and ending positions for typo;
            4. In a given interval find appropriate position for typo;
            5. Insert typo;

        Args:
            sentence (str): original sentence;
            rng (np.random.default_rng): random generator;

        Returns:
            sentence (str): original sentence, but with errors;
        """
        # Sample number of mistypings
        num_typos = self.number_of_errors_per_sent.sample(rng)
        fabric = Fabric()

        for _ in range(num_typos):
            # take len() for every typo, because with each
            # typo length of the sentence changes
            l = len(sentence)

            # sample typo and corresponding interval for position
            typo = self.type_of_typo.sample(rng)
            handler = fabric.get_handler(typo)
            position_distribution = getattr(self, typo + "_positions")
            
            # sample bin a.k.a. interval for typo's position
            # and initial exact position inside this interval
            effective_tries = l
            most_left, most_right = -1, -1
            while effective_tries >= 0:
                interval_idx = position_distribution.sample(rng)
                most_left, most_right = self._factorization_scheme(interval_idx, l)
                if most_right - most_left >= 1:  # for fixed bins that means length of sentence < 10
                    break
                effective_tries -= 1
            if most_right - most_left < 1:
                continue

            pos = rng.integers(low=most_left, high=most_right, size=1)[0]

            # Correct the position
            pos = handler.adjust_position(
                pos, most_left, most_right, self.skip_if_position_not_found,
                fabric.used_positions, rng, self.lang, sentence
            )
            if pos is not None:
                try:
                    substitutions = getattr(self, "substitutions_for_{}".format(ord(sentence[pos].lower())))
                except AttributeError:
                    substitutions = Distribution(getattr(SUBSTITUTION_OPTIONS, self.lang), False)
                sentence = handler.apply(pos, sentence, self.lang, rng, substitutions)

                if self.debug_mode:
                    used_positions_cp = fabric.used_positions.copy()
                    self.stats["used_positions_pre"].append(used_positions_cp)
                    self.stats["pos"].append(pos)

                fabric.finish(pos, typo)

                if self.debug_mode:
                    used_positions_cp = fabric.used_positions.copy()
                    self.stats["used_positions_after"].append(used_positions_cp)

        return sentence
    