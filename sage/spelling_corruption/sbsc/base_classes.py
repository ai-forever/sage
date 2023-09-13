"""Base classes for misspellings.

Includes parent abstract class and corresponding APIs for each type of errors,
as well as API to discrete distributions (`class Distribution`).
"""

import logging
from abc import ABCMeta, abstractmethod
from typing import Optional, Callable, List

import numpy as np

from .typings_positions_conditions import initialize_conditions
from ...utils.lang_utils import INSERTION_OPTIONS

conditions = initialize_conditions()
MISSPELLINGS = {}


def register_misspelling(cls):
    MISSPELLINGS[cls.description()] = cls()


class Distribution:
    """Emulates discrete distribution."""

    def __init__(self, evidences: List[int], exclude_zero: bool):
        if exclude_zero:
            evidences = [elem for elem in evidences if elem != 0]
        self.values, counts = np.unique(evidences, return_counts=True)
        self.p = counts / sum(counts)

    def sample(self, rng: np.random.default_rng):
        if len(self.values) == 0:
            raise ValueError("You cannot sample from empty distribution, provide some statistics first")
        value = rng.choice(self.values, size=1, p=self.p)[0]
        return value


class Typo(metaclass=ABCMeta):
    """Base class for all handlers.

    Attributes:
        condition (typings_positions_conditions.Condition):
            condition for appropriate position of a typo in a sentence.
    """

    def __init__(self):
        self.condition = None if self.desc is None else conditions[self.desc]

    @staticmethod
    @abstractmethod
    def description():
        """We will need this in object of Fabric class
        when instantiating an object from dict of possible misspellings.
        """

    @property
    @abstractmethod
    def desc(self):
        """We need this to identify particular type of error
        And use it while initialization.
        """

    @abstractmethod
    def apply(
            self, pos: int, sentence: str, lang: str, rng: np.random.default_rng,
            substitutions: Optional[Distribution] = None
    ) -> str:
        """Insert typo in particular `pos` in a `sentence`.

        Args:
            pos (int): position to insert typo;
            sentence (str): original sentence;
            lang (str): language code;
            rng (np.random.default_rng): random generator;
            substitutions (Distribution): optional, set of options for substitution;
        """

    def adjust_position(
            self, pos: int, most_left: int, most_right: int, skip_if_position_not_found: bool,
            used_positions: List[int], rng: np.random.default_rng, lang: str, sentence: Optional[str] = None
    ) -> int:
        """Select appropriate position in interval from `most_left` to `most_right`
        starting from `pos` in a `sentence`.

        Args:
            pos (int): starting position;
            most_left (int): starting position of interval;
            most_right (int): ending position of interval;
            skip_if_position_not_found (bool): whether to skip, when appropriate position for typo cannot be found;
            used_positions (List[int]): array of taken positions;
            rng (np.random.default_rng): random generator;
            lang (str): language code;
            sentence (str): original sentence;
        """
        effective_tries = (most_right - most_left) * 2
        cnt_tries = 0
        while self.condition.condition(pos, used_positions, sentence, lang):
            pos = rng.integers(low=most_left, high=most_right, size=1)[0]
            cnt_tries += 1
            if cnt_tries == effective_tries:
                logging.info("Falling back on {}".format(self.desc))
                pos = self._fallback(skip_if_position_not_found)(used_positions, lang, most_left, most_right, sentence)
                break
        return pos

    def _fallback(self, skip_if_position_not_found: bool) -> Callable:
        if skip_if_position_not_found:
            return self._skip_fallback_strategy
        return self._default_fallback_strategy

    def _default_fallback_strategy(self, used_positions: List[int], lang: str, most_left: Optional[int] = None,
                                   most_right: Optional[int] = None, sentence: Optional[str] = None
                                   ) -> Optional[int]:
        """Iterate through the whole `sentence` and search for appropriate position.
         When one found, stop iterating.

         Args:
             used_positions (List[int]): array of taken positions;
             lang (str): language code;
             most_left (int): starting position of interval;
             most_right (int): ending position of interval;
             sentence (str): original sentence;
        """
        pos = None
        for i, ch in enumerate(sentence):
            if self.condition.condition(i, used_positions, sentence, lang):
                continue
            pos = i
            break
        return pos

    @staticmethod
    def _skip_fallback_strategy(used_positions: List[int], lang: str, most_left: Optional[int] = None,
                                most_right: Optional[int] = None, sentence: Optional[str] = None,
                                ) -> Optional[int]:
        """Skipping current typo if there is no appropriate position for it."""

        return None


class Fabric:
    """This acts as somewhat factory for the handlers.

    Attributes:
        used_positions (List[int]): array of taken positions;
    """

    def __init__(self):
        self.used_positions = []

    def finish(self, pos: int, typo: str) -> None:
        """Alter `used_positions` after typo has been inserted.

        Args:
            pos (int): position of typo;
            typo (str): type of typo;
        """
        self.used_positions.append(pos)
        executor = conditions[typo]
        executor.alter_positions(pos, self.used_positions)

    @staticmethod
    def get_handler(typo: str) -> Typo:
        return MISSPELLINGS[typo]


@register_misspelling
class Insertion(Typo):
    """API to insertion typo.

    Insertion type of error implies insertion of unnecessary characters in
    an original sentence.

    Examples of error:
        1. Error -> Errror;
        2. Мама дома мыла раму -> Марма дома мыла раму;
    """

    @staticmethod
    def description():
        return "insertion"

    @property
    def desc(self):
        return "insertion"

    def apply(
            self, pos: int, sentence: str, lang: str, rng: np.random.default_rng,
            substitutions: Optional[Distribution] = None
    ) -> str:
        insertions = getattr(INSERTION_OPTIONS, lang)
        insertion = rng.choice(insertions, size=1)[0]
        sentence = sentence[:pos] + insertion + sentence[pos:]
        return sentence


@register_misspelling
class Deletion(Typo):
    """API to deletion typo.

    Deletion type of error implies deletion of characters in
    an original sentence.

    Examples of error:
        1. Error -> Eror;
        2. Мама дома мыла раму -> Мма дома мыла раму;
    """

    @staticmethod
    def description():
        return "deletion"

    @property
    def desc(self):
        return "deletion"

    def apply(
            self, pos: int, sentence: str, lang: str, rng: np.random.default_rng,
            substitutions: Optional[Distribution] = None
    ) -> str:
        sentence = sentence[:pos] + sentence[min(len(sentence), pos + 1):]
        return sentence


@register_misspelling
class Transposition(Typo):
    """API to transposition typo.

    Transposition type of error implies swapping two adjacent characters.

    Examples of error:
        1. Error -> Errro;
        2. Мама дома мыла раму -> Маам дома мыла раму;
    """

    @staticmethod
    def description():
        return "transposition"

    @property
    def desc(self):
        return "transposition"

    def apply(
            self, pos: int, sentence: str, lang: str, rng: np.random.default_rng,
            substitutions: Optional[Distribution] = None
    ) -> str:
        sentence = sentence[:pos] + sentence[pos + 1] + sentence[pos] + sentence[min(len(sentence), pos + 2):]
        return sentence


@register_misspelling
class Substitution(Typo):
    """API to substitution typo.

    Substitution type of error implies substitution of one character
    in an original sentence.

    Examples of error:
        1. Error -> Errar;
        2. Мама дома мыла раму -> Мама Мома мыла раму;
    """
    @staticmethod
    def description():
        return "substitution"

    @property
    def desc(self):
        return "substitution"

    def apply(
            self, pos: int, sentence: str, lang: str, rng: np.random.default_rng,
            substitutions: Optional[Distribution] = None
    ) -> str:
        substitution = substitutions.sample(rng)
        if sentence[pos].isupper():
            substitution = substitution.upper()
        sentence = sentence[:pos] + substitution + sentence[min(len(sentence), pos + 1):]
        return sentence


@register_misspelling
class ExtraSeparator(Typo):
    """API to extra separator typo.

    ExtraSeparator type of error implies insertion of extra gap
    in an original sentence.

    Examples of error:
        1. Error -> Err or;
        2. Мама дома мыла раму -> Ма ма дома мыла раму;
    """
    @staticmethod
    def description():
        return "extra_separator"

    @property
    def desc(self):
        return "extra_separator"

    def apply(
            self, pos: int, sentence: str, lang: str, rng: np.random.default_rng,
            substitutions: Optional[Distribution] = None
    ) -> str:
        sentence = sentence[:pos] + " " + sentence[pos:]
        return sentence


@register_misspelling
class MissingSeparator(Typo):
    """API to missing separator typo.

    MissingSeparator type of error implies deletion of a gap
    in an original sentence.

    Examples of error:
        1. Error specifically made for this example-> Errorspecifically made for this example;
        2. Мама дома мыла раму -> Мама домамыла раму;
    """
    @staticmethod
    def description():
        return "missing_separator"

    @property
    def desc(self):
        return "missing_separator"

    def apply(
            self, pos: int, sentence: str, lang: str, rng: np.random.default_rng,
            substitutions: Optional[Distribution] = None
    ) -> str:
        sentence = sentence[:pos] + sentence[min(len(sentence), pos + 1):]
        return sentence
