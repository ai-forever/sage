"""Conditions to search for appropriate position for corresponding typo.

Each class embodies necessary conditions in order for particular type of error
to be properly inserted.
"""

import string
from abc import ABCMeta, abstractmethod
from typing import List


class Condition(metaclass=ABCMeta):
    """Base class for all conditions."""

    @staticmethod
    @abstractmethod
    def condition(pos: int, used_positions: List[int], sentence: str, lang: str) -> bool:
        """Checks whether particular position `pos` satisfies typo's requirements.

        Args:
            pos (int): position to check on;
            used_positions (List[str]): taken positions;
            sentence (str): original sentence;
            lang (str): language of original sentence;

        Returns:
            Whether or not `pos` is appropriate position to insert a particular typo.
        """

    @staticmethod
    @abstractmethod
    def alter_positions(pos: int, used_positions: List[int]):
        """Corrects list of taken positions accordingly.

        When typo is inserted, one must include its position in a list of
        taken positions and alter present positions.

        Args:
            pos (int): position of inserted typo;
            used_positions (List[str]): list of taken positions;
        """


class InsertionConditions(Condition):
    @staticmethod
    def condition(pos: int, used_positions: List[int], sentence: str, lang: str) -> bool:
        return pos in used_positions

    @staticmethod
    def alter_positions(pos: int, used_positions: List[int]):
        for i, p in enumerate(used_positions):
            used_positions[i] = p if p <= pos else p + 1


class DeletionConditions(Condition):
    @staticmethod
    def condition(pos: int, used_positions: List[int], sentence: str, lang: str) -> bool:
        punctuation = string.punctuation.replace("-", "")
        return pos in used_positions or sentence[pos] == " " or sentence[pos] in punctuation

    @staticmethod
    def alter_positions(pos: int, used_positions: List[int]):
        for i, p in enumerate(used_positions):
            used_positions[i] = p if p <= pos else p - 1


class TranspositionConditions(Condition):
    @staticmethod
    def condition(pos: int, used_positions: List[int], sentence: str, lang: str) -> bool:
        return pos == len(sentence) - 1 or pos in used_positions or (pos + 1) in used_positions or \
               sentence[pos] in string.punctuation or sentence[pos + 1] in string.punctuation or \
               sentence[pos] == sentence[pos + 1]

    @staticmethod
    def alter_positions(pos: int, used_positions: List[int]):
        used_positions.append(pos + 1)


class SubstitutionConditions(Condition):
    @staticmethod
    def condition(pos: int, used_positions: List[int], sentence: str, lang: str) -> bool:
        return pos in used_positions or sentence[pos] == " " or sentence[pos] in string.punctuation or \
               sentence[pos] in string.digits or ((sentence[pos] in string.ascii_letters) == (lang == "ru"))

    @staticmethod
    def alter_positions(pos: int, used_positions: List[int]):
        pass


class ExtraSeparatorConditions(Condition):
    @staticmethod
    def condition(pos: int, used_positions: List[int], sentence: str, lang: str) -> bool:
        return pos == 0 or sentence[pos - 1] == " " or sentence[pos] == " " or pos in used_positions or \
               sentence[pos] in string.punctuation

    @staticmethod
    def alter_positions(pos: int, used_positions: List[int]):
        for i, p in enumerate(used_positions):
            used_positions[i] = p if p <= pos else p + 1


class MissingSeparatorConditions(Condition):
    @staticmethod
    def condition(pos: int, used_positions: List[int], sentence: str, lang: str) -> bool:
        return sentence[pos] != " "

    @staticmethod
    def alter_positions(pos: int, used_positions: List[int]):
        for i, p in enumerate(used_positions):
            used_positions[i] = p if p <= pos else p - 1


def initialize_conditions():
    return {
        "insertion": InsertionConditions,
        "deletion": DeletionConditions,
        "substitution": SubstitutionConditions,
        "transposition": TranspositionConditions,
        "extra_separator": ExtraSeparatorConditions,
        "missing_separator": MissingSeparatorConditions,
    }
