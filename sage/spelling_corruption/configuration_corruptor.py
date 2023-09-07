"""Configuration classes for corruption methods.

Currently, three options are maintained: word- and char-level Augmentex and SBSC (Statistic-based
spelling corruption).

Examples:
    from corruptor import WordAugCorruptor

    config = WordAugConfig()
    corruptor = WordAugCorruptor.from_config(config)

    ...

    from corruptor import SBSCCorruptor

    config = SBSCConfig(
        lang="ru",
        reference_dataset_name_or_path="RUSpellRU"
    )
    corruptor = SBSCCorruptor.from_config(config)
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Union, Optional


@dataclass
class WordAugConfig:
    """Word-level Augmentex config.

    Attributes:
        min_aug (int): The minimum amount of augmentation. Defaults to 1.
        max_aug (int): The maximum amount of augmentation. Defaults to 5.
        unit_prob (float): Percentage of the phrase to which augmentations will be applied. Defaults to 0.3.
    """
    min_aug: Optional[int] = field(
        default=1,
        metadata={"help": "The minimum amount of augmentation. Defaults to 1."},
    )

    max_aug: Optional[int] = field(
        default=5,
        metadata={"help": "The maximum amount of augmentation. Defaults to 5."},
    )

    unit_prob: Optional[float] = field(
        default=0.3,
        metadata={
            "help": "Percentage of the phrase to which augmentations will be applied. Defaults to 0.3."}
    )


@dataclass
class CharAugConfig:
    """Char-level Augmentex config.

    Attributes:
        min_aug (int): The minimum amount of augmentation. Defaults to 1.
        max_aug (int): The maximum amount of augmentation. Defaults to 5.
        unit_prob (float): Percentage of the phrase to which augmentations will be applied. Defaults to 0.3.
        mult_num (int): Maximum repetitions of characters. Defaults to 5.
    """
    min_aug: Optional[int] = field(
        default=1,
        metadata={"help": "The minimum amount of augmentation. Defaults to 1."},
    )

    max_aug: Optional[int] = field(
        default=5,
        metadata={"help": "The maximum amount of augmentation. Defaults to 5."},
    )

    unit_prob: Optional[float] = field(
        default=0.3,
        metadata={
            "help": "Percentage of the phrase to which augmentations will be applied. Defaults to 0.3."}
    )

    mult_num: Optional[int] = field(
        default=5,
        metadata={"help": "Maximum repetitions of characters. Defaults to 5."},
    )


@dataclass
class SBSCConfig:
    """Config for statistic-based spelling corruption.

    Attributes:
        lang (str): source language;
        typos_count (List[int]): number of typos per sentence;
        stats (Dict[str, Dict[str, List[float]]]):
            types of typos and their absolute and relative positions in a sentence;
        confusion_matrix (Dict[str, Dict[str, int]]): Candidate replacements with corresponding frequencies;
        skip_if_position_not_found (bool):
            Whether to search for suitable position in a sentence when position is not found in interval;
        reference_dataset_name_or_path (bool): Path to or name of reference dataset
        reference_dataset_split (str): Dataset split to use when acquiring statistics.
    """
    lang: str = field(
        default="ru",
        metadata={"help": "Source language"}
    )
    typos_count: Optional[List[int]] = field(
        default=None,
        metadata={"help": "Number of errors per sentence"},
    )

    stats: Optional[Dict[str, Dict[str, List[float]]]] = field(
        default=None,
        metadata={"help": "Relative and absolute positions of errors of corresponding types"},
    )

    confusion_matrix: Optional[Dict[str, Dict[str, int]]] = field(
        default=None,
        metadata={"help": "Candidate replacements with corresponding frequencies"},
    )

    skip_if_position_not_found: bool = field(
        default=True,
        metadata={
            "help": "Whether to search for suitable position in a sentence when position is not found in interval"},
    )

    reference_dataset_name_or_path: Optional[Union[str, os.PathLike]] = field(
        default="RUSpellRU",
        metadata={"help": "Path to or name of reference dataset"},
    )

    reference_dataset_split: str = field(
        default="train",
        metadata={"help": "Dataset split to use when acquiring statistics."},
    )
