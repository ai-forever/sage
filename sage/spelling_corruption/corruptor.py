"""API to available methods of spelling corruption.

Currently, three options are available: word- and char-level Augmentex and
Statistical-based spelling corruption (SBSC).

Examples:
    from configuration_corruptor import CharAugConfig

    config = CharAugConfig(min_aug=10, max_aug=50, unit_prob=0.5)
    corruptor = CharAugCorruptor.from_config(config)
    print(corruptor.corrupt(sentence))

    ...

    corruptor = SBSCCorruptor.from_default_config()
    print(corruptor.corrupt(sentence))
"""

import dataclasses
from dataclasses import asdict
from typing import List, Union, Optional
from abc import ABCMeta, abstractmethod

from augmentex.char import CharAug
from augmentex.word import WordAug

from .sbsc.sbsc import StatisticBasedSpellingCorruption
from .configuration_corruptor import WordAugConfig, CharAugConfig, SBSCConfig


class Corruptor(metaclass=ABCMeta):
    """Base class for all corruptors.

    Attributes:
        config (Dict[str, Any]): config for every particular corruption class;
        engine (Union[WordAugCorruptor, CharAugCorruptor, SBSCCorruptor]):
            corruptor class;
    """

    engine = None

    def __init__(self):
        self.config = asdict(self.get_default_config())

    @classmethod
    def from_config(cls, config: Union[WordAugConfig, CharAugConfig, SBSCConfig]):
        """Initialize corruptor from a given config.

        Args:
            config (Union[WordAugConfig, CharAugConfig, SBSCConfig]):
                config for every particular corruption class;

        Returns:
            particular corruptor class initialized from a given config;
        """
        corruptor = cls()
        corruptor.config = {field.name: getattr(config, field.name) for field in dataclasses.fields(config)}
        corruptor.engine = corruptor.engine(**corruptor.config)

        return corruptor

    @classmethod
    def from_default_config(cls):
        """Initialize corruptor from a default config.

        Returns:
            particular corruptor class initialized from a default config;
        """
        corruptor = cls()
        corruptor.engine = corruptor.engine(**corruptor.config)

        return corruptor

    @abstractmethod
    def corrupt(self, sentence: str, action: Optional[str] = None) -> str:
        pass

    @abstractmethod
    def batch_corrupt(
            self, sentences: List[str], action: Optional[str] = None, batch_prob: Optional[float] = 0.3) -> List[str]:
        pass

    @staticmethod
    @abstractmethod
    def get_default_config():
        pass


class AugCorruptor(Corruptor, metaclass=ABCMeta):
    """Base class for Augmentex-based corruptors."""

    def corrupt(self, sentence: str, action: Optional[str] = None) -> str:
        return self.engine.augment(sentence, action=action)

    def batch_corrupt(
            self, sentences: List[str], action: Optional[str] = None, batch_prob: Optional[float] = 0.3) -> List[str]:
        return self.engine.aug_batch(sentences, batch_prob=batch_prob, action=action)


class WordAugCorruptor(AugCorruptor):
    engine = WordAug

    @staticmethod
    def get_default_config():
        return WordAugConfig()


class CharAugCorruptor(AugCorruptor):
    engine = CharAug

    @staticmethod
    def get_default_config():
        return CharAugConfig()


class SBSCCorruptor(Corruptor):
    engine = StatisticBasedSpellingCorruption

    def corrupt(self, sentence: str, action: Optional[str] = None) -> str:
        return self.engine.corrupt(sentence)

    def batch_corrupt(self, sentences: List[str], action: Optional[str] = None, batch_prob: Optional[float] = 0.3) -> \
    List[str]:
        return self.engine.batch_corrupt(sentences)

    @staticmethod
    def get_default_config():
        return SBSCConfig()
