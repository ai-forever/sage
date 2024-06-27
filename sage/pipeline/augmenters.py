from abc import ABC, abstractmethod
from typing import Optional
from sage.spelling_corruption.configuration_corruptor import CharAugConfig, WordAugConfig, SBSCConfig
from sage.spelling_corruption.corruptor import CharAugCorruptor, WordAugCorruptor, SBSCCorruptor


class Augmenter(ABC):
    @abstractmethod
    def augment(self, text: str) -> str:
        """Applies augmentation to the given text.

        Args:
            text (str): The input text to augment.

        Returns:
            str: The augmented text.
        """
        pass


class CharAugmenter(Augmenter):
    def __init__(self, config: CharAugConfig):
        self.corruptor = CharAugCorruptor.from_config(config)

    def augment(self, text: str, seed: Optional[int] = None) -> str:
        return self.corruptor.corrupt(text, seed=seed)


class WordAugmenter(Augmenter):
    def __init__(self, config: WordAugConfig):
        self.corruptor = WordAugCorruptor.from_config(config)

    def augment(self, text: str, seed: Optional[int] = None) -> str:
        return self.corruptor.corrupt(text, seed=seed)


class SBSCorruptor(Augmenter):
    def __init__(self, config: SBSCConfig):
        self.corruptor = SBSCCorruptor.from_config(config)

    def augment(self, text: str, seed: Optional[int] = None) -> str:
        return self.corruptor.corrupt(text, seed=seed)
