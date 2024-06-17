import random
from typing import Optional, List
from sage.spelling_corruption.configuration_corruptor import CharAugConfig, WordAugConfig, SBSCConfig
from .augmenters import CharAugmenter, WordAugmenter, SBSCorruptor
from .config import PipelineConfig


class AugmentationPipeline:
    def __init__(self, config: PipelineConfig = PipelineConfig(), shuffle: bool = True):
        """
        Initializes the AugmentationPipeline with a given configuration and optional shuffling.

        Args:
            config (PipelineConfig): The configuration object containing settings for the augmenters.
            shuffle (bool): Whether to shuffle the order of augmenters. Default is True.
        """
        self.augmenters = []
        self.config = config
        self._add_all_augmenters()
        if shuffle:
            self._shuffle_augmenters()

    def _add_all_augmenters(self):
        """
        Adds all available augmenters (character, word, and SBS corruptor) to the pipeline.
        """
        self.add_char_augmenter()
        self.add_word_augmenter()
        self.add_sbsc_augmenter()

    def _shuffle_augmenters(self):
        """
        Randomly shuffles the order of the augmenters in the pipeline.
        """
        random.shuffle(self.augmenters)

    def add_char_augmenter(self):
        """
        Adds a character augmenter to the pipeline using the configuration settings.
        """
        char_config = CharAugConfig(
            min_aug=self.config.char_min_aug,
            max_aug=self.config.char_max_aug,
            unit_prob=self.config.char_unit_prob
        )
        self.augmenters.append(CharAugmenter(char_config))

    def add_word_augmenter(self):
        """
        Adds a word augmenter to the pipeline using the configuration settings.
        """
        word_config = WordAugConfig(
            min_aug=self.config.word_min_aug,
            max_aug=self.config.word_max_aug,
            unit_prob=self.config.word_unit_prob
        )
        self.augmenters.append(WordAugmenter(word_config))

    def add_sbsc_augmenter(self):
        """
        Adds an SBS corruptor to the pipeline using the configuration settings.
        """
        sbsc_config = SBSCConfig(
            lang=self.config.sbsc_lang,
            reference_dataset_name_or_path=self.config.sbsc_reference_dataset_name_or_path,
            reference_dataset_split=self.config.sbsc_reference_dataset_split
        )
        self.augmenters.append(SBSCorruptor(sbsc_config))

    def remove_augmenter(self, augmenter_type):
        """
        Removes all instances of the specified augmenter type from the pipeline.

        Args:
            augmenter_type (type): The type of augmenter to remove.
        """
        self.augmenters = [augmenter for augmenter in self.augmenters if not isinstance(augmenter, augmenter_type)]

    def set_order(self, order: List[int]):
        """
        Sets a specific order for the augmenters in the pipeline.

        Args:
            order (List[int]): A list of indices specifying the new order of augmenters.
        """
        self.augmenters = [self.augmenters[i] for i in order]

    def augment(self, text: str, seed: Optional[int] = None) -> str:
        """
        Applies all augmenters in the pipeline to the given text.

        Args:
            text (str): The input text to augment.
            seed (Optional[int]): An optional seed for random number generation to ensure reproducibility.

        Returns:
            str: The augmented text.
        """
        for augmenter in self.augmenters:
            text = augmenter.augment(text, seed=seed)
        return text
