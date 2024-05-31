from typing import Optional, List
import random
from .config import PipelineConfig
from .augmentors import CharAugmentor, WordAugmentor, SBSCorruptor
from sage.spelling_corruption.configuration_corruptor import CharAugConfig, WordAugConfig, SBSCConfig


class AugmentationPipeline:
    def __init__(self, config: PipelineConfig = PipelineConfig(), shuffle: bool = True):
        self.augmentors = []
        self.config = config
        self._add_all_augmentors()
        if shuffle:
            self._shuffle_augmentors()

    def _add_all_augmentors(self):
        self.add_char_augmentor()
        self.add_word_augmentor()
        self.add_sbsc_augmentor()

    def _shuffle_augmentors(self):
        random.shuffle(self.augmentors)

    def add_char_augmentor(self):
        char_config = CharAugConfig(
            min_aug=self.config.char_min_aug,
            max_aug=self.config.char_max_aug,
            unit_prob=self.config.char_unit_prob
        )
        self.augmentors.append(CharAugmentor(char_config))

    def add_word_augmentor(self):
        word_config = WordAugConfig(
            min_aug=self.config.word_min_aug,
            max_aug=self.config.word_max_aug,
            unit_prob=self.config.word_unit_prob
        )
        self.augmentors.append(WordAugmentor(word_config))

    def add_sbsc_augmentor(self):
        sbsc_config = SBSCConfig(
            lang=self.config.sbsc_lang,
            reference_dataset_name_or_path=self.config.sbsc_reference_dataset_name_or_path,
            reference_dataset_split=self.config.sbsc_reference_dataset_split
        )
        self.augmentors.append(SBSCorruptor(sbsc_config))

    def remove_augmentor(self, augmentor_type):
        self.augmentors = [augmentor for augmentor in self.augmentors if not isinstance(augmentor, augmentor_type)]

    def set_order(self, order: List[int]):
        self.augmentors = [self.augmentors[i] for i in order]

    def augment(self, text: str, seed: Optional[int] = None) -> str:
        for augmentor in self.augmentors:
            text = augmentor.augment(text, seed=seed)
        return text
