from dataclasses import dataclass, field
from sage.utils import DatasetsAvailable


@dataclass
class PipelineConfig:
    char_min_aug: int = field(default=1)
    char_max_aug: int = field(default=3)
    char_unit_prob: float = field(default=0.2)
    word_min_aug: int = field(default=1)
    word_max_aug: int = field(default=3)
    word_unit_prob: float = field(default=0.3)
    sbsc_lang: str = field(default="ru")
    sbsc_reference_dataset_name_or_path: str = field(default=DatasetsAvailable.MedSpellchecker.name)
    sbsc_reference_dataset_split: str = field(default="test")

    def set_char_params(self, min_aug: int, max_aug: int, unit_prob: float):
        self.char_min_aug = min_aug
        self.char_max_aug = max_aug
        self.char_unit_prob = unit_prob

    def set_word_params(self, min_aug: int, max_aug: int, unit_prob: float):
        self.word_min_aug = min_aug
        self.word_max_aug = max_aug
        self.word_unit_prob = unit_prob

    def set_sbsc_params(self, lang: str, dataset_name_or_path: str, dataset_split: str):
        self.sbsc_lang = lang
        self.sbsc_reference_dataset_name_or_path = dataset_name_or_path
        self.sbsc_reference_dataset_split = dataset_split
