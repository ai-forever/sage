import os
from sage.utils import DatasetsAvailable


class PipelineConfig:
    def __init__(self, lang: str = 'ru'):
        self.char_min_aug: int = 1
        self.char_max_aug: int = 3
        self.char_unit_prob: float = 0.2
        self.word_min_aug: int = 1
        self.word_max_aug: int = 3
        self.word_unit_prob: float = 0.3
        self.sbsc_lang: str = lang
        self.sbsc_reference_dataset_name_or_path: str = DatasetsAvailable.MedSpellchecker.name
        self.sbsc_reference_dataset_split: str = "test"
        self.__set_language(lang)

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

    def __set_language(self, lang: str):
        """
        Sets the language and corresponding dataset path based on the provided language.

        Args:
            lang (str): The language code ('ru' or 'en').
        """
        self.sbsc_lang = lang
        if lang == 'en':
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
            self.sbsc_reference_dataset_name_or_path = os.path.join(base_dir, "data", "example_data", "jfleg")
        elif lang != 'ru':
            raise ValueError("Unsupported language. Supported languages are 'ru' and 'en'.")

    @property
    def char_params(self):
        """Returns the parameters for character augmentation as a dictionary."""
        return {
            'min_aug': self.char_min_aug,
            'max_aug': self.char_max_aug,
            'unit_prob': self.char_unit_prob
        }

    @property
    def word_params(self):
        """Returns the parameters for word augmentation as a dictionary."""
        return {
            'min_aug': self.word_min_aug,
            'max_aug': self.word_max_aug,
            'unit_prob': self.word_unit_prob
        }

    @property
    def sbsc_params(self):
        """Returns the parameters for SBS corruptor as a dictionary."""
        return {
            'lang': self.sbsc_lang,
            'dataset_name_or_path': self.sbsc_reference_dataset_name_or_path,
            'dataset_split': self.sbsc_reference_dataset_split
        }
