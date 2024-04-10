"""Abstract API to spelling correction models.

The file also contains available pre-trained models for spelling correction
in Russian and English (yet more is to come).

To see all available models:

    models = [model.name for model in AvailableCorrectors]

To launch one of the available models:

    model_path = AvailableCorrectors.m2m100_1B.value
    ... # pass model path for initialization

"""

import os
import enum
from abc import ABCMeta, abstractmethod
from typing import List, Union, Dict, Optional, Any

import pandas as pd

from ..evaluation.scorer import Scorer
from ..utils.data_load_utils import load_available_dataset_from_hf, DatasetsAvailable


datasets_available = [dataset.name for dataset in DatasetsAvailable]


class AvailableCorrectors(enum.Enum):
    """Available models for spelling and punctuation correction"""

    sage_fredt5_large = "ai-forever/sage-fredt5-large"
    sage_fredt5_distilled_95m = "ai-forever/sage-fredt5-distilled-95m"
    sage_m2m100_1B = "ai-forever/sage-m2m100-1.2B"
    sage_mt5_large = "ai-forever/sage-mt5-large"

    m2m100_1B = "ai-forever/RuM2M100-1.2B"
    m2m100_418M = "ai-forever/RuM2M100-418M"
    fred_large = "ai-forever/FRED-T5-large-spell"
    ent5_large = "ai-forever/T5-large-spell"


class Corrector(metaclass=ABCMeta):
    """Base class for all correctors."""

    @classmethod
    @abstractmethod
    def from_pretrained(cls, model_name_or_path: Union[str, os.PathLike]):
        pass

    def correct(self, sentence: str, prefix: Optional[str] = "", **generation_params) -> List[str]:
        """
        Corrects a single input sentence.

        Parameters
        ----------
        sentence: string, source sentence to correct.
        prefix: string, some models need some sort of a prompting;
        **generation_params: parameters passed to model.generate(...);

        Returns
        -------
        string, corresponding correction.
        """

        return self.batch_correct([sentence], 1, prefix, **generation_params)[-1]

    def evaluate(
            self,
            dataset_name_or_path: Optional[Union[str, os.PathLike]],
            metrics: List,
            batch_size: int,
            prefix: str = "",
            dataset_split: str = "test",
            **generation_params,
    ) -> Dict[str, float]:
        """
        Evaluate the particular model on the spellcheck datasets.

        Args:
            dataset_name_or_path: string, a path to a locally situated dataset or a name of a dataset on HuggingFace;
            metrics: list of string, set of metrics to be used to report performance;
            batch_size: int, size of subsample of input sentences;
            prefix: string, some models need some sort of a prompting;
            dataset_split: string, train / test / dev part to be evaluated on;
            **generation_params: parameters passed to model.generate(...);

        Returns:
            dict[str, float], mapping between metric's name and its corresponding value.

        """
        dataset_name_or_path = str(dataset_name_or_path)
        if dataset_name_or_path in datasets_available:
            sources, corrections = load_available_dataset_from_hf(
                dataset_name_or_path, for_labeler=True, split=dataset_split)
        elif os.path.isdir(dataset_name_or_path):
            if os.path.isfile(os.path.join(dataset_name_or_path, "sources.txt")) and \
                    os.path.isfile(os.path.join(dataset_name_or_path, "corrections.txt")):
                src_file = open(os.path.join(dataset_name_or_path, "sources.txt"), encoding="utf8")
                corr_file = open(os.path.join(dataset_name_or_path, "corrections.txt"), encoding="utf8")
                sources = src_file.read().split("\n")
                corrections = corr_file.read().split("\n")
                src_file.close()
                corr_file.close()
                if len(sources) != len(corrections):
                    raise RuntimeError("Sources and corrections must be of the same length, but get {} vs {}".format(
                        len(sources), len(corrections)))
            elif os.path.isfile(os.path.join(dataset_name_or_path, "data.csv")):
                try:
                    data = pd.read_csv(os.path.join(dataset_name_or_path, "data.csv"))
                except Exception as e:
                    raise RuntimeError("Wrong format of file {}. Raised an error: {}".format(
                        os.path.join(dataset_name_or_path, "data.csv"), str(e)))
                if not ("source" in data and "correction" in data):
                    raise RuntimeError("You must provide 'source' and 'correction' columns in {}".format(
                        os.path.join(dataset_name_or_path, "data.csv")
                    ))
                if data.isna().any().max():
                    raise ValueError("Your data at {} contain unnecessary nans".format(
                        os.path.join(dataset_name_or_path, "data.csv")))
                sources = data.source.values.tolist()
                corrections = data.correction.values.tolist()
            else:
                raise RuntimeError("You must provide either 'data.csv' or 'sources.txt'/'corrections.txt' in {}".format(
                    dataset_name_or_path
                ))
        else:
            raise ValueError("You must provide either valid path or available dataset's name, you provided {}".format(
                dataset_name_or_path
            ))

        answers = self.batch_correct(sources, batch_size, prefix, **generation_params)
        if "num_return_sequences" in generation_params and generation_params["num_return_sequences"] > 1:
            num_sequences = generation_params["num_return_sequences"]
            answers = [batch_answers[::num_sequences] for batch_answers in answers]
        answers = sum(answers, [])
        scorer = Scorer("errant" in metrics)
        metrics_dict = scorer.score(sources, corrections, answers, metrics)
        return metrics_dict

    @abstractmethod
    def batch_correct(
            self,
            sentences: List[str],
            batch_size: int,
            prefix: Optional[str] = "",
            **generation_params,
    ) -> List[List[Any]]:
        """Correct multiple sentences"""
