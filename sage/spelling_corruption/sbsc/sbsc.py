"""API to Statistical-based Spelling Corruption method.

Examples:
    corruptor = StatisticBasedSpellingCorruption(
        lang="ru",
        reference_dataset_name_or_path="RUSpellRU",
    )
    print(corruptor.corrupt(sentence))

    ....

    from labeler import process_mistypings

    sources, corrections = load_data(...)
    typos_cnt, cm, stats = process_mistypings(sources, corrections)
    corruptor = StatisticBasedSpellingCorruption(
        lang="ru",
        typos_count=typos_cnt,
        stats=stats,
        confusion_matrix=cm,
    )
    print(corruptor.corrupt(sentence))
"""

import os
from typing import List, Dict, Optional, Union

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .model import Model
from .labeler import process_mistypings
from ...utils.data_load_utils import load_available_dataset_from_hf, DatasetsAvailable

datasets_available = [dataset.name for dataset in DatasetsAvailable]


class StatisticBasedSpellingCorruption:
    """API to `Model` class from model.py.

    Attributes:
        model (model.Model): statistic-based spelling corruption model;
    """

    def __init__(
            self,
            lang: str,
            typos_count: Optional[List[int]] = None,
            stats: Optional[Dict[str, Dict[str, List[float]]]] = None,
            confusion_matrix: Optional[Dict[str, Dict[str, int]]] = None,
            skip_if_position_not_found: bool = True,
            reference_dataset_name_or_path: Optional[Union[str, os.PathLike]] = None,
            reference_dataset_split: str = "train",
    ):
        typos_count_ = None
        stats_ = None
        confusion_matrix_ = None

        if (typos_count is None or stats is None or confusion_matrix is None) and reference_dataset_name_or_path is None:
            raise RuntimeError('''You should provide at least one of :typos_count:/:stats:/:confusion_matrix:
                                or :reference_dataset_name_or_path:''')
        if (typos_count is None or stats is None or confusion_matrix is None) and \
                reference_dataset_name_or_path is not None:
            reference_dataset_name_or_path = str(reference_dataset_name_or_path)
            if reference_dataset_name_or_path in datasets_available:
                sources, corrections = load_available_dataset_from_hf(
                    reference_dataset_name_or_path, for_labeler=True, split=reference_dataset_split)
                stats_, confusion_matrix_, typos_count_ = process_mistypings(sources, corrections)
            elif os.path.isdir(reference_dataset_name_or_path):
                if os.path.isfile(os.path.join(reference_dataset_name_or_path, "sources.txt")) and \
                   os.path.isfile(os.path.join(reference_dataset_name_or_path, "corrections.txt")):
                    src_file = open(os.path.join(reference_dataset_name_or_path, "sources.txt"))
                    corr_file = open(os.path.join(reference_dataset_name_or_path, "corrections.txt"))
                    sources = src_file.read().split("\n")
                    corrections = corr_file.read().split("\n")
                    src_file.close()
                    corr_file.close()
                    if len(sources) != len(corrections):
                        raise RuntimeError("Sources and corrections must be of the same length, but get {} vs {}".format(
                            len(sources), len(corrections)))
                    stats_, confusion_matrix_, typos_count_ = process_mistypings(sources, corrections)
                elif os.path.isfile(os.path.join(reference_dataset_name_or_path, "data.csv")):
                    try:
                        data = pd.read_csv(os.path.join(reference_dataset_name_or_path, "data.csv"))
                    except Exception as e:
                        raise RuntimeError("Wrong format of file {}. Raised an error: {}".format(
                            os.path.join(reference_dataset_name_or_path, "data.csv"), str(e)))
                    if not ("source" in data and "correction" in data):
                        raise RuntimeError("You must provide 'source' and 'correction' columns in {}".format(
                            os.path.join(reference_dataset_name_or_path, "data.csv")
                        ))
                    if data.isna().any().max():
                        raise ValueError("Your data at {} contain unnecessary nans".format(
                            os.path.join(reference_dataset_name_or_path, "data.csv")))
                    sources = data.source.values.tolist()
                    corrections = data.correction.values.tolist()
                    stats_, confusion_matrix_, typos_count_ = process_mistypings(sources, corrections)
                else:
                    raise RuntimeError("You must provide either 'data.csv' or 'sources.txt'/'corrections.txt' in {}".format(
                        reference_dataset_name_or_path
                    ))
            else:
                raise ValueError("You must provide either valid path or available dataset's name, you provided {}".format(
                    reference_dataset_name_or_path
                ))
        if typos_count is not None:
            typos_count_ = typos_count
        if stats is not None:
            stats_ = stats
        if confusion_matrix is not None:
            confusion_matrix_ = confusion_matrix

        self.model = Model(
            typos_count=typos_count_,
            stats=stats_,
            confusion_matrix=confusion_matrix_,
            skip_if_position_not_found=skip_if_position_not_found,
            lang=lang,
        )

    @staticmethod
    def show_reference_datasets_available():
        print(*datasets_available, sep="\n")

    def corrupt(self, sentence: str, seed: int) -> str:
        return self.batch_corrupt([sentence], seed)[0]

    def batch_corrupt(self, sentences: List[str], seed: int) -> List[str]:
        result = []
        pb = tqdm(total=len(sentences))
        rng = np.random.default_rng(seed)
        for sentence in sentences:
            result.append(self.model.transform(sentence, rng))
            pb.update(1)
        return result
