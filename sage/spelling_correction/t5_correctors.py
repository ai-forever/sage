"""API to T5-based models for spelling correction.

To load a model:

    from corrector import AvailableCorrectors

    model = T5ModelForSpellingCorruption.from_pretrained(AvailableCorrectors.fred_large.value)
    ...
"""

import os
from typing import List, Optional, Union, Any

import torch
from tqdm.auto import tqdm
from transformers import T5ForConditionalGeneration, AutoTokenizer

from .corrector import Corrector


class T5ModelForSpellingCorruption(Corrector):
    """T5-based models."""

    def __init__(self, model_name_or_path: Union[str, os.PathLike]):
        """
        Initialize the T5-type corrector from a pre-trained checkpoint.
        The latter can be either locally situated checkpoint or a name of a model on HuggingFace.

        NOTE: This method does not really load the weights, it just stores the path or name.

        :param model_name_or_path: the aforementioned name or path to checkpoint;
        :type model_name_or_path: str or os.PathLike;
        """
        self.model_name_or_path = model_name_or_path
        self.max_model_length = 512

    @classmethod
    def from_pretrained(cls, model_name_or_path: Union[str, os.PathLike]):
        """
        Initialize the T5-type corrector from a pre-trained checkpoint.
        The latter can be either locally situated checkpoint or a name of a model on HuggingFace.

        :param model_name_or_path: the aforementioned name or path to checkpoint;
        :type model_name_or_path: str or os.PathLike
        :return: corrector initialized from pre-trained weights
        :rtype: object of :class:`T5ModelForSpellingCorruption`
        """
        engine = cls(model_name_or_path)
        engine.model = T5ForConditionalGeneration.from_pretrained(model_name_or_path)
        engine.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        return engine

    def batch_correct(
            self,
            sentences: List[str],
            batch_size: int,
            prefix: Optional[str] = "",
            **generation_params,
    ) -> List[List[Any]]:
        """
        Corrects multiple sentences.

        :param sentences: input sentences to correct;
        :type sentences: list of str
        :param batch_size: size of subsample of input sentences;
        :type batch_size: int
        :param prefix: some models need some sort of a prompting;
        :type prefix: str
        :param generation_params: parameters passed to `generate` method of a HuggingFace model;
        :type generation_params: dict
        :return: corresponding corrections
        :rtype: list of list of str
        """
        if not hasattr(self, "model"):
            raise RuntimeError("Please load weights using `from_pretrained` method from one of the available models.")
        batches = [sentences[i:i + batch_size] for i in range(0, len(sentences), batch_size)]
        result = []
        pb = tqdm(total=len(batches))
        device = self.model.device
        for batch in batches:
            batch_prefix = [prefix + sentence for sentence in batch]
            with torch.inference_mode():
                encodings = self.tokenizer.batch_encode_plus(
                    batch_prefix, max_length=None, padding="longest", truncation=False, return_tensors='pt')
                for k, v in encodings.items():
                    encodings[k] = v.to(device)
                generated_tokens = self.model.generate(
                    **encodings, **generation_params, max_length=encodings['input_ids'].size(1) * 1.5)
                ans = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            result.append(ans)
            pb.update(1)
        return result
