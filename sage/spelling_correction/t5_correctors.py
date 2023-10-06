"""API to T5-based models for spelling correction.

To load a model:

    from corrector import AvailableCorrectors

    model = T5ModelForSpellingCorruption.from_pretrained(AvailableCorrectors.fred_large.value)
    ...
"""

import os
from typing import List, Optional, Union, Any
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, AutoTokenizer

from .corrector import Corrector


class T5ModelForSpellingCorruption(Corrector):
    """T5-based models."""

    def __init__(self, model_name_or_path: Union[str, os.PathLike]):
        self.model_name_or_path = model_name_or_path
        self.max_model_length = 512

    @classmethod
    def from_pretrained(cls, model_name_or_path: Union[str, os.PathLike]):
        engine = cls(model_name_or_path)
        engine.model = T5ForConditionalGeneration.from_pretrained(model_name_or_path)
        engine.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, eos_token="</s>")

        return engine

    def batch_correct(
            self,
            sentences: List[str],
            batch_size: int,
            prefix: Optional[str] = "",
            **generation_params,
    ) -> List[List[Any]]:
        """Correct multiple sentences"""

        if not hasattr(self, "model"):
            raise RuntimeError("Please load weights using `from_pretrained` method from one of the available models.")
        batches = [sentences[i:i + batch_size] for i in range(0, len(sentences), batch_size)]
        result = []
        pb = tqdm(total=len(batches))
        device = self.model.device
        for batch in batches:
            init_encodings = self.tokenizer.batch_encode_plus(
                batch, max_length=None, padding="longest", truncation=False, return_tensors='pt')
            batch_prefix = [prefix + sentence for sentence in batch]
            encodings = self.tokenizer.batch_encode_plus(
                batch_prefix, max_length=None, padding="longest", truncation=False, return_tensors='pt')
            for k, v in encodings.items():
                encodings[k] = v.to(device)
            generated_tokens = self.model.generate(
                **encodings, **generation_params, max_length=init_encodings["input_ids"].shape[1] + 1)
            ans = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            result.append(ans)
            pb.update(1)
        return result
