"""API to M2M100-based models for spelling correction.

To load a model:

    from corrector import AvailableCorrectors

    model = RuM2M100ModelForSpellingCorrection.from_pretrained(AvailableCorrectors.m2m100_1B.value)
    ...
"""

import os
from typing import List, Optional, Union, Any
from tqdm import tqdm
from transformers import M2M100ForConditionalGeneration
from transformers.models.m2m_100.tokenization_m2m_100 import M2M100Tokenizer

from .corrector import Corrector

# TODO: remove auth_token when public release


class RuM2M100ModelForSpellingCorrection(Corrector):
    """M2M100-based models."""

    def __init__(self, model_name_or_path: Union[str, os.PathLike]):
        self.model_name_or_path = model_name_or_path

    @classmethod
    def from_pretrained(cls, model_name_or_path: Union[str, os.PathLike]):
        engine = cls(model_name_or_path)
        engine.model = M2M100ForConditionalGeneration.from_pretrained(model_name_or_path, use_auth_token=True)
        engine.tokenizer = M2M100Tokenizer.from_pretrained(model_name_or_path, src_lang="ru", tgt_lang="ru")

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
        if "forced_bos_token_id" in generation_params:
            generation_params.pop("forced_bos_token_id")
        for batch in batches:
            encodings = self.tokenizer.batch_encode_plus(
                batch, max_length=None, padding="longest", truncation=False, return_tensors='pt')
            for k, v in encodings.items():
                encodings[k] = v.to(device)
            generated_tokens = self.model.generate(
                **encodings, **generation_params, forced_bos_token_id=self.tokenizer.get_lang_id("ru"),
                max_length=int(1.5*encodings["input_ids"].shape[1]))
            ans = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            result.append(ans)
            pb.update(1)
        return result
