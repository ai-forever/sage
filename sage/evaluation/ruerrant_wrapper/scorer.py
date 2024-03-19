"""A wrapper over the 'errant' library fork from https://github.com/Askinkaty/errant/.

This implemetation brings some changes over the fork (which provided initial adaptation
of the ERRANT metric for the Russian language). The changes deal with token merging and
error classification and are described in detail in the readme.
"""

from __future__ import annotations

import re
from collections import Counter, namedtuple
from typing import Iterable
from tqdm.auto import tqdm

from errant.annotator import Annotator
from errant.commands.compare_m2 import process_edits
from errant.commands.compare_m2 import evaluate_edits
from errant.commands.compare_m2 import merge_dict
from errant.edit import Edit
import spacy
from spacy.tokenizer import Tokenizer 
from spacy.util import compile_prefix_regex, compile_infix_regex, compile_suffix_regex 

from sage.evaluation.ruerrant_wrapper import classifier
from sage.evaluation.ruerrant_wrapper import merger


def update_spacy_tokenizer(nlp):
    """
    Changes Spacy tokenizer to parse additional patterns.
    """
    infix_re = compile_infix_regex(nlp.Defaults.infixes[:-1] + ["\]\("])
    simple_url_re = re.compile(r'''^https?://''')
    nlp.tokenizer = Tokenizer(
        nlp.vocab,
        prefix_search=compile_prefix_regex(nlp.Defaults.prefixes + ['\\\\\"']).search,
        suffix_search=compile_suffix_regex(nlp.Defaults.suffixes + ['\\\\']).search,
        infix_finditer=infix_re.finditer,
        token_match=None,
        url_match=simple_url_re.match
    )
    return nlp


class RuErrantScorer:
    """A scorer to evaluate spelling correction triplets with ERRANT metric."""

    def __init__(self) -> None:
        self.annotator = Annotator("ru",
                                   nlp=update_spacy_tokenizer(spacy.load("ru_core_news_lg")),
                                   merger=merger,
                                   classifier=classifier)

    def annotate_errors(self, orig: str, cor: str, merging: str = "rules") -> list[Edit]:
        """
        Overrides `Annotator.annotate()` function to allow multiple errors per token.
        This is nesessary to parse combined errors, e.g.:
            ["werd", "Word"] >>> Errors: ["SPELL", "CASE"]
        The `classify()` method called inside is implemented in ruerrant_classifier.py
        (also overrides the original classifier).
        """

        alignment = self.annotator.align(orig, cor, False)
        edits = self.annotator.merge(alignment, merging)
        classified_edits = []
        for edit in edits:
            classified_edits.extend(self.annotator.classify(edit))
        return sorted(classified_edits, key=lambda x: (x[0], x[2]))

    def evaluate(self,
                 sources: Iterable[str],
                 corrections: Iterable[str],
                 answers: Iterable[str]) -> dict[str, tuple[float, float, float]]:
        """
        Evaluates iterables of sources, hyp and ref corrections with ERRANT metric.

        Args:
            sources (Iterable[str]): an iterable of source texts;
            corrections (Iterable[str]): an iterable of gold corrections for the source texts;
            answers (Iterable[str]): an iterable of evaluated corrections for the source texts;

        Returns:
            dict[str, tuple[float, ...]]: a dict mapping error categories to the corresponding
            P, R, F1 metric values.
        """

        best_dict = Counter({"tp": 0, "fp": 0, "fn": 0})
        best_cats = {}
        sents = zip(sources, corrections, answers)
        pb = tqdm(sents, desc="Calculating errant metric", total=len(sources))
        for sent_id, sent in enumerate(pb):
            src = self.annotator.parse(sent[0])
            ref = self.annotator.parse(sent[1])
            hyp = self.annotator.parse(sent[2])
            # Align hyp and ref corrections and annotate errors
            hyp_edits = self.annotate_errors(src, hyp)
            ref_edits = self.annotate_errors(src, ref)
            # Process the edits for detection/correction based on args
            ProcessingArgs = namedtuple("ProcessingArgs",
                                        ["dt", "ds", "single", "multi", "filt", "cse"],
                                        defaults=[False, False, False, False, [], True])
            processing_args = ProcessingArgs()
            hyp_dict = process_edits(hyp_edits, processing_args)
            ref_dict = process_edits(ref_edits, processing_args)
            # Evaluate edits and get best TP, FP, FN hyp+ref combo.
            EvaluationArgs = namedtuple("EvaluationArgs",
                                        ["beta", "verbose"],
                                        defaults=[1.0, False])
            evaluation_args = EvaluationArgs()
            count_dict, cat_dict = evaluate_edits(
                hyp_dict, ref_dict, best_dict, sent_id, evaluation_args)
            # Merge these dicts with best_dict and best_cats
            best_dict += Counter(count_dict)  # corpus-level TP, FP, FN
            best_cats = merge_dict(best_cats, cat_dict)  # corpus-level errortype-wise TP, FP, FN
        cat_prf = {}
        for cat, values in best_cats.items():
            tp, fp, fn = values  # fp - extra corrections, fn - missed corrections
            p = float(tp) / (tp + fp) if tp + fp else 1.0
            r = float(tp) / (tp + fn) if tp + fn else 1.0
            f = (2 * p * r) / (p + r) if p + r else 0.0
            cat_prf[cat] = (p, r, f)

        for error_category in ["CASE", "PUNCT", "SPELL", "YO"]:
            if error_category not in cat_prf:
                cat_prf[error_category] = (1.0, 1.0, 1.0)

        return cat_prf
