"""Generic evaluator for spelling correction task."""
from __future__ import annotations

from typing import Iterable

from sage.evaluation.ruerrant_wrapper.scorer import RuErrantScorer
from sage.evaluation.evaluate import evaluation as calculate_word_metric


class Scorer:
    """
    Generic evaluator for spelling correction task.
    Specific evaluation function calls are implemented in the `score()` function.

    If it is not planned to use "errant" metric with a particular class instance,
    consider passing `load_errant=False` to optimize for time and memory.

    Attributes:
        errant: a RuErrantScorer instance (unless Scorer is initialized with load_errant=False).
    """

    def __init__(self, load_errant=True) -> None:
        if load_errant:
            self.errant = RuErrantScorer()
        else:
            self.errant = None

    def score(self, sources: Iterable[str], corrections: Iterable[str], answers: Iterable[str],
              metrics: Iterable[str]) -> dict[str, float]:
        """
        Evaluate spelling correction using the specified metrics.

        Args:
            sources (Iterable[str]): an iterable of source texts;
            corrections (Iterable[str]): an iterable of gold corrections for the source texts;
            answers (Iterable[str]): an iterable of evaluated corrections for the source texts;
            metrics (Iterable[str]): an iterable of metric to evaluate with;

        Returns:
            dict[str, float]: a dict mapping metric names to their values
            (the names may not be the same as in the `metrics` arg).
        """

        if metrics:
            for metric in metrics:
                if metric == "errant":
                    if self.errant is None:
                        raise AttributeError(
                            "You called for 'errant' metric which has not been loaded.",
                            "To use, reinitialize the Scorer with `load_errant=True`.")
                elif metric != "words":
                    raise ValueError(f"You provided a wrong metric name: `{metric}`.",
                                     "Available metrics are: ['errant', 'words'].")
        else:
            raise ValueError("The `metrics` argument must contain at least one metric name.")
        if isinstance(sources, str) or isinstance(corrections, str) or isinstance(answers, str):
            raise ValueError("The `sources`, `corrections`, and `answers` arguments",
                             "must be iterables of strings.")
        if "" in sources or "" in corrections or "" in answers:
            # probably too greedy condition (spacy in errant cannot parse empty strings)
            raise ValueError("All input strings must not be empty.")
        result = {}
        for metric in metrics:
            if metric == "errant" and self.errant is not None:
                metrics_by_cats = self.errant.evaluate(sources, corrections, answers)
                result_dict = {}
                metrics = ["Precision", "Recall", "F1"]
                for cat, values in metrics_by_cats.items():
                    for metric_name, metric_value in zip(metrics, values):
                        result_dict[f"{cat}_{metric_name}"] = round(float(metric_value) * 100, 2)
                result.update(result_dict)
            elif metric == "words":
                result.update(calculate_word_metric(sources, corrections, answers))
        return result
