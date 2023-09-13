"""General utils"""

import os
from typing import List, Dict, Any, Union

import numpy as np
import matplotlib.pyplot as plt


def _draw_distributions_with_spines(
        axes: plt.axes, row: int, column: int, actual_values: List[Any], reference_values: List[Any], title: str):
    """Draws two discrete distributions on a single plot."""

    axes[row][column].hist(reference_values, bins=20, ec="black", fc="red", label="Reference", density=True)
    axes[row][column].hist(actual_values, bins=20, ec="black", fc="green", label="Actual", alpha=0.7, density=True)
    axes[row][column].grid(b=True, color='grey', linestyle='-.', linewidth=0.5, alpha=0.3)
    for s in ['top', 'bottom', 'left', 'right']:
        axes[row][column].spines[s].set_visible(False)
    axes[row][column].set_title(title)
    axes[row][column].legend()


def draw_and_save_errors_distributions_comparison_charts(
        actual_typos_cnt: List[int],
        reference_typos_cnt: List[int],
        actual_stats: Dict[str, Dict[str, List]],
        reference_stats: Dict[str, Dict[str, List]],
        path_to_save: Union[str, os.PathLike],
):
    """Draws following distributions for reference and actual values:
        1. Number of errors per sentence;
        2. Types of errors;
        3. Relative positions of each type of errrors;
    ... and saves resulting charts to mentioned place `path_to_save`.

    Args:
        actual_typos_cnt (List[int]): number of errors actual;
        reference_typos_cnt (List[int]): number of errors reference;
        actual_stats (Dict[str, Dict[str, List]]): types of errors and their relative positions, actual;
        reference_stats (Dict[str, Dict[str, List]]): types of errors and their relative positions, reference;
        path_to_save (Union[str, os.PathLike]): where to save charts;
    """

    def _stats(d):
        tmp = {k: len(v["abs"]) for k, v in d.items()}
        total = sum(tmp.values())
        return tmp, total

    _, ax = plt.subplots(4, 2, figsize=(15, 15))
    plt.rcParams["figure.autolayout"] = True

    _draw_distributions_with_spines(
        ax, 0, 0, actual_typos_cnt, reference_typos_cnt, "Number of errors per sentence")
    _draw_distributions_with_spines(
        ax, 1, 0, actual_stats["insertion"]["rel"], reference_stats["insertion"]["rel"],
        "Relative positions of insertions")
    _draw_distributions_with_spines(
        ax, 1, 1, actual_stats["deletion"]["rel"], reference_stats["deletion"]["rel"],
        "Relative positions of deletions")
    _draw_distributions_with_spines(
        ax, 2, 0, actual_stats["substitution"]["rel"], reference_stats["substitution"]["rel"],
        "Relative positions of substitutions")
    _draw_distributions_with_spines(
        ax, 2, 1, actual_stats["transposition"]["rel"], reference_stats["transposition"]["rel"],
        "Relative positions of transpositions")
    _draw_distributions_with_spines(
        ax, 3, 0, actual_stats["missing_separator"]["rel"], reference_stats["missing_separator"]["rel"],
        "Relative positions of missing separators")
    _draw_distributions_with_spines(
        ax, 3, 1, actual_stats["extra_separator"]["rel"], reference_stats["extra_separator"]["rel"],
        "Relative positions of extra separators")

    width = 0.3
    d_ref, t_ref = _stats(reference_stats)
    d_act, t_act = _stats(actual_stats)
    d_ref = {k: v / (t_ref + 0.000001) for k, v in d_ref.items()}
    d_act = {k: v / (t_act + 0.000001) for k, v in d_act.items()}
    labels = list(d_ref.keys())
    ids = np.array(range(len(labels)))

    ax[0][1].barh(ids, list(d_act.values()), width, color='green', label='Actual')
    ax[0][1].barh(ids + width, list(d_ref.values()), width, color='red', label='Reference')
    ax[0][1].set(yticks=ids + width, yticklabels=labels)
    ax[0][1].grid(b=True, color='grey', linestyle='-.', linewidth=0.5, alpha=0.3)
    for s in ['top', 'bottom', 'left', 'right']:
        ax[0][1].spines[s].set_visible(False)
    ax[0][1].legend()
    ax[0][1].set_title("Types of errors")

    plt.savefig(path_to_save)
