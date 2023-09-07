"""
This module provides functionality to detect type and position of mistyping
given source sentence and corresponding corrected sentence.

"""

import re
import enum
import string
from typing import List, Dict

import numpy as np
from tqdm import tqdm


class TyposTypes(enum.Enum):
    """Available types of errors."""

    insertion = "Extra character"
    deletion = "Missing character"
    substitution = "Wrong character"
    transposition = "Two adjacent characters shuffled"
    missing_separator = "Missing gap"
    extra_separator = "Extra gap"


def make_levenshtein_table(source, correct, allow_transpositions=False, removal_cost=1.0, insertion_cost=1.0,
                           replace_cost=1.0, transposition_cost=1.0):
    first_length, second_length = len(source), len(correct)
    table = np.zeros(shape=(first_length + 1, second_length + 1), dtype=float)
    for i in range(1, second_length + 1):
        table[0][i] = i
    for i in range(1, first_length + 1):
        table[i][0] = i
    for i, first_word in enumerate(source, 1):
        for j, second_word in enumerate(correct, 1):
            if first_word == second_word:
                table[i][j] = table[i-1][j-1]
            else:
                table[i][j] = min((table[i-1][j-1] + replace_cost,
                                   table[i][j-1] + removal_cost,
                                   table[i-1][j] + insertion_cost))
                if (allow_transpositions and min(i, j) >= 2
                        and first_word == correct[j-2] and second_word == source[i-2]):
                    table[i][j] = min(table[i][j], table[i-2][j-2] + transposition_cost)
    return table


def process_group(source: str, correction: str, levenshtein_table: np.array) -> \
        [Dict[str, List[int]], Dict[str, List[int]], Dict[str, Dict[str, int]]]:

    """
    Identify type of mistyping and its position.
    
    Trace back table of Levenshtein distances and detect
    type of mistyping by the node from which it came from.
    
    Args:
        source (str): source sequence.
        correction (str): corrected sequence.
        levenshtein_table (np.array): table filled with distances between prefixes.
    
    Returns:
        d: Dict[str, List[int]], distribution of mistypings in sequence.
        d_src: Dict[str, List[int]], analogously, but positions are related to source sentence.
        confusion_matrix: Dict[str, Dict[str, int]], confusion matrix.
    
    """
    
    source = source.lower()
    correction = correction.lower()
    i = len(source)
    j = len(correction)
    d = {typo_type.name: [] for typo_type in TyposTypes}
    d_src = {typo_type.name: [] for typo_type in TyposTypes}
    confusion_matrix = {}

    while i > 0 and j > 0:
        # If characters are same
        if source[i-1] == correction[j-1]:
            i -= 1
            j -= 1
            
        # Substitution
        elif levenshtein_table[i][j] == levenshtein_table[i - 1][j - 1] + 1:
            d["substitution"].append(j - 1)
            d_src["substitution"].append(i - 1)
            correct_char = correction[j - 1]
            source_char = source[i - 1]
            if correct_char in confusion_matrix:
                if source_char in confusion_matrix[correct_char]:
                    confusion_matrix[correct_char][source_char] += 1
                else:
                    confusion_matrix[correct_char][source_char] = 1
            else:
                confusion_matrix[correct_char] = {source_char: 1}
                
            j -= 1
            i -= 1

        # Insertion
        elif levenshtein_table[i][j] == levenshtein_table[i - 1][j] + 1:
            if source[i - 1] == " ":
                d["extra_separator"].append(j)
                d_src["extra_separator"].append(i - 1)
            else:
                d["insertion"].append(j)
                d_src["insertion"].append(i - 1)
            i -= 1

        # Deletion
        elif levenshtein_table[i][j] == levenshtein_table[i][j - 1] + 1:
            if j < len(source) and correction[j - 1] == " ":
                d["missing_separator"].append(j - 1)
                d_src["missing_separator"].append(i)
            else:
                d["deletion"].append(j - 1)
                d_src["deletion"].append(i)
            j -= 1

        # Transposition
        elif min(i, j) >= 2 and levenshtein_table[i][j] == levenshtein_table[i-2][j-2] + 1:
            d["transposition"].append(j - 2)
            d_src["transposition"].append(i - 2)
            i -= 2
            j -= 2

    if i > 0:
        d["insertion"].extend([0] * i)
        d_src["insertion"].extend(list(range(i)))
    if j > 0:
        d["deletion"].extend(list(range(j)))
        d_src["deletion"].extend([0] * j)

    return d, d_src, confusion_matrix


def process_mistypings(
    src: List[str], corr: List[str],
) -> [Dict[str, Dict[str, List[float]]], Dict[str, Dict[str, int]], List[int]]:
        
    """
    Processes allignment groups and outputs mistypings distribution.
    We have following classification of mistypings that goes like this:
        1. insertion ("туберкПулёз" -> "туберкулёз")
        2. deletion ("тубркулёз" -> "туберкулёз")
        3. substitution ("тубИркулёз" -> "туберкулёз")
        4. transposition ("тубРЕкулёз" -> "туберкулёз")
        5. extra_separator ("туберкулёз" -> "тубе ркулёз")
        6. missing_separator ("острый туберкулёз" -> "острыйтуберкулёз")
    
    Args:
        src (List[str]): original sequences with mistypings.
        corr (List[str]): corrected sequences.
    
    Returns:
        global_stats: Dict[str, Dict[str, List[float]]], distributions of positions
                      across the whole corpus.
        global_cm: Dict[str, Dict[str, int]], confusion matrix on the whole corpus.
        mistypings_cnt: List[int], number of mistypings in each sentence.
        
    """
    global_stats = {typo_type.name: {"abs": [], "rel": []} for typo_type in TyposTypes}
    global_cm = {}
    mistypings_cnt = []
    pattern = string.punctuation.replace("-", "")
    
    for source, correction in tqdm(zip(src, corr)):
        source = re.sub(r"[{}]".format(pattern), "", source.lower().strip())
        correction = re.sub(r"[{}]".format(pattern), "", correction.lower().strip())
        
        dp = make_levenshtein_table(source, correction, allow_transpositions=True)
        # We gather distributions from source sentences, NOT from corrections
        _, local_stats, local_cm = process_group(source, correction, dp)
        
        mistypings_cnt.append(sum((len(v) for _, v in local_stats.items())))
        for typo, positions in local_stats.items():
            global_stats[typo]["abs"].extend(positions)
            global_stats[typo]["rel"].extend([pos / len(source) for pos in positions])
        for correct_char, candidates in local_cm.items():
            if correct_char not in global_cm:
                global_cm[correct_char] = {}
            for candidate, cnt in candidates.items():
                if candidate in global_cm[correct_char]:
                    global_cm[correct_char][candidate] += cnt
                else:
                    global_cm[correct_char][candidate] = cnt
        
    return global_stats, global_cm, mistypings_cnt
