"""This module contains evaluation utils used when assessing spelling correction performance.

The script is taken from https://www.dialog-21.ru/media/3427/sorokinaaetal.pdf by Sorokin et al.
with minor changes.
"""

import copy
import re
from typing import List, Dict
from collections import defaultdict

import numpy as np
import timeout_decorator


class TimeOutValidation(Exception):
    pass


def extract_words(line, make_lower=True, split_by_dots=False):
    line = line.strip()
    if split_by_dots:
        sents = re.split(r"[^а-яёa-z0-9\-.:/,]+", line, flags=re.I)
    else:
        sents = line.split()
    words = []
    for word in sents:
        if make_lower:
            word = word.lower().replace('ё', 'е')
        else:
            word = word.replace('ё', 'е')
            word = word.replace('Ё', 'Е')
        i = len(word) - 1
        while i >= 0 and not (word[i].isalpha() or word[i].isdigit()):
            i -= 1
        if i < 0:
            continue
        word = word[:(i+1)]
        while len(word) > 0 and not (word[0].isalpha() or word[0].isdigit()):
            word = word[1:]
        if word != "":
            words.append(word)
    return words


def levenstein_dist(source, correct, allow_transpositions=False,
                    removal_cost=1.0, insertion_cost=1.0, replace_cost=1.0, transposition_cost=1.0):
    table, _ = make_levenstein_table(source, correct, allow_transpositions=allow_transpositions,
                                  removal_cost=removal_cost, insertion_cost=insertion_cost,
                                  transposition_cost=transposition_cost)
    return table[-1][-1]


def make_levenstein_table(source, correct, allow_transpositions=False,
        removal_cost=1.0, insertion_cost=1.0, replace_cost=1.0, transposition_cost=1.0):
    """
    Builds dynamic Levenshtein table and a list of backward links to maintain alignment.

    Args:
        source (List[str]): original sentence;
        correct (List[str]): corrected sentence;
        allow_transpositions (Optional[bool]): whether to allow transpositions of adjacent characters, defaults to False;
        removal_cost (Optional[float]): cost of removal, defaults to 1.;
        insertion_cost (Optional[float]): cost of insertion, defaults to 1.;
        replace_cost (Optional[float]): cost of replacement, defaults to 1.;
        transposition_cost (Optional[float]): cost of transposition, defaults to 1.;

    Return:
        table (np.array): Levenshtein table, table[i][j] = d(source[:i], correct[:j]);
        backtraces (np.array): table of backward links;
    """
    first_length, second_length = len(source), len(correct)
    table = np.zeros(shape=(first_length + 1, second_length + 1), dtype=float)
    backtraces = [([None] * (second_length + 1)) for _ in range(first_length + 1)]
    for i in range(1, second_length + 1):
        table[0][i] = i
        backtraces[0][i] = [(0, i-1)]
    for i in range(1, first_length + 1):
        table[i][0] = i
        backtraces[i][0] = [(i-1, 0)]
    for i, first_word in enumerate(source, 1):
        for j, second_word in enumerate(correct, 1):
            if first_word == second_word:
                table[i][j] = table[i-1][j-1]
                backtraces[i][j] = [(i-1, j-1)]
            else:
                table[i][j] = min((table[i-1][j-1] + replace_cost,
                                   table[i][j-1] + removal_cost,
                                   table[i-1][j] + insertion_cost))
                if (allow_transpositions and min(i, j) >= 2
                        and first_word == correct[j-2] and second_word == source[i-2]):
                    table[i][j] = min(table[i][j], table[i-2][j-2] + transposition_cost)
                curr_backtraces = []
                if table[i-1][j-1] + replace_cost == table[i][j]:
                    curr_backtraces.append((i-1, j-1))
                if table[i][j-1] + removal_cost == table[i][j]:
                    curr_backtraces.append((i, j-1))
                if table[i-1][j] + insertion_cost == table[i][j]:
                    curr_backtraces.append((i-1, j))
                if (allow_transpositions and min(i, j) >= 2
                    and first_word == correct[j-2] and second_word == source[i-2]
                        and table[i][j] == table[i-2][j-2] + transposition_cost):
                    curr_backtraces.append((i-2, j-2))
                backtraces[i][j] = copy.copy(curr_backtraces)
    return table, backtraces


def extract_best_alignment(backtraces):
    """
    Extracts alignments from backward links table.

    Args:
        backtraces (np.array): table of backward links;
    Return:
        best_paths (List[List[Any]]): paths from (0, 0) to (m, n) in backtraces.
    """
    m, n = len(backtraces) - 1, len(backtraces[0]) - 1
    used_vertexes = {(m, n)}
    reverse_path_graph = defaultdict(list)
    vertexes_queue = [(m, n)]
    # builds graph of best paths in table
    while len(vertexes_queue) > 0:
        i, j = vertex = vertexes_queue.pop(0)
        if i > 0 or j > 0:
            for new_vertex in backtraces[i][j]:
                reverse_path_graph[new_vertex].append(vertex)
                if new_vertex not in used_vertexes:
                    vertexes_queue.append(new_vertex)
                    used_vertexes.add(new_vertex)
    # traverse paths back
    best_paths = []
    current_path = [(0, 0)]
    last_indexes, neighbor_vertexes_list = [], []
    while len(current_path) > 0:
        if current_path[-1] != (m, n):
            children = reverse_path_graph[current_path[-1]]
            if len(children) > 0:
                current_path.append(children[0])
                last_indexes.append(0)
                neighbor_vertexes_list.append(children)
                continue
        else:
            best_paths.append(copy.copy(current_path))
        while len(last_indexes) > 0 and last_indexes[-1] == len(neighbor_vertexes_list[-1]) - 1:
            current_path.pop()
            last_indexes.pop()
            neighbor_vertexes_list.pop()
        if len(last_indexes) == 0:
            break
        last_indexes[-1] += 1
        current_path[-1] = neighbor_vertexes_list[-1][last_indexes[-1]]
    return best_paths


def extract_basic_alignment_paths(paths_in_alignments, source, correct):
    """
    Extracts identical substitutions from paths in Levenshtein table.

    Args:
        paths_in_alignments (List[List[Any]]):  paths from (0, 0) to (m, n) in table.
        source (List[str]): original sentence;
        correct (List[str]): corrected sentence;

    Return:
        answer (List[Any]): identical substitutions;
    """
    m, n = len(source), len(correct)
    are_symbols_equal = np.zeros(dtype=bool, shape=(m, n))
    for i, a in enumerate(source):
        for j, b in enumerate(correct):
            are_symbols_equal[i][j] = (a == b)
    answer = set()
    for path in paths_in_alignments:
        answer.add(tuple(elem for elem in path[1:] if (elem[0] > 0 and elem[1] > 0
                                                       and are_symbols_equal[elem[0]-1][elem[1]-1])))
    return list(answer)


def extract_levenstein_alignments(source, correct, replace_cost=1.0):
    """
    Finds positions of identical substitutions in source and correction.

    Args:
        source (List[str]): original sentence;
        correct (List[str]): corrected sentence;

    Return:
        basic_alignment_paths (List[List[tuple(int)]]): identical substitutions
    """
    table, backtraces = make_levenstein_table(source, correct, replace_cost=replace_cost)
    paths_in_alignments = extract_best_alignment(backtraces)
    basic_alignment_paths = extract_basic_alignment_paths(paths_in_alignments, source, correct)
    return basic_alignment_paths


def get_partition_indexes(first, second):
    """
    Builds alingment between groups found in source sentence and corresponding correction.

    Indexes i and j indicate ending of a group if last characters of words first[i] and second[j]
    appear in a path in a Levenshtein table between " ".join(first) and " ".join(second).

    Args:
        first (List[str]): list of original words;
        second (List[str]): list of corrected words;

    Return:
        answer (List[tuple(int)]): alignment between first and second.
    """
    m, n = len(first), len(second)
    answer = [(0, 0)]
    if m <= 1 or n <= 1:
        answer += [(m, n)]
    else:
        levenstein_table, backtraces = make_levenstein_table(" ".join(first), " ".join(second))
        best_paths_in_table = extract_best_alignment(backtraces)
        good_partitions, other_partitions = set(), set()
        word_ends = [0], [0]
        last = -1
        for i, word in enumerate(first):
            last = last + len(word) + 1
            word_ends[0].append(last)
        last = -1
        for i, word in enumerate(second):
            last = last + len(word) + 1
            word_ends[1].append(last)
        for path in best_paths_in_table:
            current_indexes = [(0, 0)]
            first_pos, second_pos = 0, 0
            is_partition_good = True
            for i, j in path[1:]:
                if i > word_ends[0][first_pos]:
                    first_pos += 1
                if j > word_ends[1][second_pos]:
                    second_pos += 1
                if i == word_ends[0][first_pos] and j == word_ends[1][second_pos]:
                    if first_pos > current_indexes[-1][0] and second_pos > current_indexes[-1][1]:
                        current_indexes.append((first_pos, second_pos))
                        if first_pos < len(first):
                            first_pos += 1
                        if second_pos < len(second):
                            second_pos += 1
                    else:
                        is_partition_good = False
            if current_indexes[-1] == (m, n):
                if is_partition_good:
                    good_partitions.add(tuple(current_indexes))
                else:
                    other_partitions.add(tuple(current_indexes))
            else:
                current_indexes = current_indexes[:-1] + [(m, n)]
                other_partitions.add(tuple(current_indexes))
        if len(good_partitions) >= 1:
            answer = list(good_partitions)[0]
        else:
            answer = list(other_partitions)[0]
    return answer


@timeout_decorator.timeout(10, timeout_exception = TimeOutValidation)
def align_sents(source, correct, return_only_different=False, replace_cost=1.0,
                partition_intermediate=True, groups_in_source=None):
    """
    Finds positions of groups' endings in source sentence and corresponding correction.

    Args:
        source (List[str]): original sentence;
        correct (List[str]): corrected sentence;
        return_only_different (bool): whether to return only indexes of non-identical corrections;
        replace_cost (float): cost of non-identical replacements;

    Return:
        answer (List[tuple(tuple(int))]): groups, if answer[i] == ((i, j), (k, l)), then source[i:j] and correct[k:l]
                                          resemble the same group.
    """
    if groups_in_source is None:
        groups_in_source = []
    alignments = extract_levenstein_alignments(source, correct, replace_cost=replace_cost)
    m, n = len(source), len(correct)
    prev = 0, 0
    answer = []
    for i, j in alignments[0]:
        if i > prev[0] + 1 or j > prev[1] + 1:
            if partition_intermediate:
                partition_indexes =\
                    get_partition_indexes(source[prev[0]: i-1], correct[prev[1]: j-1])
                if partition_indexes is not None:
                    for pos, (f, s) in enumerate(partition_indexes[:-1]):
                        answer.append(((prev[0] + f, prev[0] + partition_indexes[pos+1][0]),
                                       (prev[1] + s, prev[1] + partition_indexes[pos+1][1])))
                else:
                    answer.append(((prev[0], i-1), (prev[1], j-1)))
            else:
                answer.append(((prev[0], i-1), (prev[1], j-1)))
        answer.append(((i-1, i), (j-1, j)))
        prev = i, j
    if m > prev[0] or n > prev[1]:
        if partition_intermediate:
            partition_indexes =\
                    get_partition_indexes(source[prev[0]: m], correct[prev[1]: n])
            if partition_indexes is not None:
                for pos, (f, s) in enumerate(partition_indexes[:-1]):
                        answer.append(((prev[0] + f, prev[0] + partition_indexes[pos+1][0]),
                                       (prev[1] + s, prev[1] + partition_indexes[pos+1][1])))
            else:
                answer.append(((prev[0], m), (prev[1], n)))
        else:
            answer.append(((prev[0], m), (prev[1], n)))
    positions_in_answer = []
    indexes_in_source = [elem[0] for elem in answer]
    end_in_answer = -1
    for pos, (i_ref, j_ref) in enumerate(groups_in_source):
        start_in_answer = end_in_answer + 1
        while (start_in_answer < len(indexes_in_source) and indexes_in_source[start_in_answer][0] < i_ref):
            start_in_answer += 1
        if start_in_answer == len(indexes_in_source):
            break
        i, j = indexes_in_source[start_in_answer]
        end_in_answer = start_in_answer
        if i == i_ref:
            while (end_in_answer < len(indexes_in_source) and indexes_in_source[end_in_answer][1] < j_ref):
                end_in_answer += 1
            if end_in_answer == len(indexes_in_source):
                break
            if indexes_in_source[end_in_answer][1] == j_ref:
                positions_in_answer.append((start_in_answer, end_in_answer))
    prev_end = -1
    new_answer = []
    for start_in_answer, end_in_answer in positions_in_answer:
        new_answer.extend(answer[prev_end+1: start_in_answer])
        new_answer.append(((answer[start_in_answer][0][0], answer[end_in_answer][0][1]),
                           (answer[start_in_answer][1][0], answer[end_in_answer][1][1])))
        prev_end = end_in_answer
    new_answer.extend(answer[prev_end+1:])
    answer = new_answer
    if return_only_different:
        answer = [((i, j), (k, l)) for ((i, j), (k, l)) in answer if source[i:j] != correct[k:l]]
    return answer


def make_corrections_data(source_sents, correct_sents, answer_sents):
    etalon_corrections = dict()
    answer_corrections = dict()
    for num, (source, correct, answer) in\
            enumerate(zip(source_sents, correct_sents, answer_sents)):
        try:
            print(num)
            correct_indexes = align_sents(source, correct, return_only_different=True, replace_cost=1.9)
            src_indexes = align_sents(source, answer, return_only_different=True, replace_cost=1.9,
                                  groups_in_source=[elem[0] for elem in correct_indexes])
        
            for ((i, j), (k, l)) in correct_indexes:
                etalon_corrections[(num, i, j)] = tuple(correct[k:l])
            for ((i, j), (k, l)) in src_indexes:
                answer_corrections[(num, i, j)] = tuple(answer[k:l])
        except TimeOutValidation:
            print("Skipping {} line, because operation timed out...".format(num))
            
    return etalon_corrections, answer_corrections


def measure_quality(etalon_corrections, answer_corrections):
    TP = 0
    for triple, answer_correction in answer_corrections.items():
        etalon_correction = etalon_corrections.get(triple)
        if etalon_correction == answer_correction:
            TP += 1
    precision = TP / len(answer_corrections)
    recall = TP / len(etalon_corrections)
    f_measure = 2 * precision * recall / (precision + recall)
    return TP, precision, recall, f_measure


def output_differences(diff_file, source_sents, correct_sents, answer_sents,
                       etalon_corrections, answer_corrections):
    false_positives = defaultdict(list)
    false_negatives = defaultdict(list)
    miscorrections = defaultdict(list)
    for (num, i, j), answer_correction in answer_corrections.items():
        etalon_correction = etalon_corrections.get((num, i, j))
        if etalon_correction is None:
            false_positives[num].append(((i, j), answer_correction))
        elif etalon_correction != answer_correction:
            miscorrections[num].append(((i, j), answer_correction, etalon_correction))
    for (num, i, j), etalon_correction in etalon_corrections.items():
        answer_correction = answer_corrections.get((num, i, j))
        if answer_correction is None:
            false_negatives[num].append(((i, j), etalon_correction))
    with open(diff_file, "w", encoding="utf8") as fout:
        width = 24
        for num, sent in enumerate(source_sents):
            current_false_positives = false_positives[num]
            current_false_negatives = false_negatives[num]
            current_miscorrections = miscorrections[num]
            if (len(current_false_positives) == 0 and len(current_false_negatives) == 0 and
                    len(current_miscorrections) == 0):
                continue
            fout.write("{0}\n{1}\n{2}\n".format(
                " ".join(sent), " ".join(answer_sents[num]), " ".join(correct_sents[num])))
            for (i, j), answer_correction in current_false_positives:
                fout.write("{0:<{width}}{1:<{width}}{2:<{width}}\n".format(" ".join(sent[i:j]),
                    " ".join(answer_correction), " ".join(sent[i:j]), width=width))
            for (i, j), etalon_correction in current_false_negatives:
                fout.write("{0:<{width}}{1:<{width}}{2:<{width}}\n".format(" ".join(sent[i:j]),
                    " ".join(sent[i:j]), " ".join(etalon_correction), width=width))
            for (i, j), answer_correction, etalon_correction in current_miscorrections:
                fout.write("{0:<{width}}{1:<{width}}{2:<{width}}\n".format(" ".join(sent[i:j]),
                    " ".join(answer_correction), " ".join(etalon_correction), width=width))
            fout.write("\n")
    return


def test(task=1):
    if task == 0:
        first_sent = ['фотка', 'классная', 'кстате', 'хоть', 'и', 'не', 'по', 'теме']
        second_sent = ['фотка', 'классная', 'кстати', 'хотя', 'не', 'по', 'теме']
        align_sents(first_sent, second_sent, replace_cost=1.9)
    elif task == 1:
        first, second = 'жж', 'ж'
        align_sents(first, second)


def evaluation(
    sources: List[str],
    corrections: List[str],
    answers: List[str],
    to_output_differences: bool = False,
    path_to_diff: str = "diff.txt",
) -> Dict[str, float]:
    
    # Substitute empty strings
    for i, ans in enumerate(answers):
        if len(ans.strip(" ")) == 0:
            print("empty string")
            answers[i] = sources[i]
            
    source_sents = [extract_words(line.strip().strip('\ufeff')) for line in sources]
    correct_sents = [extract_words(line.strip().strip('\ufeff')) for line in corrections]
    answer_sents = [extract_words(line.strip().strip('\ufeff')) for line in answers]
    
    etalon_corrections, answer_corrections = make_corrections_data(source_sents, correct_sents, answer_sents)

    TP, precision, recall, f_measure = measure_quality(etalon_corrections, answer_corrections)

    if to_output_differences:
        output_differences(path_to_diff, source_sents, correct_sents, answer_sents,
                           etalon_corrections, answer_corrections)

    return {
            "Precision": round(precision * 100, 2),
            "Recall": round(recall * 100, 2),
            "F1": round(f_measure * 100, 2)
        }
    