from __future__ import annotations

import itertools
import re
from string import punctuation

import Levenshtein
from errant.alignment import Alignment
from errant.edit import Edit


def get_rule_edits(alignment: Alignment) -> list[Edit]:
    """Groups word-level alignment according to merging rules."""
    edits = []
    # Split alignment into groups
    alignment_groups = group_alignment(alignment, "new")
    for op, group in alignment_groups:
        group = list(group)
        # Ignore M
        if op == "M":
            continue
        # T is always split
        if op == "T":
            for seq in group:
                edits.append(Edit(alignment.orig, alignment.cor, seq[1:]))
        # Process D, I and S subsequence
        else:
            processed = process_seq(group, alignment)
            # Turn the processed sequence into edits
            for seq in processed:
                edits.append(Edit(alignment.orig, alignment.cor, seq[1:]))
    return edits


def group_alignment(alignment: Alignment, mode: str = "default") -> list[tuple[str, list[tuple]]]:
    """
    Does initial alignment grouping:
    1. Make groups of MDM, MIM od MSM.
    2. In remaining operations, make groups of Ms, groups of Ts, and D/I/Ss.
    Do not group what was on the sides of M[DIS]M: SSMDMS -> [SS, MDM, S], not [MDM, SSS].
    3. Sort groups by the order in which they appear in the alignment.
    """
    if mode == "new":
        op_groups = []
        # Format operation types sequence as string to use regex sequence search
        all_ops_seq = "".join([op[0][0] for op in alignment.align_seq])
        # Find M[DIS]M groups and merge (need them to detect hyphen vs. space spelling)
        ungrouped_ids = list(range(len(alignment.align_seq)))
        for match in re.finditer("M[DIS]M", all_ops_seq):
            start, end = match.start(), match.end()
            op_groups.append(("MSM", alignment.align_seq[start:end]))
            for idx in range(start, end):
                ungrouped_ids.remove(idx)
        # Group remaining operations by default rules (groups of M, T and rest)
        if ungrouped_ids:
            def get_group_type(operation):
                return operation if operation in {"M", "T"} else "DIS"
            curr_group = [alignment.align_seq[ungrouped_ids[0]]]
            last_oper_type = get_group_type(curr_group[0][0][0])
            for i, idx in enumerate(ungrouped_ids[1:], start=1):
                operation = alignment.align_seq[idx]
                oper_type = get_group_type(operation[0][0])
                if (oper_type == last_oper_type and
                        (idx - ungrouped_ids[i-1] == 1 or oper_type in {"M", "T"})):
                    curr_group.append(operation)
                else:
                    op_groups.append((last_oper_type, curr_group))
                    curr_group = [operation]
                last_oper_type = oper_type
            if curr_group:
                op_groups.append((last_oper_type, curr_group))
        # Sort groups by the start id of the first group entry
        op_groups = sorted(op_groups, key=lambda x: x[1][0][1])
    else:
        grouped = itertools.groupby(alignment.align_seq,
                                    lambda x: x[0][0] if x[0][0] in {"M", "T"} else False)
        op_groups = [(op, list(group)) for op, group in grouped]
    return op_groups


def process_seq(seq: list[tuple], alignment: Alignment) -> list[tuple]:
    """Applies merging rules to previously formed alignment groups (`seq`)."""
    # Return single alignments
    if len(seq) <= 1:
        return seq
    # Get the ops for the whole sequence
    ops = [op[0] for op in seq]
    # Merge all D xor I ops. (95% of human multi-token edits contain S).
    if set(ops) == {"D"} or set(ops) == {"I"}:
        return merge_edits(seq)

    # Get indices of all start-end combinations in the seq: 012 = 01, 02, 12
    combos = list(itertools.combinations(range(0, len(seq)), 2))
    # Sort them starting with largest spans first
    combos.sort(key=lambda x: x[1] - x[0], reverse=True)
    # Loop through combos
    for start, end in combos:
        # Ignore ranges that do NOT contain a substitution, deletion or insertion.
        if not any(type_ in ops[start:end + 1] for type_ in ["D", "I", "S"]):
            continue
        # Get the tokens in orig and cor.
        o = alignment.orig[seq[start][1]:seq[end][2]]
        c = alignment.cor[seq[start][3]:seq[end][4]]
        if ops[start:end + 1] in [["M", "D", "M"], ["M", "I", "M"], ["M", "S", "M"]]:
            # merge hyphens
            if (o[start + 1].text == "-" or c[start + 1].text == "-") and len(o) != len(c):
                return (process_seq(seq[:start], alignment)
                        + merge_edits(seq[start:end + 1])
                        + process_seq(seq[end + 1:], alignment))
            # if it is not a hyphen-space edit, return only punct edit
            return seq[start + 1: end]
        # Merge possessive suffixes: [friends -> friend 's]
        if o[-1].tag_ == "POS" or c[-1].tag_ == "POS":
            return (process_seq(seq[:end - 1], alignment)
                    + merge_edits(seq[end - 1:end + 1])
                    + process_seq(seq[end + 1:], alignment))
        # Case changes
        if o[-1].lower == c[-1].lower:
            # Merge first token I or D: [Cat -> The big cat]
            if (start == 0 and
                    (len(o) == 1 and c[0].text[0].isupper()) or
                    (len(c) == 1 and o[0].text[0].isupper())):
                return (merge_edits(seq[start:end + 1])
                        + process_seq(seq[end + 1:], alignment))
            # Merge with previous punctuation: [, we -> . We], [we -> . We]
            if (len(o) > 1 and is_punct(o[-2])) or \
                    (len(c) > 1 and is_punct(c[-2])):
                return (process_seq(seq[:end - 1], alignment)
                        + merge_edits(seq[end - 1:end + 1])
                        + process_seq(seq[end + 1:], alignment))
        # Merge whitespace/hyphens: [acat -> a cat], [sub - way -> subway]
        s_str = re.sub("['-]", "", "".join([tok.lower_ for tok in o]))
        t_str = re.sub("['-]", "", "".join([tok.lower_ for tok in c]))
        if s_str == t_str:
            return (process_seq(seq[:start], alignment)
                    + merge_edits(seq[start:end + 1])
                    + process_seq(seq[end + 1:], alignment))
        # Merge same POS or auxiliary/infinitive/phrasal verbs:
        # [to eat -> eating], [watch -> look at]
        pos_set = set([tok.pos for tok in o] + [tok.pos for tok in c])
        if len(o) != len(c) and (len(pos_set) == 1 or pos_set.issubset({"AUX", "PART", "VERB"})):
            return (process_seq(seq[:start], alignment)
                    + merge_edits(seq[start:end + 1])
                    + process_seq(seq[end + 1:], alignment))
        # Split rules take effect when we get to smallest chunks
        if end - start < 2:
            # Split adjacent substitutions
            if len(o) == len(c) == 2:
                return (process_seq(seq[:start + 1], alignment)
                        + process_seq(seq[start + 1:], alignment))
            # Split similar substitutions at sequence boundaries
            if ((ops[start] == "S" and char_cost(o[0].text, c[0].text) > 0.75) or
                    (ops[end] == "S" and char_cost(o[-1].text, c[-1].text) > 0.75)):
                return (process_seq(seq[:start + 1], alignment)
                        + process_seq(seq[start + 1:], alignment))
            # Split final determiners
            if (end == len(seq) - 1 and
                    ((ops[-1] in {"D", "S"} and o[-1].pos == "DET") or
                     (ops[-1] in {"I", "S"} and c[-1].pos == "DET"))):
                return process_seq(seq[:-1], alignment) + [seq[-1]]
    return seq


def is_punct(token) -> bool:
    return token.text in punctuation


def char_cost(a: str, b: str) -> float:
    """Calculate the cost of character alignment; i.e. char similarity."""

    return Levenshtein.ratio(a, b)


def merge_edits(seq: list[tuple]) -> list[tuple]:
    """Merge the input alignment sequence to a single edit span."""

    if seq:
        return [("X", seq[0][1], seq[-1][2], seq[0][3], seq[-1][4])]
    return seq
