from __future__ import annotations

from collections import defaultdict
from string import punctuation

import Levenshtein
from errant.edit import Edit


def edit_to_tuple(edit: Edit, idx: int = 0) -> tuple[int, int, str, str, int]:
    cor_toks_str = " ".join([tok.text for tok in edit.c_toks])
    return [edit.o_start, edit.o_end, edit.type, cor_toks_str, idx]


def classify(edit: Edit) -> list[Edit]:
    """Classifies an Edit via updating its `type` attribute."""
    # Insertion and deletion
    if ((not edit.o_toks and edit.c_toks) or (edit.o_toks and not edit.c_toks)):
        error_cats = get_one_sided_type(edit.o_toks, edit.c_toks)
    elif edit.o_toks != edit.c_toks:
        error_cats = get_two_sided_type(edit.o_toks, edit.c_toks)
    else:
        error_cats = {"NA": edit.c_toks[0].text}
    new_edit_list = []
    if error_cats:
        for error_cat, correct_str in error_cats.items():
            edit.type = error_cat
            edit_tuple = edit_to_tuple(edit)
            edit_tuple[3] = correct_str
            new_edit_list.append(edit_tuple)
    return new_edit_list


def get_edit_info(toks):
    pos = []
    dep = []
    morph = dict()
    for tok in toks:
        pos.append(tok.tag_)
        dep.append(tok.dep_)
        morphs = str(tok.morph).split('|')
        for m in morphs:
            if len(m.strip()):
                k, v = m.strip().split('=')
                morph[k] = v
    return pos, dep, morph


def get_one_sided_type(o_toks, c_toks):
    """Classifies a zero-to-one or one-to-zero error based on a token list."""
    pos_list, _, _ = get_edit_info(o_toks if o_toks else c_toks)
    if "PUNCT" in pos_list:
        return {"PUNCT": c_toks[0].text if c_toks else ""}
    return {"SPELL": c_toks[0].text if c_toks else ""}


def get_two_sided_type(o_toks, c_toks) -> dict[str, str]:
    """Classifies a one-to-one or one-to-many or many-to-one error based on token lists."""
    # one-to-one cases
    if len(o_toks) == len(c_toks) == 1:
        if o_toks[0].text in punctuation + "..." and c_toks[0].text in punctuation + "...":
            return {"PUNCT": c_toks[0].text}
        source_w, correct_w = o_toks[0].text, c_toks[0].text
        if source_w != correct_w:
            # if both string are lowercase or both are uppercase,
            # and there is no "ё" in both, then it may be only "SPELL" error type
            if (((source_w.islower() and correct_w.islower()) or
                (source_w.isupper() and correct_w.isupper())) and
                    "ё" not in source_w + correct_w):
                return {"SPELL": correct_w}
            # edits with multiple errors (e.g. SPELL + CASE)
            # Step 1. Make char-level Levenstein table
            char_edits = Levenshtein.editops(source_w, correct_w)
            # Step 2. Classify operations (CASE, YO, SPELL)
            edits_classified = classify_char_edits(char_edits, source_w, correct_w)
            # Step 3. Combine the same-typed errors into minimal string pairs
            separated_edits = get_edit_strings(source_w, correct_w, edits_classified)
            return separated_edits
    # one-to-many and many-to-one cases
    joint_corr_str = " ".join([tok.text for tok in c_toks])
    joint_corr_str = joint_corr_str.replace("- ", "-").replace(" -", "-")
    return {"SPELL": joint_corr_str}


def classify_char_edits(char_edits, source_w, correct_w):
    """Classifies char-level Levenstein operations into SPELL, YO and CASE."""
    edits_classified = []
    for edit in char_edits:
        if edit[0] == "replace":
            if "ё" in [source_w[edit[1]], correct_w[edit[2]]]:
                edits_classified.append((*edit, "YO"))
            elif source_w[edit[1]].lower() == correct_w[edit[2]].lower():
                edits_classified.append((*edit, "CASE"))
            else:
                if (
                    (source_w[edit[1]].islower() and correct_w[edit[2]].isupper()) or
                    (source_w[edit[1]].isupper() and correct_w[edit[2]].islower())
                ):
                    edits_classified.append((*edit, "CASE"))
                edits_classified.append((*edit, "SPELL"))
        else:
            edits_classified.append((*edit, "SPELL"))
    return edits_classified


def get_edit_strings(source: str, correction: str,
                     edits_classified: list[tuple]) -> dict[str, str]:
    """
    Applies classified (SPELL, YO and CASE) char operations to source word separately.
    Returns a dict mapping error type to source string with corrections of this type only.
    """
    separated_edits = defaultdict(lambda: source)
    shift = 0  # char position shift to consider on deletions and insertions
    for edit in edits_classified:
        edit_type = edit[3]
        curr_src = separated_edits[edit_type]
        if edit_type == "CASE":  # SOURCE letter spelled in CORRECTION case
            if correction[edit[2]].isupper():
                correction_char = source[edit[1]].upper()
            else:
                correction_char = source[edit[1]].lower()
        else:
            if edit[0] == "delete":
                correction_char = ""
            elif edit[0] == "insert":
                correction_char = correction[edit[2]]
            elif source[edit[1]].isupper():
                correction_char = correction[edit[2]].upper()
            else:
                correction_char = correction[edit[2]].lower()
        if edit[0] == "replace":
            separated_edits[edit_type] = curr_src[:edit[1] + shift] + correction_char + \
                curr_src[edit[1]+shift + 1:]
        elif edit[0] == "delete":
            separated_edits[edit_type] = curr_src[:edit[1] + shift] + \
                curr_src[edit[1]+shift + 1:]
            shift -= 1
        elif edit[0] == "insert":
            separated_edits[edit_type] = curr_src[:edit[1] + shift] + correction_char + \
                curr_src[edit[1]+shift:]
            shift += 1
    return dict(separated_edits)
