"""Language-related utils"""

from collections import namedtuple

AVAILABLE_LANG_CODES = ["rus", "eng"]


class InsertionOptions(namedtuple("insertion_options", AVAILABLE_LANG_CODES)):
    pass


class SubstitutionOptions(namedtuple("substitution_options", AVAILABLE_LANG_CODES)):
    pass


INSERTION_OPTIONS = InsertionOptions(
    rus=list("абвгдеёжзийклмнопрстуфхцчшщъыьэюя"),
    eng=list("abcdefghijklmnopqrstuvwxyz")
)

SUBSTITUTION_OPTIONS = SubstitutionOptions(
    rus=list("абвгдеёжзийклмнопрстуфхцчшщъыьэюя "),
    eng=list("abcdefghijklmnopqrstuvwxyz ")
)
