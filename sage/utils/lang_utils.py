"""Language-related utils"""

from collections import namedtuple

AVAILABLE_LANG_CODES = ["ru", "en"]


class InsertionOptions(namedtuple("insertion_options", AVAILABLE_LANG_CODES)):
    pass


class SubstitutionOptions(namedtuple("substitution_options", AVAILABLE_LANG_CODES)):
    pass


INSERTION_OPTIONS = InsertionOptions(
    ru=list("абвгдеёжзийклмнопрстуфхцчшщъыьэюя"),
    en=list("abcdefghijklmnopqrstuvwxyz")
)

SUBSTITUTION_OPTIONS = SubstitutionOptions(
    ru=list("абвгдеёжзийклмнопрстуфхцчшщъыьэюя "),
    en=list("abcdefghijklmnopqrstuvwxyz ")
)
