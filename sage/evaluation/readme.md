### Evaluation in SAGE

#### RuERRANT

RuERRANT is an adaptation of the ERRANT metric ([repo](https://github.com/chrisjbryant/errant), [paper-2016](https://aclanthology.org/C16-1079.pdf), [paper-2017](https://aclanthology.org/P17-1074.pdf)) to the Russian language. The adaptation was primarily done in https://github.com/Askinkaty/errant and further developed within SAGE. The changes to the original ERRANT implementation for English are the following:

1. Basic parsing model changed to Spacy's `ru_core_news_lg`.
1. Included a dictionary of Russian words (main forms).
1. Introduced detection of error correction types specific for Russian (degrees of adjectives, verb aspect).
1. [our contribution] Introduced a simplified error correction typology:
    - `CASE`: spelling corrections including only character case change;
    - `PUNCT`: punctuation corrections;
    - `YO`: spelling corrections regarding "е"/"ё" substitutions;
    - `SPELL`: all other word-level spelling corrections.
1. [our contribution] Introduced detection of multiple error correction types per word, e.g. "федор" -> "Фёдор" contains both CASE and YO corrections.
1. [our contribution] Introduced detection of inner word punctuation corrections which covers joint ("AB") vs. hyphen ("A-B") vs. space ("A B") word spelling. Corrections of this type are attributed to the `SPELL` category.

#### Scoring

To score model's corrections against gold corrections, use a Scorer instance:

```python
from sage.evaluation.scorer import Scorer

s = Scorer()

s.score(
    ["спел Кейс ее .", "спел Кейс ее ."],
    ["спелл кейс её !", "спелл кейс её !"],
    ["спел кейс её .", "спелл Кейс ее !"],
    metrics=["errant"]
)
>>> {'CASE_Precision': 100.0, 'CASE_Recall': 50.0, 'CASE_F1': 66.67,
     'YO_Precision': 100.0, 'YO_Recall': 50.0, 'YO_F1': 66.67,
     'SPELL_Precision': 100.0, 'SPELL_Recall': 50.0, 'SPELL_F1': 66.67,
     'PUNCT_Precision': 100.0, 'PUNCT_Recall': 50.0, 'PUNCT_F1': 66.67}
```
