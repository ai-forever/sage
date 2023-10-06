<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="images/sage-white.svg">
    <source media="(prefers-color-scheme: light)" srcset="images/sage-black.svg">
    <img alt="SAGE" src="images/sage-white.svg" style="max-width: 100%;">
  </picture>
</p>

<p align="center">
    <a href="https://opensource.org/licenses/MIT">
    <img alt="License" src="https://img.shields.io/badge/License-MIT-yellow.svg">
    </a>
    <a href="https://github.com/ai-forever/sage/releases">
    <img alt="Release" src="https://img.shields.io/badge/release-v1.0.0-blue">
    </a>
    <a href="https://arxiv.org/abs/2308.09435">
    <img alt="Paper" src="https://img.shields.io/badge/arXiv-2308.09435-red">
    </a>

</p>

<h2 align="center">
    <p> Spelling correction, corruption and evaluation for multiple languages
</p>
</h2>

<div align="center">
  <h4>
    <a href="#installation">Install</a> |
    <a href="#spelling-correction">Models</a> |
    <a href="#evaluation">Evaluation</a> |
    <a href="#statistic-based-spelling-corruption-sbsc">SBSC</a> |
    <a href="#augmentex">Augmentex</a> |
    <a href="#citation">Papers</a>
  </h4>
</div>

SAGE (Spell checking via Augmentation and Generative distribution Emulation) is 
a complete solution that you need when working on a spelling problem:

- 💯 Spelling correction with State-of-the-art pre-trained 🤗Transformer models:
  - [M2M100-1.2B](https://huggingface.co/ai-forever/RuM2M100-1.2B)
  - [M2M100-418M](https://huggingface.co/ai-forever/RuM2M100-418M)
  - [FredT5-large](https://huggingface.co/ai-forever/FRED-T5-large-spell)
  - [T5-large](https://huggingface.co/ai-forever/T5-large-spell)

  You can test them out right here [![Try Model Generation In Colab!](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ai-forever/sage/blob/main/notebooks/text_correction_demo.ipynb)
- 🧩 Augment your data with spelling corruption algorithms, take a look at a quick demo [![Try Model Generation In Colab!](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ai-forever/sage/blob/main/notebooks/text_corruption_demo.ipynb)
- 📊 Evaluate performance of spelling correction tools.

## Table of contents

- [Installation](#installation)
  - [Regular install](#regular-install)
  - [Editable install](#editable-install)
- [Quick demo](#quick-demo)
- [Spelling corruption](#spelling-corruption)
  - [Statistic-based Spelling Corruption (SBSC)](#statistic-based-spelling-corruption-sbsc)
  - [Augmentex](#augmentex)
- [Spelling correction](#spelling-correction)
  - [RUSpellRU evaluation](#ruspellru-evaluation)
  - [MultidomainGold evaluation](#multidomaingold-evaluation)
  - [MedSpellchecker evaluation](#medspellchecker-evaluation)
  - [GitHubTypoCorpusRu evaluation](#githubtypocorpusru-evaluation)
- [Evaluation](#evaluation)
- [Citation](#citation)

## Installation
### Regular install
```commandline
git clone https://github.com/ai-forever/sage.git
cd sage
pip install .
pip install -r requirements.txt
```

### Editable install
```commandline
git clone https://github.com/ai-forever/sage.git
cd sage
pip install -e .
pip install -r requirements.txt
```

## Quick demo
Lets spoil some text:
```python
import sage
from sage.spelling_corruption import SBSCConfig, SBSCCorruptor
from sage.utils import DatasetsAvailable

text = "Заметьте, не я это предложил!"

# Instantiate SBSC corruptor from a dataset with errors in medical anamnesis
config = SBSCConfig(
    reference_dataset_name_or_path=DatasetsAvailable.MedSpellchecker.name,
    reference_dataset_split="test"
)
corruptor = SBSCCorruptor.from_config(config)

corruptor.corrupt(text, seed=1)
# 'Заветьте, не я это предложил!'
```
... now with Augmentex:

```python
import sage
from sage.spelling_corruption import WordAugConfig, WordAugCorruptor

text = "Заметьте, не я это предложил!"

# Instantiate WordAugCorruptor corruptor with a custom set of parameters
config = WordAugConfig(
    min_aug=1,
    max_aug=5,
    unit_prob=0.4,
)
corruptor = WordAugCorruptor.from_config(config)

corruptor.corrupt(text, seed=1)
# 'это не предложил! Заметьте, я'
```

... or for the English language:

```python
import os
from sage.spelling_corruption import SBSCConfig, SBSCCorruptor

text = "Screw you guys, I am going home. (c)"

# Instantiate SBSC corruptor from a JFLEG dataset
config = SBSCConfig(
    lang="en",
    reference_dataset_name_or_path=os.path.join("data", "example_data", "jfleg"),
)
corruptor = SBSCCorruptor.from_config(config)

corruptor.corrupt(text, seed=1)
# 'Screw you kuys, I am going home. (c)'
```

Now we can use our models to restore the initial text back:
```python
from sage.spelling_correction import AvailableCorrectors
from sage.spelling_correction import RuM2M100ModelForSpellingCorrection, T5ModelForSpellingCorruption

text_ru = "Заветьте, не я это предложил!"
text_en = "Screw you kuys, I am going home. (c)"

corrector_1b = RuM2M100ModelForSpellingCorrection.from_pretrained(AvailableCorrectors.m2m100_1B.value)
corrector_en = T5ModelForSpellingCorruption.from_pretrained(AvailableCorrectors.ent5_large.value)

corrector_1b.correct(text_ru)
# ['Заметьте, не я это предложил!']

corrector_en.correct(text_en, prefix="grammar: ")
# ['Screw you guys, I am going home. (c)']
```

Evaluate performance of the models on open benchmarks for spelling correction:
```python
import os
import torch
from sage.utils import DatasetsAvailable
from sage.spelling_correction import AvailableCorrectors
from sage.spelling_correction import RuM2M100ModelForSpellingCorrection, T5ModelForSpellingCorruption

corrector_418m = RuM2M100ModelForSpellingCorrection.from_pretrained(AvailableCorrectors.m2m100_418M.value)
corrector_en = T5ModelForSpellingCorruption.from_pretrained(AvailableCorrectors.ent5_large.value)

corrector_418m.model.to(torch.device("cuda:0"))
corrector_en.model.to(torch.device("cuda:0"))

metrics = corrector_418m.evaluate(DatasetsAvailable.RUSpellRU.name, batch_size=32)
print(metrics)
# {'Precision': 57.74, 'Recall': 61.18, 'F1': 59.41}

metrics = corrector_en.evaluate(os.path.join("data", "example_data", "jfleg"), prefix="grammar: ", batch_size=32)
print(metrics)
# {'Precision': 83.43, 'Recall': 84.25, 'F1': 83.84}

```

_NOTE_: if you are launching code snippet in Colab you'd probably end up with MEMORY ERROR, so manage evaluation 
procedures so that you meet available device's restrictions. As a feasible workaround you can execute 
```python
del corrector_418m.model
```
to free some space. 

## Spelling Corruption
We implemented two methods for spelling corruption. **S**tatistic-**b**ased **S**pelling **C**orruption (**SBSC**) aims 
to mimic human behaviour when making an error. While [Augmentex](#augmentex) relies on rule-based heuristics and common
errors and mistypings especially those committed while typing text on a keyboard. 

🚀 Both methods proved their effectiveness for spelling correction systems and celebrated substantial **performance gains**
fully reported in our [Paper](https://arxiv.org/abs/2308.09435).

### Statistic-based Spelling Corruption (SBSC)
This method is thoroughly described in our another [Paper](https://www.dialog-21.ru/media/5914/martynovnplusetal056.pdf) 
and in this 🗣️[Talk](https://youtu.be/yFfkV0Qjuu0?si=XmKfocCSLnKihxS_). 

Briefly, SBSC follows two simple steps:
- 🧠 Analyze errors, their type and positions in a source text;
- ✏️ Reproduce errors from the source text in a new sentence;

🧠 To analyze errors in a source sentence we need its corresponding correction in order to build 
[Levenshtein matrix](https://en.wikipedia.org/wiki/Levenshtein_distance), traverse it back starting from the 
bottom right entry and determine the exact position and type of an error. We then aggregate all obtained statistics and 
normalize it to valid discrete distributions. 

✏️ "Reproduce" step is even less complicated: we just sample number of errors per sentence, their types and relative
positions from corresponding distributions and apply them to a correct sentence.

As stated, you need a parallel dataset to "fit" SBSC. We provide a set of four datasets with natural errors covering
exhaustive range of domains:

- **RUSpellRU**: texts collected from [LiveJournal](https://www.livejournal.com/media), with manually corrected typos and errors;
- **MultidomainGold**: examples from 7 text sources, including the open web, news, social media, reviews, subtitles, policy documents and literary works;
- **MedSpellChecker**: texts with errors from medical anamnesis;
- **GitHubTypoCorpusRu**: spelling errors and typos in commits from GitHub;

You can use them as simple as
```python
import sage
from sage.spelling_corruption import SBSCConfig, SBSCCorruptor
from sage.utils import DatasetsAvailable

# Instantiate SBSC corruptor from a dataset with errors in medical anamnesis
config = SBSCConfig(
    reference_dataset_name_or_path=DatasetsAvailable.MedSpellchecker.name,
    reference_dataset_split="test"
)
corruptor = SBSCCorruptor.from_config(config)
```

... or you can initialize your SBSC from locally stored dataset:
```python
import os
from sage.spelling_corruption import SBSCConfig, SBSCCorruptor

# Instantiate SBSC corruptor from a JFLEG dataset
config = SBSCConfig(
    lang="en",
    reference_dataset_name_or_path=os.path.join("data", "example_data", "jfleg"),
)
corruptor = SBSCCorruptor.from_config(config)
```

✅ To check how good SBSC actually approximates original errors, you can plot side-by-side graphs of original and 
synthetically generated distributions:

<p align="center">
    <br>
    <img src="images/ruspellru_side_by_side.jpg" width="400" style="float:center; padding-right:60px"/> 
    <img src="images/bea60k_side_by_side.jpg" width="400" style="float:center; padding-left:60px"/>
    <br>
<p>

To access these graphs you can simply
```python
from sage.utils import load_available_dataset_from_hf, draw_and_save_errors_distributions_comparison_charts
from sage.spelling_corruption.sbsc.labeler import process_mistypings
from sage.spelling_corruption import SBSCCorruptor

sources, corrections = load_available_dataset_from_hf("RUSpellRU", for_labeler=True, split="train")
ruspellru_stats, ruspellru_confusion_matrix, ruspellru_typos_cnt = process_mistypings(sources, corrections)

corruptor = SBSCCorruptor.from_default_config()
spoiled_sentences = corruptor.batch_corrupt(corrections)

sbsc_stats, sbsc_confusion_matrix, sbsc_typos_cnt = process_mistypings(spoiled_sentences, corrections)

draw_and_save_errors_distributions_comparison_charts(
    actual_typos_cnt = sbsc_typos_cnt,
    reference_typos_cnt=ruspellru_typos_cnt,
    actual_stats=sbsc_stats,
    reference_stats=ruspellru_stats,
    path_to_save="ruspellru_sbsc.jpg"
)
```

### Augmentex
Augmentex introduces rule-based and common statistic (empowered by [KartaSlov](https://kartaslov.ru) project) 
approach to insert errors in text. It is fully described again in the [Paper](https://www.dialog-21.ru/media/5914/martynovnplusetal056.pdf)
and in this 🗣️[Talk](https://youtu.be/yFfkV0Qjuu0?si=XmKfocCSLnKihxS_).

🖇️ Augmentex allows you to operate on two levels of granularity when it comes to text corruption and offers you sets of 
specific methods suited for particular level:
- **Word level**:
  - _replace_ - replace a random word with its incorrect counterpart;
  - _delete_ - delete random word;
  - _swap_ - swap two random words;
  - _stopword_ - add random words from stop-list;
  - _reverse_ - change a case of the first letter of a random word;
- **Character level**:
  - _shift_ - randomly swaps upper / lower case in a string;
  - _orfo_ - substitute correct characters with their common incorrect counterparts;
  - _typo_ - substitute correct characters as if they are mistyped on a keyboard;
  - _delete_ - delete random character;
  - _multiply_ - multiply random character;
  - _swap_ - swap two adjacent characters;
  - _insert_ - insert random character;

To access Augmentex you only need these few manipulations:
```python
from sage.spelling_corruption import CharAugConfig, CharAugCorruptor

config = CharAugConfig(
    unit_prob=0.3, # proportion of characters that is going to undergo edits
    min_aug=1, # minimum number of edits
    max_aug=5, # maximum number of edits 
    mult_num=3 # `multiply` edit
)
corruptor = CharAugCorruptor.from_config(config)
```

... or like this:

```python
from sage.spelling_corruption import WordAugConfig, WordAugCorruptor

config = WordAugConfig(
    unit_prob=0.4, # proportion of characters that is going to undergo edits
    min_aug=1, # minimum number of edits
    max_aug=5, # maximum number of edits 
)
corruptor = WordAugCorruptor.from_config(config)
```

Augmentex has been created by our fellow team, the project has its own [repo](https://github.com/ai-forever/augmentex), do not forget to take a look! 

## Spelling Correction
Our methodology for obtaining model with optimal performance on spellchecking task is thoroughly described in our
[Paper](https://arxiv.org/abs/2308.09435). And the algorithm is simple and generally consists of two steps:

- Pre-train model on extensive parallel corpus with synthetically generated errors;
- Fine-tune on combinations of available datasets for spelling correction with "human-made" errors;

We use [Augmentex](#augmentex) and [SBSC](#statistic-based-spelling-corruption-sbsc) for both generating large synthetic corpora and augmenting datasets with natural errors. 
We release 4 pre-trains of our models.

We've 3 🤗Transformer models for Russian 🇷🇺:
  - [M2M100-1.2B](https://huggingface.co/ai-forever/RuM2M100-1.2B)
  - [M2M100-418M](https://huggingface.co/ai-forever/RuM2M100-418M)
  - [FredT5-large](https://huggingface.co/ai-forever/FRED-T5-large-spell)

And one model for English 🇬🇧:
  - [T5-large](https://huggingface.co/ai-forever/T5-large-spell)

Models for the Russian language have been pre-trained on combination of Russian Wikipedia and videos transcriptions with 
artificial errors generated by [SBSC](#statistic-based-spelling-corruption-sbsc) on statistics gathered from train split of [RUSpellRU](https://huggingface.co/datasets/ai-forever/spellcheck_benchmark). 
T5 for English trained on mixture of English Wikipedia articles and news posts with synthetic errors inserted by [SBSC](#statistic-based-spelling-corruption-sbsc) fitted on statistics from 5k subsample
of [BEA60k](https://github.com/neuspell/neuspell/tree/master).

📚 We also validate our pre-trains for Russian on all available datasets with "human-made" errors:

- **RUSpellRU**: texts collected from [LiveJournal](https://www.livejournal.com/media), with manually corrected typos and errors;
- **MultidomainGold**: examples from 7 text sources, including the open web, news, social media, reviews, subtitles, policy documents and literary works;
- **MedSpellChecker**: texts with errors from medical anamnesis;
- **GitHubTypoCorpusRu**: spelling errors and typos in commits from GitHub;

📈 Here we report evaluation of some setups:
- Zero-shot evaluation of pre-trained (**Pre-train**) checkpoints, which we publicly release;
- Additional fine-tuning (**Pre-train + fine-tune**) on the target dataset;

Full list of setups and corresponding performances are in the [Paper](https://arxiv.org/abs/2308.09435).

_NOTE:_ **MedSpellChecker** and **GitHubTypoCorpusRu** do not have train split, so their performance on 
**Pre-train + fine-tune** setup is reported as a result of fine-tuning on combination of **RUSpellRU** and **MultidomainGold**
datasets.

#### RUSpellRU Evaluation

| Model                                | Precision | Recall   | F1       |
|--------------------------------------|-----------|----------|----------|
| M2M100-1.2B (Pre-train)              | 59.4      | 43.3     | 50.1     |
| M2M100-1.2B (Pre-train + fine-tune)  | 82.9      | 72.5     | **77.3** |
| M2M100-418M (Pre-train)              | 57.7      | 61.2     | 59.4     |
| M2M100-418M (Pre-train + fine-tune)  | 81.8      | 63.4     | 71.4     |
| FredT5-large (Pre-train)             | 58.5      | 42.4     | 49.2     |
| FredT5-large (Pre-train + fine-tune) | 55.1      | 73.2     | 62.9     |
| ChatGPT text-davinci-003             | 55.9      | **75.3** | 64.2     |
| Yandex.Speller                       | **83.0**  | 59.8     | 69.5     |


#### MultidomainGold Evaluation

| Model                                | Precision | Recall   | F1       |
|--------------------------------------|-----------|----------|----------|
| M2M100-1.2B (Pre-train)              | 56.4      | 44.8     | 49.9     |
| M2M100-1.2B (Pre-train + fine-tune)  | **62.5**  | 60.9     | **61.7** |
| M2M100-418M (Pre-train)              | 32.8      | 56.3     | 41.5     |
| M2M100-418M (Pre-train + fine-tune)  | 57.9      | 56.5     | 57.2     |
| FredT5-large (Pre-train)             | 42.5      | 42.0     | 42.2     |
| FredT5-large (Pre-train + fine-tune) | 61.7      | 60.5     | 61.1     |
| ChatGPT gpt-4-0314                   | 34.0      | **73.2** | 46.4     |
| Yandex.Speller                       | 52.9      | 51.4     | 52.2     |


#### MedSpellchecker Evaluation

| Model                                | Precision | Recall   | F1       |
|--------------------------------------|-----------|----------|----------|
| M2M100-1.2B (Pre-train)              | 63.7      | 57.8     | 60.6     |
| M2M100-1.2B (Pre-train + fine-tune)  | 78.8      | **71.4** | **74.9** |
| M2M100-418M (Pre-train)              | 23.2      | 64.5     | 34.1     |
| M2M100-418M (Pre-train + fine-tune)  | 73.1      | 62.4     | 67.3     |
| FredT5-large (Pre-train)             | 37.2      | 51.7     | 43.3     |
| FredT5-large (Pre-train + fine-tune) | 37.5      | 59.3     | 45.9     |
| ChatGPT gpt-4-0314                   | 54.2      | 69.4     | 60.9     |
| Yandex.Speller                       | **80.6**  | 47.8     | 60.0     |


#### GitHubTypoCorpusRu Evaluation

| Model                                | Precision | Recall   | F1       |
|--------------------------------------|-----------|----------|----------|
| M2M100-1.2B (Pre-train)              | 45.7      | 41.4     | 43.5     |
| M2M100-1.2B (Pre-train + fine-tune)  | 47.1      | 42.9     | 44.9     |
| M2M100-418M (Pre-train)              | 27.5      | 42.6     | 33.4     |
| M2M100-418M (Pre-train + fine-tune)  | 42.8      | 37.8     | 40.2     |
| FredT5-large (Pre-train)             | 52.7      | 42.4     | 46.6     |
| FredT5-large (Pre-train + fine-tune) | 61.2      | 45.4     | **52.1** |
| ChatGPT text-davinci-003             | 46.5      | **58.1** | 51.7     |
| Yandex.Speller                       | **67.7**  | 37.5     | 48.3     |

All the mentioned datasets are available as HuggingFace datasets [here](https://huggingface.co/datasets/ai-forever/spellcheck_benchmark) and through the API of our library: 
```python
from sage.utils import load_available_dataset_from_hf, DatasetsAvailable

print([dataset.name for dataset in DatasetsAvailable])
# ['MultidomainGold', 'RUSpellRU', 'MedSpellchecker', 'GitHubTypoCorpusRu']

gold_dataset = load_available_dataset_from_hf(DatasetsAvailable.MultidomainGold.name, for_labeler=False)
print(len(gold_dataset))
# 7678

sources, corrections = load_available_dataset_from_hf(DatasetsAvailable.RUSpellRU.name, for_labeler=True, split="train")
print(len(sources), len(corrections))
# 2000 2000
```

## Evaluation
We also provide functionality to evaluate the performance of spelling correction systems and rank them. 

🎯 Here is what you get and how you can interpret these:
- **Precision**: one minus share of unnecessary amendments; 
- **Recall**: proportion of expected corrections;
- **F1**: famous geometric mean of aforementioned two;

You can obtain these metrics simply by
```python
from sage.evaluation import evaluation
from sage.utils import DatasetsAvailable, load_available_dataset_from_hf

sources, corrections = load_available_dataset_from_hf(DatasetsAvailable.RUSpellRU.name, for_labeler=True, split="test")
metrics = evaluation(sources, corrections, corrections)
print(metrics)
# {'Precision': 100.0, 'Recall': 100.0, 'F1': 100.0}
```

... or by directly assessing the model:
```python
import torch
from sage.spelling_correction import AvailableCorrectors, RuM2M100ModelForSpellingCorrection, T5ModelForSpellingCorruption
from sage.utils import DatasetsAvailable

corrector = RuM2M100ModelForSpellingCorrection.from_pretrained(AvailableCorrectors.m2m100_418M.value)
corrector.model.to(torch.device("cuda:0"))

metrics = corrector.evaluate(DatasetsAvailable.MultidomainGold.name, batch_size=16)
print(metrics)
# {'Precision': 32.82, 'Recall': 57.69, 'F1': 41.84}

corrector = T5ModelForSpellingCorruption.from_pretrained(AvailableCorrectors.ent5_large.value)
corrector.model.to(torch.device("cuda:0"))

metrics = corrector.evaluate("../data/example_data/jfleg/", batch_size=32, prefix="grammar: ")
print(metrics)
# {'Precision': 83.43, 'Recall': 84.25, 'F1': 83.84}
```

📌 Credit for evaluation script goes to Aleksei Sorokin and his notable [work](https://www.dialog-21.ru/media/3427/sorokinaaetal.pdf) 
in proceedings of [SpellRueval](https://www.dialog-21.ru/evaluation/2016/spelling_correction/). 

## Citation
If you want to know more about our work take a look at these publications:

💥 Our first [Paper](https://arxiv.org/abs/2308.09435) provides a thorough description of the methodology used to obtain SOTA 
models for spelling corrections as well the comprehensive reports of all experiments that have been carried out. 

💫 While our Dialogue-2023 [Paper](https://www.dialog-21.ru/media/5914/martynovnplusetal056.pdf) focuses on exploiting 
resources for the task of spelling correction and procedures on obtaining high-quality parallel corpuses. 

```
@misc{martynov2023methodology,
      title={A Methodology for Generative Spelling Correction
via Natural Spelling Errors Emulation across Multiple Domains and Languages}, 
      author={Nikita Martynov and Mark Baushenko and Anastasia Kozlova and
Katerina Kolomeytseva and Aleksandr Abramov and Alena Fenogenova},
      year={2023},
      eprint={2308.09435},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}

@inproceedings{martynov2023augmentation,
  title={Augmentation methods for spelling corruptions},
  author={Martynov, Nikita and Baushenko, Mark and Abramov, Alexander and Fenogenova, Alena},
  booktitle={Proceedings of the International Conference “Dialogue},
  volume={2023},
  year={2023}
}
```

📌 Feel free to ask any questions regarding our work at corresponding point of contact:

_nikita.martynov.98@list.ru_