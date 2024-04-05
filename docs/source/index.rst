.. role:: raw-html-m2r(raw)
   :format: html

.. image:: images/sage-black.svg
   :align: center


.. raw:: html

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
       <a href="https://colab.research.google.com/github/ai-forever/sage/blob/main/notebooks/text_correction_demo.ipynb">
       <img alt="Paper" src="https://colab.research.google.com/assets/colab-badge.svg/">
       </a>
       </a>
       <a href="https://colab.research.google.com/github/ai-forever/sage/blob/main/notebooks/text_corruption_demo.ipynb">
       <img alt="Paper" src="https://colab.research.google.com/assets/colab-badge.svg/">
       </a>
   </p>


.. raw:: html

   <h1 align="center">
       <p> Spelling correction, corruption and evaluation for multiple languages
   </p>
   </h1>



.. raw:: html

   <div align="center">
     <h4>
       <a href="#id1">Install</a> |
       <a href="#id12">Models</a> |
       <a href="#id24">Evaluation</a> |
       <a href="#id7">SBSC</a> |
       <a href="#id9">Augmentex</a> |
       <a href="#id24">Papers</a>
     </h4>
   </div>


SAGE (Spell checking via Augmentation and Generative distribution Emulation) is 
a complete solution that you need when working on a spelling problem:

💯 Spelling correction with State-of-the-art pre-trained 🤗Transformer models:
  1️⃣ `sage-fredt5-large <https://huggingface.co/ai-forever/sage-fredt5-large>`_

  2️⃣ `sage-fredt5-distilled-95m <https://huggingface.co/ai-forever/sage-fredt5-distilled-95m>`_

  3️⃣ `sage-mt5-large <https://huggingface.co/ai-forever/sage-mt5-large>`_

  4️⃣ `sage-m2m100-1.2B <https://huggingface.co/ai-forever/sage-m2m100-1.2B>`_

  5️⃣ `T5-large <https://huggingface.co/ai-forever/T5-large-spell>`_

  6️⃣ `M2M100-1.2B <https://huggingface.co/ai-forever/RuM2M100-1.2B>`_

  7️⃣ `M2M100-418M <https://huggingface.co/ai-forever/RuM2M100-418M>`_

  6️⃣ `FredT5-large <https://huggingface.co/ai-forever/FRED-T5-large-spell>`_



🧩 **Augment your data with spelling corruption algorithms**

📊 **Evaluate performance of spelling correction tools**


Table of contents
-----------------


* `Installation <#id1>`_

  * `Regular install <#id2>`_
  * `Editable install <#id3>`_

* `Quick demo <#id4>`_
* `Spelling corruption <#id5>`_

  * `Statistic-based Spelling Corruption (SBSC) <#id7>`_
  * `Augmentex <#id9>`_

* `Spelling correction <#id12>`_

  * `RUSpellRU evaluation <#ruspellru-evaluation>`_
  * `MultidomainGold evaluation <#multidomaingold-evaluation>`_
  * `MedSpellchecker evaluation <#medspellchecker-evaluation>`_
  * `GitHubTypoCorpusRu evaluation <#githubtypocorpusru-evaluation>`_

* `Evaluation <#id24>`_
* `Citation <#id24>`_


Installation
------------

Regular install
^^^^^^^^^^^^^^^
.. code-block:: commandline

   git clone https://github.com/ai-forever/sage.git
   cd sage
   pip install .

To install extra requirements that you are going to need when working with ERRANT-based metric run

.. code-block:: commandline

   pip install -e ".[errant]"

or just

.. code-block:: commandline

   pip install -e .[errant]


Editable install
^^^^^^^^^^^^^^^^

.. code-block:: commandline

   git clone https://github.com/ai-forever/sage.git
   cd sage
   pip install -e .

and proceed with extra requirements install as above.

Quick demo
----------

Lets spoil some text:

.. code-block:: python

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
   # 'Заиетьте, не я эт о пред ложил!'

... now with Augmentex:

.. code-block:: python

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

... or for the English language:

.. code-block:: python

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

Now we can use our models to restore the initial text back:

.. code-block:: python

    from sage.spelling_correction import AvailableCorrectors
    from sage.spelling_correction import RuM2M100ModelForSpellingCorrection, T5ModelForSpellingCorruption

    text_ru = "Замтьте не я это предложил"
    text_en = "Screw you kuys, I am going home. (c)"

    corrector_fred = T5ModelForSpellingCorruption.from_pretrained(AvailableCorrectors.sage_fredt5_large.value)
    corrector_m2m = RuM2M100ModelForSpellingCorrection.from_pretrained(AvailableCorrectors.m2m100_1B.value)
    corrector_en = T5ModelForSpellingCorruption.from_pretrained(AvailableCorrectors.ent5_large.value)

    print(corrector_fred.correct(text_ru))
    # ['Заметьте, не я это предложил.']

    print(corrector_m2m.correct(text_ru))
    # ['Заметьте не я это предложил']

    print(corrector_en.correct(text_en, prefix="grammar: "))
    # ['Screw you guys, I am going home. (c)']

Evaluate performance of the models on open benchmarks for spelling correction:

.. code-block:: python

    import os
    import torch
    from sage.utils import DatasetsAvailable
    from sage.spelling_correction import AvailableCorrectors
    from sage.spelling_correction import T5ModelForSpellingCorruption

    corrector_fred_95m = T5ModelForSpellingCorruption.from_pretrained(AvailableCorrectors.sage_fredt5_distilled_95m.value)
    corrector_mt5 = T5ModelForSpellingCorruption.from_pretrained(AvailableCorrectors.sage_mt5_large.value)

    corrector_fred_95m.model.to(torch.device("cuda:0"))
    corrector_mt5.model.to(torch.device("cuda:0"))

    metrics = corrector_fred_95m.evaluate("RUSpellRU", metrics=["errant", "ruspelleval"], batch_size=32)
    print(metrics)
    # {'CASE_Precision': 94.41, 'CASE_Recall': 92.55, 'CASE_F1': 93.47, 'SPELL_Precision': 77.52, 'SPELL_Recall': 64.09, 'SPELL_F1': 70.17, 'PUNCT_Precision': 86.77, 'PUNCT_Recall': 80.59, 'PUNCT_F1': 83.56, 'YO_Precision': 46.21, 'YO_Recall': 73.83, 'YO_F1': 56.84, 'Precision': 83.48, 'Recall': 74.75, 'F1': 78.87}

    metrics = corrector_mt5.evaluate("/content/sage/data/example_data/jfleg", metrics=["ruspelleval"], batch_size=16)
    print(metrics)
    # {'Precision': 75.94, 'Recall': 88.15, 'F1': 81.59}

*NOTE*\ : if you are launching code snippet in Colab you'd probably end up with MEMORY ERROR, so manage evaluation 
procedures so that you meet available device's restrictions. As a feasible workaround you can execute 

.. code-block:: python

   del corrector_fred_95m.model

to free some space. 

Spelling Corruption
-------------------

We implemented two methods for spelling corruption. **S**\ tatistic-\ **b**\ ased **S**\ pelling **C**\ orruption (\ **SBSC**\ ) aims 
to mimic human behaviour when making an error. While `Augmentex <#augmentex>`_ relies on rule-based heuristics and common
errors and mistypings especially those committed while typing text on a keyboard. 

🚀 Both methods proved their effectiveness for spelling correction systems and celebrated substantial **performance gains**
fully reported in our `Paper <https://aclanthology.org/2024.findings-eacl.10/>`_.

Statistic-based Spelling Corruption (SBSC)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This method is thoroughly described in our another `Paper <https://www.dialog-21.ru/media/5914/martynovnplusetal056.pdf>`_ 
and in this 🗣️\ `Talk <https://youtu.be/yFfkV0Qjuu0?si=XmKfocCSLnKihxS_>`_. 

Briefly, SBSC follows two simple steps:


* 🧠 Analyze errors, their type and positions in a source text;
* ✏️ Reproduce errors from the source text in a new sentence;

🧠 To analyze errors in a source sentence we need its corresponding correction in order to build 
`Levenshtein matrix <https://en.wikipedia.org/wiki/Levenshtein_distance>`_\ , traverse it back starting from the 
bottom right entry and determine the exact position and type of an error. We then aggregate all obtained statistics and 
normalize it to valid discrete distributions. 

✏️ "Reproduce" step is even less complicated: we just sample number of errors per sentence, their types and relative
positions from corresponding distributions and apply them to a correct sentence.

As stated, you need a parallel dataset to "fit" SBSC. We provide a set of four datasets with natural errors covering
exhaustive range of domains:


* **RUSpellRU**\ : texts collected from `LiveJournal <https://www.livejournal.com/media>`_\ , with manually corrected typos and errors;
* **MultidomainGold**\ : examples from 7 text sources, including the open web, news, social media, reviews, subtitles, policy documents and literary works;
* **MedSpellChecker**\ : texts with errors from medical anamnesis;
* **GitHubTypoCorpusRu**\ : spelling errors and typos in commits from GitHub;

You can use them as simple as

.. code-block:: python

   import sage
   from sage.spelling_corruption import SBSCConfig, SBSCCorruptor
   from sage.utils import DatasetsAvailable

   # Instantiate SBSC corruptor from a dataset with errors in medical anamnesis
   config = SBSCConfig(
       reference_dataset_name_or_path=DatasetsAvailable.MedSpellchecker.name,
       reference_dataset_split="test"
   )
   corruptor = SBSCCorruptor.from_config(config)

... or you can initialize your SBSC from locally stored dataset:

.. code-block:: python

   import os
   from sage.spelling_corruption import SBSCConfig, SBSCCorruptor

   # Instantiate SBSC corruptor from a JFLEG dataset
   config = SBSCConfig(
       lang="en",
       reference_dataset_name_or_path=os.path.join("data", "example_data", "jfleg"),
   )
   corruptor = SBSCCorruptor.from_config(config)

✅ To check how good SBSC actually approximates original errors, you can plot side-by-side graphs of original and 
synthetically generated distributions:


|pic1|  |pic2|

.. |pic1| image:: images/bea60k_side_by_side.jpg
   :width: 45%

.. |pic2| image:: images/ruspellru_side_by_side.jpg
   :width: 45%



To access these graphs you can simply

.. code-block:: python

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

Augmentex
^^^^^^^^^

Augmentex introduces rule-based and common statistic (empowered by `KartaSlov <https://kartaslov.ru>`_ project) 
approach to insert errors in text. It is fully described again in the `Paper <https://www.dialog-21.ru/media/5914/martynovnplusetal056.pdf>`_
and in this 🗣️\ `Talk <https://youtu.be/yFfkV0Qjuu0?si=XmKfocCSLnKihxS_>`_.

🖇️ Augmentex allows you to operate on two levels of granularity when it comes to text corruption and offers you sets of 
specific methods suited for particular level:


* **Word level**\ :

  * *replace* - replace a random word with its incorrect counterpart;
  * *delete* - delete random word;
  * *swap* - swap two random words;
  * *stopword* - add random words from stop-list;
  * *reverse* - change a case of the first letter of a random word;

* **Character level**\ :

  * *shift* - randomly swaps upper / lower case in a string;
  * *orfo* - substitute correct characters with their common incorrect counterparts;
  * *typo* - substitute correct characters as if they are mistyped on a keyboard;
  * *delete* - delete random character;
  * *multiply* - multiply random character;
  * *swap* - swap two adjacent characters;
  * *insert* - insert random character;

To access Augmentex you only need these few manipulations:

.. code-block:: python

   from sage.spelling_corruption import CharAugConfig, CharAugCorruptor

   config = CharAugConfig(
       unit_prob=0.3, # proportion of characters that is going to undergo edits
       min_aug=1, # minimum number of edits
       max_aug=5, # maximum number of edits 
       mult_num=3 # `multiply` edit
   )
   corruptor = CharAugCorruptor.from_config(config)

... or like this:

.. code-block:: python

   from sage.spelling_corruption import WordAugConfig, WordAugCorruptor

   config = WordAugConfig(
       unit_prob=0.4, # proportion of characters that is going to undergo edits
       min_aug=1, # minimum number of edits
       max_aug=5, # maximum number of edits 
   )
   corruptor = WordAugCorruptor.from_config(config)

Augmentex has been created by our fellow team, the project has its own `repo <https://github.com/ai-forever/augmentex>`_\ , do not forget to take a look! 

Spelling Correction
-------------------

Our methodology for obtaining model with optimal performance on spellchecking task is thoroughly described in our
`Paper <https://aclanthology.org/2024.findings-eacl.10/>`_. And the algorithm is simple and generally consists of two steps:


* Pre-train model on extensive parallel corpus with synthetically generated errors;
* Fine-tune on combinations of available datasets for spelling correction with "human-made" errors;

We use `Augmentex <#augmentex>`_ and `SBSC <#statistic-based-spelling-corruption-sbsc>`_ for both generating large synthetic corpora and augmenting datasets with natural errors. 
We release 4 pre-trains of our models.

We've 6 🤗Transformer models for Russian 🇷🇺:


* `sage-fredt5-large <https://huggingface.co/ai-forever/sage-fredt5-large>`_
* `sage-fredt5-distilled-95m <https://huggingface.co/ai-forever/sage-fredt5-distilled-95m>`_
* `sage-m2m100-1.2B <https://huggingface.co/ai-forever/sage-m2m100-1.2B>`_
* `M2M100-1.2B <https://huggingface.co/ai-forever/RuM2M100-1.2B>`_
* `M2M100-418M <https://huggingface.co/ai-forever/RuM2M100-418M>`_
* `FredT5-large <https://huggingface.co/ai-forever/FRED-T5-large-spell>`_

And two model for English 🇬🇧:


* `sage-mt5-large <https://huggingface.co/ai-forever/sage-mt5-large>`_
* `T5-large <https://huggingface.co/ai-forever/T5-large-spell>`_

Models for the Russian language have been pre-trained on combination of Russian Wikipedia and videos transcriptions with 
artificial errors generated by `SBSC <#statistic-based-spelling-corruption-sbsc>`_ on statistics gathered from train split of `RUSpellRU <https://huggingface.co/datasets/ai-forever/spellcheck_benchmark>`_. 
T5 for English trained on mixture of English Wikipedia articles and news posts with synthetic errors inserted by `SBSC <#statistic-based-spelling-corruption-sbsc>`_ fitted on statistics from 5k subsample
of `BEA60k <https://github.com/neuspell/neuspell/tree/master>`_.

📚 We also validate our pre-trains for Russian on all available datasets with "human-made" errors:


* **RUSpellRU**\ : texts collected from `LiveJournal <https://www.livejournal.com/media>`_\ , with manually corrected typos and errors;
* **MultidomainGold**\ : examples from 7 text sources, including the open web, news, social media, reviews, subtitles, policy documents and literary works;
* **MedSpellChecker**\ : texts with errors from medical anamnesis;
* **GitHubTypoCorpusRu**\ : spelling errors and typos in commits from GitHub;
* **BEA60K**\ : English spelling errors collected from several domains;
* **JFLEG**\ : 1601 sentences in English, which contain about 2 thousand spelling errors;

📈 Here we report evaluation of some setups:


* Zero-shot evaluation of pre-trained checkpoints;
* Additional fine-tuning (ft.) on the target dataset;

Full list of setups and corresponding performances are in the `Paper <https://aclanthology.org/2024.findings-eacl.10/>`_.

**RUSpellRU**, **MultidomainGold**, **MedSpellChecker** and **GitHubTypoCorpusRu** come from `spellcheck_punctuation_benchmark <https://huggingface.co/datasets/ai-forever/spellcheck_punctuation_benchmark>`_.
The benchmark accounts for both punctuation and spelling errors. For the simplicity and better representativeness we report results only for those models
(`sage-fredt5-large <https://huggingface.co/ai-forever/sage-fredt5-large>`_, `sage-fredt5-distilled-95m <https://huggingface.co/ai-forever/sage-fredt5-distilled-95m>`_) that deal with both types of errors (the Russian language).
The detailed metrics for other checkpoints can be found either in the `Paper <https://aclanthology.org/2024.findings-eacl.10/>`_, `post <ссылка на новый хабр>`_ or corresponding model card.


*NOTE:* **MedSpellChecker** and **GitHubTypoCorpusRu** do not have train split, so their performance on 
**Pre-train + fine-tune** setup is reported as a result of fine-tuning on combination of **RUSpellRU** and **MultidomainGold**
datasets.

**RUSpellRU Evaluation**

+---------------------------------+-------------+--------------+------------+------------+-------------+-----------+------------+-------------+-----------+
| Model                           | Pr. (spell) | Rec. (spell) | F1 (spell) | Pr. (punc) | Rec. (punc) | F1 (punc) | Pr. (case) | Rec. (case) | F1 (case) |
+=================================+=============+==============+============+============+=============+===========+============+=============+===========+
| sage-ai-service                 | 90.3        | 86.3         | 88.2       | 90.3       | 86.6        | 88.4      | 95.2       | 95.9        | 95.6      |
+---------------------------------+-------------+--------------+------------+------------+-------------+-----------+------------+-------------+-----------+
| sage-fredt5-large               | 57.3        | 68.0         | 62.2       | 86.7       | 46.1        | 60.2      | 92.1       | 67.8        | 78.1      |
+---------------------------------+-------------+--------------+------------+------------+-------------+-----------+------------+-------------+-----------+
| sage-fredt5-large (ft.)         | 88.4        | 80.9         | 84.5       | 88.2       | 85.3        | 86.8      | 95.5       | 94.0        | 94.7      |
+---------------------------------+-------------+--------------+------------+------------+-------------+-----------+------------+-------------+-----------+
| sage-fredt5-distilled-95m (ft.) | 83.5        | 74.8         | 78.9       | 86.8       | 80.6        | 83.6      | 94.4       | 92.5        | 93.5      |
+---------------------------------+-------------+--------------+------------+------------+-------------+-----------+------------+-------------+-----------+
| gpt-3.5-turbo                   | 33.6        | 58.5         | 42.7       | 85.9       | 64.6        | 73.7      | 84.9       | 73.9        | 79.0      |
+---------------------------------+-------------+--------------+------------+------------+-------------+-----------+------------+-------------+-----------+
| gpt-4                           | 54.9        | 76.7         | 64.0       | 84.0       | 82.3        | 83.2      | 91.5       | 90.2        | 90.9      |
+---------------------------------+-------------+--------------+------------+------------+-------------+-----------+------------+-------------+-----------+

**MultidomainGold Evaluation**

+---------------------------------+-------------+--------------+------------+------------+-------------+-----------+------------+-------------+-----------+
| Model                           | Pr. (spell) | Rec. (spell) | F1 (spell) | Pr. (punc) | Rec. (punc) | F1 (punc) | Pr. (case) | Rec. (case) | F1 (case) |
+=================================+=============+==============+============+============+=============+===========+============+=============+===========+
| sage-ai-service                 | 81.6        | 77.7         | 79.6       | 70.2       | 67.5        | 68.8      | 80.5       | 80.5        | 80.5      |
+---------------------------------+-------------+--------------+------------+------------+-------------+-----------+------------+-------------+-----------+
| sage-fredt5-large               | 43.4        | 49.7         | 46.3       | 21.8       | 21.3        | 21.6      | 58.8       | 23.9        | 34.0      |
+---------------------------------+-------------+--------------+------------+------------+-------------+-----------+------------+-------------+-----------+
| sage-fredt5-large (ft.)         | 80.3        | 75.1         | 77.6       | 69.0       | 66.5        | 67.7      | 78.6       | 80.0        | 79.3      |
+---------------------------------+-------------+--------------+------------+------------+-------------+-----------+------------+-------------+-----------+
| sage-fredt5-distilled-95m (ft.) | 77.2        | 69.9         | 73.4       | 66.8       | 63.4        | 65.0      | 76.8       | 79.1        | 77.9      |
+---------------------------------+-------------+--------------+------------+------------+-------------+-----------+------------+-------------+-----------+
| gpt-3.5-turbo                   | 18.8        | 48.1         | 27.1       | 42.0       | 31.8        | 36.2      | 47.1       | 51.3        | 49.1      |
+---------------------------------+-------------+--------------+------------+------------+-------------+-----------+------------+-------------+-----------+
| gpt-4                           | 25.4        | 68.0         | 37.0       | 57.8       | 54.3        | 56.0      | 54.0       | 67.5        | 60.0      |
+---------------------------------+-------------+--------------+------------+------------+-------------+-----------+------------+-------------+-----------+

**MedSpellchecker Evaluation**

+---------------------------------+-------------+--------------+------------+------------+-------------+-----------+------------+-------------+-----------+
| Model                           | Pr. (spell) | Rec. (spell) | F1 (spell) | Pr. (punc) | Rec. (punc) | F1 (punc) | Pr. (case) | Rec. (case) | F1 (case) |
+=================================+=============+==============+============+============+=============+===========+============+=============+===========+
| sage-ai-service                 | 71.3        | 73.5         | 72.4       | 75.1       | 69.2        | 72.0      | 80.9       | 72.8        | 76.6      |
+---------------------------------+-------------+--------------+------------+------------+-------------+-----------+------------+-------------+-----------+
| sage-fredt5-large               | 35.2        | 54.5         | 42.8       | 19.2       | 13.2        | 15.7      | 48.7       | 36.8        | 41.9      |
+---------------------------------+-------------+--------------+------------+------------+-------------+-----------+------------+-------------+-----------+
| sage-fredt5-large (ft.)         | 72.5        | 72.2         | 72.3       | 74.6       | 66.4        | 70.3      | 79.3       | 85.1        | 82.1      |
+---------------------------------+-------------+--------------+------------+------------+-------------+-----------+------------+-------------+-----------+
| sage-fredt5-distilled-95m (ft.) | 65.1        | 64.8         | 64.9       | 78.6       | 63.1        | 70.0      | 63.5       | 74.7        | 68.7      |
+---------------------------------+-------------+--------------+------------+------------+-------------+-----------+------------+-------------+-----------+
| gpt-3.5-turbo                   | 14.7        | 45.9         | 22.3       | 69.9       | 52.3        | 59.8      | 26.4       | 41.8        | 32.3      |
+---------------------------------+-------------+--------------+------------+------------+-------------+-----------+------------+-------------+-----------+
| gpt-4                           | 37.8        | 72.3         | 49.6       | 81.4       | 64.3        | 71.9      | 73.0       | 62.1        | 67.1      |
+---------------------------------+-------------+--------------+------------+------------+-------------+-----------+------------+-------------+-----------+

**GitHubTypoCorpusRu Evaluation**

+---------------------------------+-------------+--------------+------------+------------+-------------+-----------+------------+-------------+-----------+
| Model                           | Pr. (spell) | Rec. (spell) | F1 (spell) | Pr. (punc) | Rec. (punc) | F1 (punc) | Pr. (case) | Rec. (case) | F1 (case) |
+=================================+=============+==============+============+============+=============+===========+============+=============+===========+
| sage-ai-service                 | 70.8        | 56.3         | 62.7       | 48.9       | 35.8        | 41.4      | 32.9       | 45.3        | 38.1      |
+---------------------------------+-------------+--------------+------------+------------+-------------+-----------+------------+-------------+-----------+
| sage-fredt5-large               | 46.0        | 46.6         | 46.3       | 22.7       | 18.3        | 20.2      | 12.0       | 13.2        | 12.6      |
+---------------------------------+-------------+--------------+------------+------------+-------------+-----------+------------+-------------+-----------+
| sage-fredt5-large (ft.)         | 67.5        | 53.2         | 59.5       | 48.5       | 38.0        | 42.6      | 37.3       | 50.0        | 42.7      |
+---------------------------------+-------------+--------------+------------+------------+-------------+-----------+------------+-------------+-----------+
| sage-fredt5-distilled-95m (ft.) | 57.8        | 48.5         | 52.7       | 45.2       | 39.5        | 42.1      | 29.9       | 46.2        | 36.3      |
+---------------------------------+-------------+--------------+------------+------------+-------------+-----------+------------+-------------+-----------+
| gpt-3.5-turbo                   | 23.7        | 38.7         | 29.4       | 37.6       | 23.3        | 28.7      | 19.6       | 35.9        | 25.3      |
+---------------------------------+-------------+--------------+------------+------------+-------------+-----------+------------+-------------+-----------+
| gpt-4                           | 27.0        | 52.8         | 35.7       | 45.9       | 32.6        | 38.2      | 25.7       | 36.8        | 30.2      |
+---------------------------------+-------------+--------------+------------+------------+-------------+-----------+------------+-------------+-----------+

**BEA60K Evaluation**

+---------------------------------------------------+-----------+--------+------+
| Model                                             | Precision | Recall | F1   |
+===================================================+===========+========+======+
| sage-mt5-large                                    | 64.7      | 83.8   | 73.0 |
+---------------------------------------------------+-----------+--------+------+
| T5-large-spell                                    | 66.5      | 83.1   | 73.9 |
+---------------------------------------------------+-----------+--------+------+
| gpt-3.5-turbo                                     | 66.9      | 84.1   | 74.5 |
+---------------------------------------------------+-----------+--------+------+
| gpt-4                                             | 68.6      | 85.2   | 76.0 |
+---------------------------------------------------+-----------+--------+------+
| `Bert <https://github.com/neuspell/neuspell>`_    | 65.8      | 79.6   | 72.0 |
+---------------------------------------------------+-----------+--------+------+
| `SC-LSTM <https://github.com/neuspell/neuspell>`_ | 62.2      | 80.3   | 72.0 |
+---------------------------------------------------+-----------+--------+------+

**JFLEG Evaluation**

+---------------------------------------------------+-----------+--------+------+
| Model                                             | Precision | Recall | F1   |
+===================================================+===========+========+======+
| sage-mt5-large                                    | 74.9      | 88.4   | 81.1 |
+---------------------------------------------------+-----------+--------+------+
| T5-large-spell                                    | 83.4      | 84.3   | 83.8 |
+---------------------------------------------------+-----------+--------+------+
| gpt-3.5-turbo                                     | 77.8      | 88.6   | 82.9 |
+---------------------------------------------------+-----------+--------+------+
| gpt-4                                             | 77.9      | 88.3   | 82.8 |
+---------------------------------------------------+-----------+--------+------+
| `Bert <https://github.com/neuspell/neuspell>`_    | 78.5      | 85.4   | 81.8 |
+---------------------------------------------------+-----------+--------+------+
| `SC-LSTM <https://github.com/neuspell/neuspell>`_ | 80.6      | 86.1   | 83.2 |
+---------------------------------------------------+-----------+--------+------+

**RUSpellRU**, **MultidomainGold**, **MedSpellChecker** and **GitHubTypoCorpusRu** are available as HuggingFace datasets `here <https://huggingface.co/datasets/ai-forever/spellcheck_punctuation_benchmark>`_ and through the API of our library:

.. code-block:: python

    from sage.utils import load_available_dataset_from_hf, DatasetsAvailable

    print([dataset.name for dataset in DatasetsAvailable])
    # ['MultidomainGold', 'RUSpellRU', 'MedSpellchecker', 'GitHubTypoCorpusRu', 'MultidomainGold_orth', 'RUSpellRU_orth', 'MedSpellchecker_orth', 'GitHubTypoCorpusRu_orth']

    gold_dataset = load_available_dataset_from_hf(DatasetsAvailable.MultidomainGold.name, for_labeler=False)
    print(len(gold_dataset))
    # 7675

    sources, corrections = load_available_dataset_from_hf(DatasetsAvailable.RUSpellRU.name, for_labeler=True, split="train")
    print(len(sources), len(corrections))
    # 2000 2000

Evaluation
----------

We also provide functionality to evaluate the performance of spelling correction systems and rank them. 

🎯 Currently two options are available:

* `ruspelleval <https://www.dialog-21.ru/media/3427/sorokinaaetal.pdf>`_;
* `ERRANT <https://github.com/chrisjbryant/errant>`_-based metric adapted for the Russian language;

Both algorithms output Precision, Recall and F1 scores that can be interpreted like the following:

* **Precision**: one minus share of unnecessary amendments;
* **Recall**: proportion of expected corrections;
* **F1**: famous geometric mean of aforementioned two;

You can obtain these metrics simply by

.. code-block:: python

    from sage.evaluation import Scorer
    from sage.utils import DatasetsAvailable, load_available_dataset_from_hf

    sources, corrections = load_available_dataset_from_hf(DatasetsAvailable.RUSpellRU.name, for_labeler=True, split="test")

    scorer = Scorer()
    metrics = scorer.score(sources, corrections, corrections, metrics=["ruspelleval", "errant"])
    print(metrics)
    # {'Precision': 100.0, 'Recall': 100.0, 'F1': 100.0, 'CASE_Precision': 100.0, 'CASE_Recall': 100.0, 'CASE_F1': 100.0, 'SPELL_Precision': 100.0, 'SPELL_Recall': 100.0, 'SPELL_F1': 100.0, 'PUNCT_Precision': 100.0, 'PUNCT_Recall': 100.0, 'PUNCT_F1': 100.0, 'YO_Precision': 100.0, 'YO_Recall': 100.0, 'YO_F1': 100.0}

... or by directly assessing the model:

.. code-block:: python

    import os
    import torch
    from sage.utils import DatasetsAvailable
    from sage.spelling_correction import AvailableCorrectors
    from sage.spelling_correction import T5ModelForSpellingCorruption

    corrector_fred_95m = T5ModelForSpellingCorruption.from_pretrained(AvailableCorrectors.sage_fredt5_distilled_95m.value)
    corrector_mt5 = T5ModelForSpellingCorruption.from_pretrained(AvailableCorrectors.sage_mt5_large.value)

    corrector_fred_95m.model.to(torch.device("cuda:0"))
    corrector_mt5.model.to(torch.device("cuda:0"))

    metrics = corrector_fred_95m.evaluate("RUSpellRU", metrics=["errant", "ruspelleval"], batch_size=32)
    print(metrics)
    # {'CASE_Precision': 94.41, 'CASE_Recall': 92.55, 'CASE_F1': 93.47, 'SPELL_Precision': 77.52, 'SPELL_Recall': 64.09, 'SPELL_F1': 70.17, 'PUNCT_Precision': 86.77, 'PUNCT_Recall': 80.59, 'PUNCT_F1': 83.56, 'YO_Precision': 46.21, 'YO_Recall': 73.83, 'YO_F1': 56.84, 'Precision': 83.48, 'Recall': 74.75, 'F1': 78.87}

    metrics = corrector_mt5.evaluate("/content/sage/data/example_data/jfleg", metrics=["ruspelleval"], batch_size=16)
    print(metrics)
    # {'Precision': 75.94, 'Recall': 88.15, 'F1': 81.59}

The metrics output by ERRANT based algorithm are indicated by the corresponding prefix, which refers to the specific type of errors:

* *CASE*: erroneously used case;
* *SPELL*: spelling and grammar errors;
* *PUNCT*: punctuation errors;
* *YO*: unnecessary replacement of "YO" (ё) letter;

📌 Credit for evaluation script goes to Aleksei Sorokin and his notable `work <https://www.dialog-21.ru/media/3427/sorokinaaetal.pdf>`_ 
in proceedings of `SpellRueval <https://www.dialog-21.ru/evaluation/2016/spelling_correction/>`_. 

Citation
--------

If you want to know more about our work take a look at these publications:

💥 Our first `Paper <https://aclanthology.org/2024.findings-eacl.10/>`_ provides a thorough description of the methodology used to obtain SOTA
models for spelling corrections as well the comprehensive reports of all experiments that have been carried out. 

💫 While our Dialogue-2023 `Paper <https://www.dialog-21.ru/media/5914/martynovnplusetal056.pdf>`_ focuses on exploiting 
resources for the task of spelling correction and procedures on obtaining high-quality parallel corpuses. 

.. code-block::

    @inproceedings{martynov-etal-2024-methodology,
        title = "A Methodology for Generative Spelling Correction via Natural Spelling Errors Emulation across Multiple Domains and Languages",
        author = "Martynov, Nikita  and
          Baushenko, Mark  and
          Kozlova, Anastasia  and
          Kolomeytseva, Katerina  and
          Abramov, Aleksandr  and
          Fenogenova, Alena",
        editor = "Graham, Yvette  and
          Purver, Matthew",
        booktitle = "Findings of the Association for Computational Linguistics: EACL 2024",
        month = mar,
        year = "2024",
        address = "St. Julian{'}s, Malta",
        publisher = "Association for Computational Linguistics",
        url = "https://aclanthology.org/2024.findings-eacl.10",
        pages = "138--155",
        abstract = "Large language models excel in text generation and generalization, however they face challenges in text editing tasks, especially in correcting spelling errors and mistyping.In this paper, we present a methodology for generative spelling correction (SC), tested on English and Russian languages and potentially can be extended to any language with minor changes. Our research mainly focuses on exploring natural spelling errors and mistyping in texts and studying how those errors can be emulated in correct sentences to enrich generative models{'} pre-train procedure effectively. We investigate the effects of emulations in various text domains and examine two spelling corruption techniques: 1) first one mimics human behavior when making a mistake through leveraging statistics of errors from a particular dataset, and 2) second adds the most common spelling errors, keyboard miss clicks, and some heuristics within the texts.We conducted experiments employing various corruption strategies, models{'} architectures, and sizes in the pre-training and fine-tuning stages and evaluated the models using single-domain and multi-domain test sets. As a practical outcome of our work, we introduce SAGE (Spell checking via Augmentation and Generative distribution Emulation).",
    }

   @inproceedings{martynov2023augmentation,
     title={Augmentation methods for spelling corruptions},
     author={Martynov, Nikita and Baushenko, Mark and Abramov, Alexander and Fenogenova, Alena},
     booktitle={Proceedings of the International Conference “Dialogue},
     volume={2023},
     year={2023}
   }

📌 Feel free to ask any questions regarding our work at corresponding point of contact:

*nikita.martynov.98@list.ru*

.. toctree::
   :caption: Datasets
   :hidden:

   rst/datasets/RUSpellRU.rst
   rst/datasets/MultidomainGold.rst
   rst/datasets/MedSpellchecker.rst
   rst/datasets/GitHubTypoCorpusRu.rst

.. toctree::
   :caption: Models
   :hidden:

   rst/spelling_correction/M2M100-1.2B.rst
   rst/spelling_correction/M2M100-418M.rst
   rst/spelling_correction/FredT5-large.rst
   rst/spelling_correction/T5.rst

.. toctree::
   :caption: Augmentation
   :hidden:

   rst/spelling_corruption/SBSC.rst
   rst/spelling_corruption/Augmentex.rst

.. toctree::
   :caption: Evaluation
   :hidden:

   rst/evaluation/Errant.rst
   rst/evaluation/WOS.rst

