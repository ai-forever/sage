✏️ Augmentex
-------------------

We implemented two methods for spelling corruption. **S**\ tatistic-\ **b**\ ased **S**\ pelling **C**\ orruption (\ **SBSC**\ ) aims
to mimic human behaviour when making an error. While `Augmentex <https://github.com/ai-forever/augmentex>`_ relies on rule-based heuristics and common
errors and mistypings especially those committed while typing text on a keyboard.

🚀 Both methods proved their effectiveness for spelling correction systems and celebrated substantial **performance gains**
fully reported in our `Paper <https://aclanthology.org/2024.findings-eacl.10/>`_.


**Augmentex** introduces rule-based and common statistic (empowered by `KartaSlov <https://kartaslov.ru>`_ project)
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
