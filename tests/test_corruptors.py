import os

from sage.spelling_corruption import WordAugCorruptor, CharAugCorruptor, SBSCCorruptor
from sage.spelling_corruption import WordAugConfig, CharAugConfig, SBSCConfig

corrections = [
    "Я пошёл домой. Больше мне тут делать нечего",
    "Заметьте, - не я это предложил."
]

en_corrections = [
    "Father tell me: we get or we deserve",
    "Screw you guys, I am going home (c)."
]

if __name__ == "__main__":
    word_aug_default = WordAugCorruptor.from_default_config()
    word_aug_config = WordAugConfig(
        min_aug=2, max_aug=6, unit_prob=0.1
    )
    word_aug_custom = WordAugCorruptor.from_config(word_aug_config)

    char_aug_default = CharAugCorruptor.from_default_config()
    char_aug_config = CharAugConfig(
        min_aug=2, max_aug=6, unit_prob=0.1, mult_num=3
    )
    char_aug_custom = CharAugCorruptor.from_config(char_aug_config)

    sbsc_default = SBSCCorruptor.from_default_config()
    sbsc_config = SBSCConfig(
        reference_dataset_name_or_path="MedSpellchecker",
        reference_dataset_split="test"
    )
    sbsc_custom = SBSCCorruptor.from_config(sbsc_config)

    sbsc_config_en = SBSCConfig(
        lang="en",
        reference_dataset_name_or_path=os.path.join(os.getcwd(), "data", "example_data", "bea60k", "subsample")
    )
    sbsc_english = SBSCCorruptor.from_config(sbsc_config_en)

    print("\n------------------------------------------------------------------\n")

    print("word_aug_default: {}".format(word_aug_default.corrupt(corrections[0])))
    print("word_aug_custom: {}".format(word_aug_custom.corrupt(corrections[0])))
    print("char_aug_default: {}".format(char_aug_default.corrupt(corrections[0])))
    print("char_aug_custom: {}".format(char_aug_custom.corrupt(corrections[0])))
    print("sbsc_default: {}".format(sbsc_default.corrupt(corrections[0])))
    print("sbsc_custom: {}".format(sbsc_custom.corrupt(corrections[0])))
    print("sbsc_english: {}".format(sbsc_english.corrupt(en_corrections[0])))

    print("\n------------------------------------------------------------------\n")

    print("word_aug_default: \n{}\n".format("\n".join(word_aug_default.batch_corrupt(corrections, batch_prob=0.5))))
    print("word_aug_custom: \n{}\n".format("\n".join(word_aug_custom.batch_corrupt(corrections))))
    print("char_aug_default: \n{}\n".format("\n".join(char_aug_default.batch_corrupt(corrections, batch_prob=0.5))))
    print("char_aug_custom: \n{}\n".format("\n".join(char_aug_custom.batch_corrupt(corrections))))
    print("sbsc_default: \n{}\n".format("\n".join(sbsc_default.batch_corrupt(corrections))))
    print("sbsc_custom: \n{}\n".format("\n".join(sbsc_custom.batch_corrupt(corrections))))
    print("sbsc_english: {}".format("\n".join(sbsc_english.batch_corrupt(en_corrections))))
