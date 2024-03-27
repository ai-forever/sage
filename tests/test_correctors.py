from sage.spelling_correction import RuM2M100ModelForSpellingCorrection, T5ModelForSpellingCorruption
from sage.spelling_correction import AvailableCorrectors

example_sentences_ru = [
    "Я пшёл домой",
    "Очень классная тетка ктобы что не говорил."
]

example_sentences_en = [
    "Fathr telll me, we get or we dereve.",
    "Scrw you guys, I am goin homee. (c)"
]

example_sentences_ru_en = [
    "Перведи мне текст на аглиском: \"Screw you kuys, I am goin hme (c).",
    "\"Don't you went to go upstayers?\", - сказл мне както дед."
]

if __name__ == "__main__":
    m2m_large_corrector = RuM2M100ModelForSpellingCorrection.from_pretrained(AvailableCorrectors.m2m100_1B.value)
    m2m_small_corrector = RuM2M100ModelForSpellingCorrection.from_pretrained(AvailableCorrectors.m2m100_418M.value)
    fred_corrector = T5ModelForSpellingCorruption.from_pretrained(AvailableCorrectors.fred_large.value)
    ent5_corrector = T5ModelForSpellingCorruption.from_pretrained(AvailableCorrectors.ent5_large.value)

    sage_fredt5_large = T5ModelForSpellingCorruption.from_pretrained(AvailableCorrectors.sage_fredt5_large.value)
    sage_fredt5_distilled = T5ModelForSpellingCorruption.from_pretrained(
        AvailableCorrectors.sage_fredt5_distilled_95m.value)
    sage_mt5_large = T5ModelForSpellingCorruption.from_pretrained(AvailableCorrectors.sage_mt5_large.value)
    sage_m2m_100 = RuM2M100ModelForSpellingCorrection.from_pretrained(AvailableCorrectors.sage_m2m100_1B.value)

    print("\n------------------------------------------------------------------\n")

    print("m2m_large_corrector: {}".format(m2m_large_corrector.correct(example_sentences_ru[0])))
    print("m2m_small_corrector: {}".format(m2m_small_corrector.correct(example_sentences_ru[0])))
    print("fred_corrector: {}".format(fred_corrector.correct(example_sentences_ru[0], prefix="Исправь: ")))
    print("ent5_corrector: {}".format(ent5_corrector.correct(example_sentences_en[0], prefix="grammar: ")))

    print("sage_fredt5_large: {}".format(sage_fredt5_large.correct(example_sentences_ru[0], prefix="<LM>")))
    print("sage_fredt5_distilled: {}".format(sage_fredt5_distilled.correct(example_sentences_ru[0], prefix="<LM>")))
    print("sage_mt5_large: {}".format(sage_mt5_large.correct(example_sentences_ru_en[0])))
    print("sage_m2m_100: {}".format(sage_m2m_100.correct(example_sentences_ru[0])))

    print("\n------------------------------------------------------------------\n")

    print("\nm2m_large_corrector:\n")
    print("\n".join(["{}: {}".format(k, v[0]) for k, v in zip(
        example_sentences_ru, m2m_large_corrector.batch_correct(example_sentences_ru, 1))]))

    print("\nm2m_small_corrector: \n")
    print("\n".join(["{}: {}".format(k, v[0]) for k, v in zip(
        example_sentences_ru, m2m_small_corrector.batch_correct(example_sentences_ru, 1))]))

    print("\nfred_corrector: \n")
    print("\n".join(["{}: {}".format(k, v[0]) for k, v in zip(
        example_sentences_ru, fred_corrector.batch_correct(example_sentences_ru, 1, "Исправь: "))]))

    print("\nent5_corrector: \n")
    print("\n".join(["{}: {}".format(k, v[0]) for k, v in zip(
        example_sentences_en, ent5_corrector.batch_correct(example_sentences_en, 1, "grammar: "))]))

    print("\nsage_fredt5_large:\n")
    print("\n".join(["{}: {}".format(k, v[0]) for k, v in zip(
        example_sentences_ru, sage_fredt5_large.batch_correct(example_sentences_ru, 1, prefix="<LM>"))]))

    print("\nsage_fredt5_distilled: \n")
    print("\n".join(["{}: {}".format(k, v[0]) for k, v in zip(
        example_sentences_ru, sage_fredt5_distilled.batch_correct(example_sentences_ru, 1, prefix="<LM>"))]))

    print("\nsage_mt5_large: \n")
    print("\n".join(["{}: {}".format(k, v[0]) for k, v in zip(
        example_sentences_ru_en, sage_mt5_large.batch_correct(example_sentences_ru, 1))]))

    print("\nsage_m2m_100: \n")
    print("\n".join(["{}: {}".format(k, v[0]) for k, v in zip(
        example_sentences_ru, sage_m2m_100.batch_correct(example_sentences_en, 1))]))
