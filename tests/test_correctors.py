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

if __name__ == "__main__":
    m2m_large_corrector = RuM2M100ModelForSpellingCorrection.from_pretrained(AvailableCorrectors.m2m100_1B.value)
    m2m_small_corrector = RuM2M100ModelForSpellingCorrection.from_pretrained(AvailableCorrectors.m2m100_418M.value)
    fred_corrector = T5ModelForSpellingCorruption.from_pretrained(AvailableCorrectors.fred_large.value)
    ent5_corrector = T5ModelForSpellingCorruption.from_pretrained(AvailableCorrectors.ent5_large.value)

    print("\n------------------------------------------------------------------\n")

    print("m2m_large_corrector: {}".format(m2m_large_corrector.correct(example_sentences_ru[0])))
    print("m2m_small_corrector: {}".format(m2m_small_corrector.correct(example_sentences_ru[0])))
    print("fred_corrector: {}".format(fred_corrector.correct(example_sentences_ru[0], prefix="Исправь: ")))
    print("ent5_corrector: {}".format(ent5_corrector.correct(example_sentences_en[0], prefix="grammar: ")))

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
