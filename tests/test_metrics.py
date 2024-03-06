import unittest

from sage.evaluation.ruerrant_wrapper.scorer import RuErrantScorer
from sage.evaluation.scorer import Scorer


class TestRuErrantErrorAnnotation(unittest.TestCase):

    scorer = RuErrantScorer()

    def test_1w_spell_only(self):
        source = self.scorer.annotator.parse("карова")
        correction = self.scorer.annotator.parse("корова")
        errors_true = [
            [0, 1, "SPELL", "корова", 0],
        ]
        self.assertEqual(self.scorer.annotate_errors(source, correction), errors_true)

    def test_1w_spell_case(self):
        source = self.scorer.annotator.parse("карова")
        correction = self.scorer.annotator.parse("Корова")
        errors_true = [
            [0, 1, "CASE", "Карова", 0],
            [0, 1, "SPELL", "корова", 0],
        ]
        self.assertEqual(self.scorer.annotate_errors(source, correction), errors_true)

    def test_1w_spell_case_yo(self):
        source = self.scorer.annotator.parse("йемённ")
        correction = self.scorer.annotator.parse("Йемен")
        errors_true = [
            [0, 1, "CASE", "Йемённ", 0],
            [0, 1, "SPELL", "йемён", 0],
            [0, 1, "YO", "йеменн", 0],
        ]
        self.assertEqual(self.scorer.annotate_errors(source, correction), errors_true)

    def test_1w_spell_case_same_char(self):
        source = self.scorer.annotator.parse("иемен")
        correction = self.scorer.annotator.parse("Йемен")
        errors_true = [
            [0, 1, "CASE", "Иемен", 0],
            [0, 1, "SPELL", "йемен", 0],
        ]
        self.assertEqual(self.scorer.annotate_errors(source, correction), errors_true)

    def test_1w_spell_case_2x(self):
        source = self.scorer.annotator.parse("ПИРИДАЧА")
        correction = self.scorer.annotator.parse("Передача")
        errors_true = [
            [0, 1, "CASE", "Пиридача", 0],
            [0, 1, "SPELL", "ПЕРЕДАЧА", 0],
        ]
        self.assertEqual(self.scorer.annotate_errors(source, correction), errors_true)

    def test_inn_punct_hyphen2space(self):
        source = self.scorer.annotator.parse("как-будто")
        correction = self.scorer.annotator.parse("как будто")
        errors_true = [
            [0, 3, "SPELL", "как будто", 0],
        ]
        self.assertEqual(self.scorer.annotate_errors(source, correction), errors_true)

    def test_inn_punct_space2hyphen(self):
        source = self.scorer.annotator.parse("кто то")
        correction = self.scorer.annotator.parse("кто-то")
        errors_true = [
            [0, 2, "SPELL", "кто-то", 0],
        ]
        self.assertEqual(self.scorer.annotate_errors(source, correction), errors_true)

    def test_inn_punct_joint2space(self):
        source = self.scorer.annotator.parse("какбудто")
        correction = self.scorer.annotator.parse("как будто")
        errors_true = [
            [0, 1, "SPELL", "как будто", 0],
        ]
        self.assertEqual(self.scorer.annotate_errors(source, correction), errors_true)

    def test_inn_punct_space2joint(self):
        source = self.scorer.annotator.parse("золо то")
        correction = self.scorer.annotator.parse("золото")
        errors_true = [
            [0, 2, "SPELL", "золото", 0],
        ]
        self.assertEqual(self.scorer.annotate_errors(source, correction), errors_true)

    def test_inn_punct_joint2hyphen(self):
        source = self.scorer.annotator.parse("ктото")
        correction = self.scorer.annotator.parse("кто-то")
        errors_true = [
            [0, 1, "SPELL", "кто-то", 0],
        ]
        self.assertEqual(self.scorer.annotate_errors(source, correction), errors_true)

    def test_inn_punct_hyphen2joint(self):
        source = self.scorer.annotator.parse("золо-то")
        correction = self.scorer.annotator.parse("золото")
        errors_true = [
            [0, 3, "SPELL", "золото", 0],
        ]
        self.assertEqual(self.scorer.annotate_errors(source, correction), errors_true)

    def test_inn_punct_hyphen2comma(self):
        source = self.scorer.annotator.parse("Если и ответит кто-то правильно.")
        correction = self.scorer.annotator.parse("Если и ответит кто, то правильно.")
        errors_true = [
            [4, 5, "PUNCT", ",", 0],
        ]
        self.assertEqual(self.scorer.annotate_errors(source, correction), errors_true)

    def test_inn_punct_comma2hyphen(self):
        source = self.scorer.annotator.parse("Если и ответит кто, то правильно.")
        correction = self.scorer.annotator.parse("Если и ответит кто-то правильно.")
        errors_true = [
            [4, 5, "PUNCT", "-", 0],
        ]
        self.assertEqual(self.scorer.annotate_errors(source, correction), errors_true)

    def test_dot2multidot(self):
        source = self.scorer.annotator.parse("Человек.")
        correction = self.scorer.annotator.parse("Человек...")
        errors_true = [
            [1, 2, "PUNCT", "...", 0],
        ]
        self.assertEqual(self.scorer.annotate_errors(source, correction), errors_true)

    def test_multidot2dot(self):
        source = self.scorer.annotator.parse("Человек...")
        correction = self.scorer.annotator.parse("Человек.")
        errors_true = [
            [1, 2, "PUNCT", ".", 0],
        ]
        self.assertEqual(self.scorer.annotate_errors(source, correction), errors_true)

    def test_2w_spell_case(self):
        source = self.scorer.annotator.parse("рас два")
        correction = self.scorer.annotator.parse("Раз два")
        errors_true = [
            [0, 1, "CASE", "Рас", 0],
            [0, 1, "SPELL", "раз", 0],
        ]
        self.assertEqual(self.scorer.annotate_errors(source, correction), errors_true)

    def test_2w_spell_case_punct1(self):
        source = self.scorer.annotator.parse("рас два")
        correction = self.scorer.annotator.parse("Раз два!")
        errors_true = [
            [0, 1, "CASE", "Рас", 0],
            [0, 1, "SPELL", "раз", 0],
            [2, 2, "PUNCT", "!", 0],
        ]
        self.assertEqual(self.scorer.annotate_errors(source, correction), errors_true)

    def test_2w_spell_case_punct2(self):
        source = self.scorer.annotator.parse("рас два!")
        correction = self.scorer.annotator.parse("Раз два")
        errors_true = [
            [0, 1, "CASE", "Рас", 0],
            [0, 1, "SPELL", "раз", 0],
            [2, 3, "PUNCT", "", 0],
        ]
        self.assertEqual(self.scorer.annotate_errors(source, correction), errors_true)

    def test_sent1(self):
        source = self.scorer.annotator.parse("очень классная тетка ктобы что не говорил.")
        correction = self.scorer.annotator.parse("Очень классная тётка, кто бы что ни говорил.")
        errors_true = [
            [0, 1, "CASE", "Очень", 0],
            [2, 3, "YO", "тётка", 0],
            [3, 3, "PUNCT", ",", 0],
            [3, 4, "SPELL", "кто бы", 0],
            [5, 6, "SPELL", "ни", 0],
        ]
        self.assertEqual(self.scorer.annotate_errors(source, correction), errors_true)

    def test_sent2(self):
        source = self.scorer.annotator.parse("Может выгоднее втулку продать и купить колесо в сборе?")
        correction = self.scorer.annotator.parse("Может, выгоднее втулку продать и купить колесо в сборе?")
        errors_true = [
            [1, 1, "PUNCT", ",", 0],
        ]
        self.assertEqual(self.scorer.annotate_errors(source, correction), errors_true)

    def test_sent3(self):
        source = self.scorer.annotator.parse("Довольно большая часть пришедших сходила с дорожек и усаживалась на траву.")
        correction = self.scorer.annotator.parse("Довольно большая часть пришедших сходила с дорожек и усаживалась на траву.")
        errors_true = []
        self.assertEqual(self.scorer.annotate_errors(source, correction), errors_true)

    def test_sent4(self):
        source = self.scorer.annotator.parse("Симпатичнейшое шпионское устройство, такой себе гламурный фотоаппарат девушки Бонда - миниатюрная модель камеры Superheadz Clap Camera.")
        correction = self.scorer.annotator.parse("Симпатичнейшее шпионское устройство, такой себе гламурный фотоаппарат девушки Бонда, миниатюрная модель камеры Superheadz Clap Camera.")
        errors_true = [
            [0, 1, "SPELL", "Симпатичнейшее", 0],
            [10, 11, "PUNCT", ",", 0],
        ]
        self.assertEqual(self.scorer.annotate_errors(source, correction), errors_true)

    def test_sent5(self):
        source = self.scorer.annotator.parse("Ну не было поста, так небыло!")
        correction = self.scorer.annotator.parse("Ну не было поста, так не было.")
        errors_true = [
            [6, 7, "SPELL", "не было", 0],
            [7, 8, "PUNCT", ".", 0],
        ]
        self.assertEqual(self.scorer.annotate_errors(source, correction), errors_true)

    def test_sent6(self):
        source = self.scorer.annotator.parse("Хотя странно, когда я забирала к себе на выходные старого кота, который живет у родителей, да и собаку в придачу, то такого концерта мой кот не устраивал.")
        correction = self.scorer.annotator.parse("Хотя странно, когда я забирала к себе на выходные старого кота, который живёт у родителей, да и собаку в придачу, то такого концерта мой кот не устраивал.")
        errors_true = [
            [14, 15, "YO", "живёт", 0],
        ]
        self.assertEqual(self.scorer.annotate_errors(source, correction), errors_true)

    def test_sent7(self):
        source = self.scorer.annotator.parse("Думаю, что лет через 10 ретроспективно просматривать это будет мне невероятно интересно.")
        correction = self.scorer.annotator.parse("Думаю, что лет через 10 ретроспективно просматривать это будет мне невероятно интересно.")
        errors_true = []
        self.assertEqual(self.scorer.annotate_errors(source, correction), errors_true)

    def test_sent8(self):
        source = self.scorer.annotator.parse("Зато я считаю, что это будет полезно и для меня и для всех тех, кто меня окружает, ведь когда расстаешься с человеком на какое-то время, то многое становится прозрачным, я имею ввиду мы начинаем понимать какое место в нашей повседневности занимает этот человек...")
        correction = self.scorer.annotator.parse("Зато я считаю, что это будет полезно и для меня, и для всех тех, кто меня окружает. Ведь когда расстаешься с человеком на какое-то время, то многое становится прозрачным. Я имею в виду, мы начинаем понимать, какое место в нашей повседневности занимает этот человек.")
        errors_true = [
            [11, 11, "PUNCT", ",", 0],
            [19, 21, "SPELL", ". Ведь", 0],
            [35, 37, "SPELL", ". Я", 0],
            [38, 39, "SPELL", "в виду", 0],
            [39, 39, "PUNCT", ",", 0],
            [42, 42, "PUNCT", ",", 0],
            [50, 51, "PUNCT", ".", 0],
        ]
        self.assertEqual(self.scorer.annotate_errors(source, correction), errors_true)

    def test_sent9(self):
        source = self.scorer.annotator.parse("Пояним эту мысль.")
        correction = self.scorer.annotator.parse("Поясним эту мысль.")
        errors_true = [
            [0, 1, "SPELL", "Поясним", 0],
        ]
        self.assertEqual(self.scorer.annotate_errors(source, correction), errors_true)

    def test_sent10(self):
        source = self.scorer.annotator.parse("она прямо бурлит у меня в крови, тормошит какими-то советами, смотрит на меня из глаз моей дочки, что носит ее имя.")
        correction = self.scorer.annotator.parse("Она прямо бурлит у меня в крови, тормошит какими-то советами, смотрит на меня из глаз моей дочки, что носит её имя.")
        errors_true = [
            [0, 1, "CASE", "Она", 0],
            [24, 25, "YO", "её", 0],
        ]
        self.assertEqual(self.scorer.annotate_errors(source, correction), errors_true)

    def test_sent11(self):
        source = self.scorer.annotator.parse("Иногда мне сложно понять: как можно не любить своего ребенка.")
        correction = self.scorer.annotator.parse("Иногда мне сложно понять, как можно не любить своего ребенка.")
        errors_true = [
            [4, 5, "PUNCT", ",", 0]
        ]
        self.assertEqual(self.scorer.annotate_errors(source, correction), errors_true)

    def test_sent12(self):
        source = self.scorer.annotator.parse("в массе своей они конечно все оччччень милые )")
        correction = self.scorer.annotator.parse("В массе своей они, конечно, все очень милые.")
        errors_true = [
            [0, 1, "CASE", "В", 0],
            [4, 4, "PUNCT", ",", 0],
            [5, 5, "PUNCT", ",", 0],
            [6, 7, "SPELL", "очень", 0],
            [8, 9, "PUNCT", ".", 0],
        ]
        self.assertEqual(self.scorer.annotate_errors(source, correction), errors_true)


class TestScorerLoadingAndRunning(unittest.TestCase):

    scorer_without_errant = Scorer(load_errant=False)

    def test_score_metrics_format_exception1(self):
        """Empty `metrics`."""
        with self.assertRaises(ValueError):
            self.scorer_without_errant.score([""], [""], [""], [])

    def test_score_metrics_format_exception2(self):
        """Not a list of strings `metrics`."""
        with self.assertRaises(ValueError):
            self.scorer_without_errant.score([""], [""], [""], [1])

    def test_score_metrics_format_exception3(self):
        """Misspelled metric in `metrics`."""
        with self.assertRaises(ValueError):
            self.scorer_without_errant.score([""], [""], [""], ["word"])

    def test_score_metrics_format_exception4(self):
        """Calling "errant" metric while RuErrantScorer has not been loaded."""
        with self.assertRaises(AttributeError):
            self.scorer_without_errant.score([""], [""], [""], ["errant"])


class TestScorerEvaluation(unittest.TestCase):

    scorer_with_errant = Scorer(load_errant=True)

    def test_evaluation_empty(self):
        with self.assertRaises(ValueError):
            self.scorer_with_errant.score([""], [""], [""], metrics=["words"])

    def test_evaluation_words(self):
        res = self.scorer_with_errant.score(["карова"], ["Корова"], ["корова"], metrics=["words"])
        res_gold = {
            "Precision": 100,
            "Recall": 100,
            "F1": 100,
        }
        self.assertEqual(res, res_gold)

    def test_evaluation_errant_spell_case(self):
        res = self.scorer_with_errant.score(["карова"], ["Корова"], ["корова"], metrics=["errant"])
        res_gold = {
            "CASE_Precision": 100,
            "CASE_Recall": 0.0,
            "CASE_F1": 0.0,
            "PUNCT_Precision": 100.0,
            "PUNCT_Recall": 100.0,
            "PUNCT_F1": 100.0,
            "SPELL_Precision": 100,
            "SPELL_Recall": 100,
            "SPELL_F1": 100,
            "YO_Precision": 100.0,
            "YO_Recall": 100.0,
            "YO_F1": 100.0,
        }
        self.assertEqual(res, res_gold)

    def test_evaluation_errant_spell_case_punct(self):
        res = self.scorer_with_errant.score(
            ["в массе своей они конечно все оччччень милые )"],
            ["В массе своей они, конечно, все очень милые."],
            ["В массе своей они конечно все оччень милые."],
            metrics=["errant"])
        res_gold = {
            "CASE_Precision": 100.0,
            "CASE_Recall": 100.0,
            "CASE_F1": 100.0,
            "PUNCT_Precision": 100.0,
            "PUNCT_Recall": 33.33,
            "PUNCT_F1": 50.0,
            "SPELL_Precision": 0.0,
            "SPELL_Recall": 0.0,
            "SPELL_F1": 0.0,
            "YO_Precision": 100.0,
            "YO_Recall": 100.0,
            "YO_F1": 100.0,
        }
        self.assertEqual(res, res_gold)

    def test_evaluation_errant_spell_case_punct_yo(self):
        res = self.scorer_with_errant.score(
            ["спел Кейс ее .", "спел Кейс ее ."],
            ["спелл кейс её !", "спелл кейс её !"],
            ["спел кейс её .", "спелл Кейс ее !"],
            metrics=["errant"])
        res_gold = {
            "CASE_Precision": 100.0,
            "CASE_Recall": 50.0,
            "CASE_F1": 66.67,
            "YO_Precision": 100.0,
            "YO_Recall": 50.0,
            "YO_F1": 66.67,
            "SPELL_Precision": 100.0,
            "SPELL_Recall": 50.0,
            "SPELL_F1": 66.67,
            "PUNCT_Precision": 100.0,
            "PUNCT_Recall": 50.0,
            "PUNCT_F1": 66.67,
        }
        self.assertEqual(res, res_gold)


if __name__ == "__main__":
    unittest.main()
