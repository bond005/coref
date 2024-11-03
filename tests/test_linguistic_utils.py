import os
import sys
import unittest

from pymorphy3.analyzer import Parse

try:
    from linguistic_utils.linguistic_utils import parse_text, find_best_parsing
    from linguistic_utils.linguistic_utils import find_main_token, inflect_subphrase
    from linguistic_utils.linguistic_utils import initialize_nlp
except:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from linguistic_utils.linguistic_utils import parse_text, find_best_parsing
    from linguistic_utils.linguistic_utils import find_main_token, inflect_subphrase
    from linguistic_utils.linguistic_utils import initialize_nlp


class TestLinguisticUtils(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.nlp, cls.morph = initialize_nlp()

    def test_find_best_parsing_01(self):
        word = 'стали'
        pos = 'NOUN'
        case = 'gent'
        number = 'sing'
        variant = find_best_parsing(self.morph.parse(word), pos, case, number)
        self.assertIsInstance(variant, Parse)
        self.assertEqual(variant.normal_form, 'сталь')

    def test_find_best_parsing_02(self):
        word = 'стали'
        pos = 'VERB'
        case = ''
        number = ''
        variant = find_best_parsing(self.morph.parse(word), pos, case, number)
        self.assertIsInstance(variant, Parse)
        self.assertEqual(variant.normal_form, 'стать')

    def test_parse_text(self):
        s = 'Новосибирский государственный университет находится в Академгородке.'
        true_token_bounds = [(0, 13), (14, 29), (30, 41), (42, 51), (52, 53), (54, 67), (67, 68)]
        true_lemmas = ['новосибирский', 'государственный', 'университет', 'находиться', 'в', 'академгородок', '.']
        parsed = parse_text(s, self.nlp, self.morph)
        self.assertIsInstance(parsed, list)
        self.assertEqual(len(parsed), len(true_token_bounds))
        for idx, val in enumerate(parsed):
            self.assertIsInstance(val, tuple)
            self.assertEqual(len(val), 3)
            self.assertIsInstance(val[0], int)
            self.assertIsInstance(val[1], int)
            self.assertIsInstance(val[2], Parse)
            self.assertEqual(val[0], true_token_bounds[idx][0])
            self.assertEqual(val[1], true_token_bounds[idx][1])
            self.assertEqual(val[2].normal_form, true_lemmas[idx])

    def test_find_main_token_01(self):
        s = 'Новосибирский государственный университет'
        true_main_token = 2
        res = find_main_token(s, self.nlp)
        self.assertIsInstance(res, tuple)
        self.assertEqual(len(res), 2)
        self.assertIsInstance(res[0], int)
        self.assertIsInstance(res[1], bool)
        self.assertEqual(res[0], true_main_token)
        self.assertTrue(res[1])

    def test_find_main_token_02(self):
        s = 'Институт теплофизики Сибирского отделения РАН'
        true_main_token = 0
        res = find_main_token(s, self.nlp)
        self.assertIsInstance(res, tuple)
        self.assertEqual(len(res), 2)
        self.assertIsInstance(res[0], int)
        self.assertIsInstance(res[1], bool)
        self.assertEqual(res[0], true_main_token)
        self.assertTrue(res[1])

    def test_find_main_token_03(self):
        s = 'Сибирского государственного университета телекоммуникаций и информатики'
        true_main_token = 2
        res = find_main_token(s, self.nlp)
        self.assertIsInstance(res, tuple)
        self.assertEqual(len(res), 2)
        self.assertIsInstance(res[0], int)
        self.assertIsInstance(res[1], bool)
        self.assertEqual(res[0], true_main_token)
        self.assertTrue(res[1])

    def test_find_main_token_04(self):
        s = 'В Новосибирском государственном университете учится много студентов.'
        true_main_token = 4
        res = find_main_token(s, self.nlp)
        self.assertIsInstance(res, tuple)
        self.assertEqual(len(res), 2)
        self.assertIsInstance(res[0], int)
        self.assertIsInstance(res[1], bool)
        self.assertEqual(res[0], true_main_token)
        self.assertFalse(res[1])

    def test_inflect_subphrase_01(self):
        s = 'В Новосибирском государственном университете учится много студентов.'
        true_transformation = 'Новосибирские государственные университеты'
        tokens = parse_text(s, self.nlp, self.morph)
        predicted_transformation = inflect_subphrase(s, tokens, 1, 4, self.nlp,
                                                     'nomn', 'plur')
        self.assertIsInstance(predicted_transformation, tuple)
        self.assertEqual(len(predicted_transformation), 2)
        self.assertIsInstance(predicted_transformation[0], str)
        self.assertIsInstance(predicted_transformation[1], bool)
        self.assertTrue(predicted_transformation[1])
        self.assertEqual(predicted_transformation[0], true_transformation)

    def test_inflect_subphrase_02(self):
        s = 'В Новосибирском государственном университете учится много студентов.'
        true_transformation = 'учится много студентов'
        tokens = parse_text(s, self.nlp, self.morph)
        predicted_transformation = inflect_subphrase(s, tokens, 4, 7, self.nlp,
                                                     'nomn', 'plur')
        self.assertIsInstance(predicted_transformation, tuple)
        self.assertEqual(len(predicted_transformation), 2)
        self.assertIsInstance(predicted_transformation[0], str)
        self.assertIsInstance(predicted_transformation[1], bool)
        self.assertFalse(predicted_transformation[1])
        self.assertEqual(predicted_transformation[0], true_transformation)


if __name__ == '__main__':
    unittest.main(verbosity=2)
