import os
import sys
import unittest

try:
    from io_utils.io_utils import load_rucoco, find_entity
except:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from io_utils.io_utils import load_rucoco, find_entity


class TestFindEntity(unittest.TestCase):
    def test_find_entity_01(self):
        bounds_of_entities = [(0, 10), (45, 55), (12, 34)]
        char_idx = 4
        true_entity_idx = 0
        found_idx = find_entity(bounds_of_entities, char_idx)
        self.assertIsInstance(found_idx, int)
        self.assertEqual(found_idx, true_entity_idx)

    def test_find_entity_02(self):
        bounds_of_entities = [(0, 10), (45, 55), (12, 34)]
        char_idx = 9
        true_entity_idx = 0
        found_idx = find_entity(bounds_of_entities, char_idx)
        self.assertIsInstance(found_idx, int)
        self.assertEqual(found_idx, true_entity_idx)

    def test_find_entity_03(self):
        bounds_of_entities = [(0, 10), (45, 55), (12, 34)]
        char_idx = 12
        true_entity_idx = 2
        found_idx = find_entity(bounds_of_entities, char_idx)
        self.assertIsInstance(found_idx, int)
        self.assertEqual(found_idx, true_entity_idx)

    def test_find_entity_04(self):
        bounds_of_entities = [(0, 10), (45, 55), (12, 34)]
        char_idx = 11
        true_entity_idx = -1
        found_idx = find_entity(bounds_of_entities, char_idx)
        self.assertIsInstance(found_idx, int)
        self.assertEqual(found_idx, true_entity_idx)

    def test_find_entity_05(self):
        bounds_of_entities = [(0, 10), (45, 55), (12, 34)]
        char_idx = 34
        true_entity_idx = -1
        found_idx = find_entity(bounds_of_entities, char_idx)
        self.assertIsInstance(found_idx, int)
        self.assertEqual(found_idx, true_entity_idx)


class TestLoadRuCoCo(unittest.TestCase):
    def test_loading(self):
        dataset_name = os.path.join(os.path.dirname(__file__), 'testdata', 'dataset')
        res = load_rucoco(dataset_name)
        self.assertIsInstance(res, list)
        self.assertEqual(len(res), 2)
        for sample in res:
            self.assertIsInstance(sample, tuple)
            self.assertEqual(len(sample), 2)
            self.assertIsInstance(sample[0], str)
            self.assertIsInstance(sample[1], list)
            self.assertGreater(len(sample[1]), 0)
            for chain in sample[1]:
                self.assertIsInstance(chain, list)
                self.assertGreater(len(chain), 0)
                for it in chain:
                    self.assertIsInstance(it, tuple)
                    self.assertEqual(len(it), 2)
                    self.assertIsInstance(it[0], int)
                    self.assertIsInstance(it[1], int)
                    self.assertGreaterEqual(it[0], 0)
                    self.assertGreater(it[1], it[0])
                    self.assertLessEqual(it[1], len(sample[0]))
                    self.assertEqual(sample[0][it[0]:it[1]], sample[0][it[0]:it[1]].strip())


if __name__ == '__main__':
    unittest.main(verbosity=2)
