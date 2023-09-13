import unittest

from sage.utils import load_available_dataset_from_hf, DatasetsAvailable

datasets_available = [dataset.name for dataset in DatasetsAvailable]

DF2LENS = {
    "MultidomainGold": {"train": 3571, "test": 4107},
    "RUSpellRU": {"train": 2000, "test": 2008},
    "MedSpellchecker": {"test": 1054},
    "GitHubTypoCorpusRu": {"test": 868},
}


class TestUtils(unittest.TestCase):
    def test_load_datasets_from_hf(self):
        for dataset_name in datasets_available:
            splits = list(DF2LENS[dataset_name].keys())
            for split in splits:
                dataset_split = load_available_dataset_from_hf(dataset_name, split=split, for_labeler=False)
                self.assertEqual(len(dataset_split), DF2LENS[dataset_name][split])
                sources, corrections = load_available_dataset_from_hf(dataset_name, split=split, for_labeler=True)
                self.assertEqual(len(sources), DF2LENS[dataset_name][split])
                self.assertEqual(len(corrections), DF2LENS[dataset_name][split])
            dataset = load_available_dataset_from_hf(dataset_name, for_labeler=False)
            self.assertEqual(len(dataset), sum(DF2LENS[dataset_name].values()))


if __name__ == '__main__':
    unittest.main()
