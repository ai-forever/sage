import unittest
from sage.utils import DatasetsAvailable
from sage.pipeline import AugmentationPipeline, PipelineConfig


class TestAugmentationPipeline(unittest.TestCase):
    def setUp(self):
        self.pipeline_config = PipelineConfig()
        self.pipeline_config.set_char_params(min_aug=1, max_aug=3, unit_prob=0.2)
        self.pipeline_config.set_word_params(min_aug=1, max_aug=3, unit_prob=0.3)
        self.pipeline_config.set_sbsc_params(lang="ru",
                                             dataset_name_or_path=DatasetsAvailable.MedSpellchecker.name,
                                             dataset_split="test")
        self.sample_text = "Заметьте, не я это предложил!"

    def test_char_augmentation(self):
        pipeline = AugmentationPipeline(config=self.pipeline_config, shuffle=False)
        pipeline.add_char_augmenter()
        pipeline.set_order([0])  # Only CharAugmenter
        augmented_text = pipeline.augment(self.sample_text, seed=1)
        self.assertNotEqual(self.sample_text, augmented_text)
        print(f"Char Augmentation Result: {augmented_text}")

    def test_word_augmentation(self):
        pipeline = AugmentationPipeline(config=self.pipeline_config, shuffle=True)
        pipeline.add_word_augmenter()
        pipeline.set_order([0])  # Only WordAugmenter
        augmented_text = pipeline.augment(self.sample_text, seed=2)
        self.assertNotEqual(self.sample_text, augmented_text)
        print(f"Word Augmentation Result: {augmented_text}")

    def test_sbsc_augmentation(self):
        pipeline = AugmentationPipeline(config=self.pipeline_config, shuffle=False)
        pipeline.add_sbsc_augmenter()
        pipeline.set_order([0])  # Only SBSCorruptor
        augmented_text = pipeline.augment(self.sample_text, seed=1)
        self.assertNotEqual(self.sample_text, augmented_text)
        print(f"SBSC Augmentation Result: {augmented_text}")

    def test_all_augmentations(self):
        pipeline = AugmentationPipeline(config=self.pipeline_config, shuffle=True)
        pipeline.add_char_augmenter()
        pipeline.add_word_augmenter()
        pipeline.add_sbsc_augmenter()
        pipeline.set_order([0, 1, 2])  # All augmenters
        augmented_text = pipeline.augment(self.sample_text, seed=3)
        self.assertNotEqual(self.sample_text, augmented_text)
        print(f"All Augmentations Result: {augmented_text}")

    if __name__ == "__main__":
        unittest.main()
