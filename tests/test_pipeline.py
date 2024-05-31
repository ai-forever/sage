import unittest
from sage.utils import DatasetsAvailable
from sage.pipeline import AugmentationPipeline, PipelineConfig


class TestAugmentationPipeline(unittest.TestCase):

    def setUp(self):
        self.pipeline_config = PipelineConfig(
            char_min_aug=1,
            char_max_aug=3,
            char_unit_prob=0.2,
            word_min_aug=1,
            word_max_aug=3,
            word_unit_prob=0.3,
            sbsc_lang="ru",
            sbsc_reference_dataset_name_or_path=DatasetsAvailable.MedSpellchecker.name,
            sbsc_reference_dataset_split="test"
        )
        self.sample_text = "Заметьте, не я это предложил!"

    def test_char_augmentation(self):
        pipeline = AugmentationPipeline(config=self.pipeline_config, shuffle=False)
        pipeline.add_char_augmentor()
        pipeline.set_order([0])  # Только CharAugmentor
        augmented_text = pipeline.augment(self.sample_text, seed=1)
        self.assertNotEqual(self.sample_text, augmented_text)
        print(f"Char Augmentation Result: {augmented_text}")

    def test_word_augmentation(self):
        pipeline = AugmentationPipeline(config=self.pipeline_config, shuffle=False)
        pipeline.add_word_augmentor()
        pipeline.set_order([0])  # Только WordAugmentor
        augmented_text = pipeline.augment(self.sample_text, seed=1)
        self.assertNotEqual(self.sample_text, augmented_text)
        print(f"Word Augmentation Result: {augmented_text}")

    def test_sbsc_augmentation(self):
        pipeline = AugmentationPipeline(config=self.pipeline_config, shuffle=False)
        pipeline.add_sbsc_augmentor()
        pipeline.set_order([0])  # Только SBSCorruptor
        augmented_text = pipeline.augment(self.sample_text, seed=1)
        self.assertNotEqual(self.sample_text, augmented_text)
        print(f"SBSC Augmentation Result: {augmented_text}")

    def test_all_augmentations(self):
        pipeline = AugmentationPipeline(config=self.pipeline_config, shuffle=False)
        pipeline.add_char_augmentor()
        pipeline.add_word_augmentor()
        pipeline.add_sbsc_augmentor()
        pipeline.set_order([0, 1, 2])  # Все аугментаторы
        augmented_text = pipeline.augment(self.sample_text, seed=1)
        self.assertNotEqual(self.sample_text, augmented_text)
        print(f"All Augmentations Result: {augmented_text}")

    if __name__ == "__main__":
        unittest.main()
