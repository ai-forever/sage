from sage.pipeline import PipelineConfig
from sage.pipeline import AugmentationPipeline
from sage.utils import DatasetsAvailable


def simple_use():
    pipeline_config = PipelineConfig()
    pipeline = AugmentationPipeline(config=pipeline_config)
    sample_text = "Заметьте, не я это предложил!"
    augmented_text = pipeline.augment(sample_text, seed=1)
    print(augmented_text)


def advanced_use():
    pipeline_config = PipelineConfig()

    pipeline_config.set_char_params(min_aug=2, max_aug=4, unit_prob=0.3)
    pipeline_config.set_sbsc_params(lang="ru", dataset_name_or_path=DatasetsAvailable.MedSpellchecker.name,
                                    dataset_split="test")

    pipeline = AugmentationPipeline(config=pipeline_config, shuffle=False)

    pipeline.add_char_augmentor()
    pipeline.add_sbsc_augmentor()

    sample_text = "Заметьте, не я это предложил!"
    augmented_text = pipeline.augment(sample_text, seed=1)
    print(augmented_text)


if __name__ == "__main__":
    simple_use()
    advanced_use()
