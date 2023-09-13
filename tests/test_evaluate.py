from sage.utils import load_available_dataset_from_hf, DatasetsAvailable
from sage.evaluation.evaluate import evaluation


if __name__ == "__main__":
    ruspell_sources, ruspell_corrections = load_available_dataset_from_hf(
        DatasetsAvailable.RUSpellRU.name, for_labeler=True, split="test")
    gold_sources, gold_corrections = load_available_dataset_from_hf(
        DatasetsAvailable.MultidomainGold.name, for_labeler=True, split="test")
    med_sources, med_corrections = load_available_dataset_from_hf(
        DatasetsAvailable.MedSpellchecker.name, for_labeler=True, split="test")
    git_sources, git_corrections = load_available_dataset_from_hf(
        DatasetsAvailable.GitHubTypoCorpusRu.name, for_labeler=True, split="test")

    all_win_metrics = evaluation(ruspell_sources, ruspell_corrections, ruspell_corrections)
    print("Precision={Precision} Recall={Recall} FMeasure={F1}".format(**all_win_metrics))
    all_lose_metrics = evaluation(ruspell_sources, ruspell_corrections, ruspell_sources[:-1] + [ruspell_corrections[-1]])
    print("Precision={Precision} Recall={Recall} FMeasure={F1}".format(**all_lose_metrics))

