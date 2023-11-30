from .loaders import (
    DAVISDataset,
    MNISTDataset,
    UCF101Dataset,
    VimeoDataset,
    YouTubeVOSDataset,
    X4KDataset,
)

import pytorch_lightning as pl
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from torch.utils.data import DataLoader, Dataset

import warnings

warnings.filterwarnings("ignore", category=UserWarning)


class CollectiveDataloader(pl.LightningDataModule):
    def __init__(self, datasets, dataloader_params):
        super().__init__()
        self.train_set = CollectiveDataset(
            datasets, dataloader_params, split="train"
        ).datasets
        self.validation_set = CollectiveDataset(
            datasets, dataloader_params, "validation"
        ).datasets

    def train_dataloader(self):
        return self.train_set
        # return CombinedLoader(self.train_set, "sequential")

    def val_dataloader(self):
        return self.validation_set
        # return CombinedLoader(self.validation_set, "sequential")


class CollectiveDataset(Dataset):
    dataset_loaders = {
        "mnist": MNISTDataset,
        "kinetics": UCF101Dataset,
        "davis": DAVISDataset,
        "vimeo": VimeoDataset,
        "vimeo_septuplet": VimeoDataset,
        "youtube": YouTubeVOSDataset,
        "x4": X4KDataset,
    }

    def __init__(self, datasets, dataloader_params, split="train"):
        loaded_datasets = {}
        for dataset, info in datasets.items():
            if dataset not in self.dataset_loaders:
                print(f"Skipping {dataset} dataset, could not be found")
                continue
            if split == "validation" and dataset in {
                "youtube",
                "x4",
                "vimeo_septuplet",
            }:
                print(f"Skipping validation for {dataset}, does not exist!")
                continue
            loaded_datasets[dataset] = DataLoader(
                self.dataset_loaders[dataset](**info, split=split),
                **dataloader_params[dataset],
            )
        self.datasets = loaded_datasets
