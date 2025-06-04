import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader
import os
from .baseLoader import BaseDatasetReader


class MitbihECG(Dataset):
    def __init__(self, signals: np.ndarray, labels: np.ndarray):
        self.X = torch.tensor(signals, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class Mitbih_ReadDataset(BaseDatasetReader):
    def __init__(
        self,
        dataset_path: str,
        labels: List[str],
        data_shape: Tuple[int, int, int],
        batch_size: int = 10,
        dataset_class=MitbihECG,
        is_s3: bool = False,
        s3_region_name=None,
        s3_endpoint_url=None,
        s3_access_key_id=None,
        s3_secret_access_key=None,
    ):
        """
        :param dataset_path: Path to the CSV file or S3 URI (if is_s3=True).
        :param labels: List of class names to include (e.g., ["Non-Ectopic Beats", ...]).
        :param data_shape: Tuple for reshaping signals. Use (1, 187) or (1, 1, 187).
        :param batch_size: Batch size for DataLoader.
        :param dataset_class: Dataset wrapper class.
        :param is_s3: If True, load from S3 and unzip locally.
        """
        super().__init__(
            dataset_path,
            labels,
            data_shape,
            batch_size,
            dataset_class,
            is_s3=is_s3,
            s3_region_name=s3_region_name,
            s3_endpoint_url=s3_endpoint_url,
            s3_access_key_id=s3_access_key_id,
            s3_secret_access_key=s3_secret_access_key,
        )

        self.cls_dict = {
            "Non-Ectopic Beats": 0,
            "Superventrical Ectopic": 1,
            "Ventricular Beats": 2,
            "Unknown": 3,
            "Fusion Beats": 4,
        }

        self._load_and_process()

    def _load_and_process(self):
        """
        Carica e processa i dati ECG dal file CSV.
        Filtra i dati in base alle classi selezionate.
        """
        # Trova il file CSV nella directory (necessario se proveniente da uno ZIP)
        if os.path.isdir(self.dataset_path):
            csv_files = [f for f in os.listdir(self.dataset_path) if f.endswith(".csv")]
            if not csv_files:
                raise FileNotFoundError("Nessun file CSV trovato nel dataset_path.")
            csv_path = os.path.join(self.dataset_path, csv_files[0])
        else:
            csv_path = self.dataset_path

        data = pd.read_csv(csv_path, header=None)

        selected_classes = [self.cls_dict[label] for label in self.labels]
        data = data[data[187].isin(selected_classes)]

        self.class_counts = {
            label: len(data[data[187] == self.cls_dict[label]]) for label in self.labels
        }

        X = data.iloc[:, :-1].values
        y = data[187].values

        if self.data_shape == (1, 187):
            X = X.reshape(-1, 1, 187)
        elif self.data_shape == (1, 1, 187):
            X = X.reshape(-1, 1, 1, 187)
        else:
            raise ValueError(f"Unsupported data_shape: {self.data_shape}")

        self.y = y
        self.X = X
        self.dataset = self.dataset_class(X, y)

    def load_data(self) -> Tuple[DataLoader, Dict[str, int]]:
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        print("Sample count per class:")
        for cls, count in self.class_counts.items():
            print(f"Class '{cls}': {count} samples")
        return dataloader, self.class_counts

    def compute_class_distribution(self) -> Tuple[int, Dict[str, float]]:
        class_counts = Counter(self.y)
        total_samples = sum(class_counts.values())

        class_percentages = {
            class_name: (count / total_samples)
            for class_name, count in class_counts.items()
        }

        rounded_percentages = {
            self.get_class_name(class_label): round(percentage, 2)
            for class_label, percentage in class_percentages.items()
        }

        total_percentage = sum(rounded_percentages.values())
        difference = 1 - total_percentage
        if rounded_percentages:
            class_with_max_percentage = max(
                rounded_percentages, key=rounded_percentages.get
            )
            rounded_percentages[class_with_max_percentage] += difference

        return total_samples, rounded_percentages

    def get_class_name(self, label):
        for class_name, class_label in self.cls_dict.items():
            if class_label == label:
                return class_name
        return "Unknown"
