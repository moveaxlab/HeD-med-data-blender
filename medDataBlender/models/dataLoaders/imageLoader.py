import cv2
import pathlib
import os
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Dict, Any
import numpy as np
from .baseLoader import BaseDatasetReader


class OasisMRI(Dataset):
    """
    Custom PyTorch Dataset for handling MRI images.
    This dataset assumes grayscale images and adds an additional channel dimension.
    """

    def __init__(self, images: np.ndarray, labels: np.ndarray):
        self.images = torch.tensor(images, dtype=torch.float32).unsqueeze(1)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.images[idx], self.labels[idx]


class OasisMRI_ReadDataset(BaseDatasetReader):
    """
    Classe per leggere e processare il dataset di immagini MRI, anche da S3 se specificato.
    """

    def __init__(
        self,
        dataset_path: str,
        labels: List[str],
        data_shape: Tuple[int, int],
        batch_size: int = 10,
        dataset_class=OasisMRI,
        is_s3: bool = False,
        s3_region_name=None,
        s3_endpoint_url=None,
        s3_access_key_id=None,
        s3_secret_access_key=None,
    ):
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
        self.keyword_range = [150, 160]
        self.keyword = [
            f"_{i}" for i in range(self.keyword_range[0], self.keyword_range[-1] + 1)
        ]

    def _return_list_images(self) -> None:
        """
        Recupera la lista delle immagini per ciascuna classe e le memorizza in self.images.
        """
        self.images = []
        for label in self.labels:
            dir_path = os.path.join(self.dataset_path, label)
            if not os.path.isdir(dir_path):
                print(f"Attenzione: la directory {dir_path} non esiste.")
                self.images.append([])
                continue
            self.images.append(list(pathlib.Path(dir_path).glob("*.*")))

    def load_data(self) -> Tuple[DataLoader, Dict[str, int]]:
        """
        Carica le immagini, le normalizza e crea un DataLoader.
        Filtra le immagini in base alle parole chiave predefinite.

        :return: DataLoader e dizionario contenente il conteggio delle immagini per classe.
        """
        self._return_list_images()
        final_images = []
        labels = []

        for label_idx, label_name in enumerate(self.labels):
            count = 0
            for img_path in self.images[label_idx]:
                if any(
                    str(img_path).endswith(keyword + ".jpg") for keyword in self.keyword
                ) or "generated" in str(img_path):
                    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        print(f"Impossibile leggere l'immagine: {img_path}")
                        continue
                    img = cv2.resize(img, self.data_shape)
                    img = (img / 127.5) - 1.0
                    final_images.append(img)
                    labels.append(label_idx)
                    count += 1
            self.class_counts[label_name] = count

        images = np.array(final_images)
        labels = np.array(labels)

        print("Conteggio immagini per classe:")
        for label, count in self.class_counts.items():
            print(f"Classe '{label}': {count} immagini")

        dataset = self.dataset_class(images, labels)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        return dataloader, self.class_counts

    def compute_class_distribution(self) -> Tuple[int, Dict[str, float]]:
        total_images = sum(self.class_counts.values())
        class_percentages = {
            class_name: (count / total_images)
            for class_name, count in self.class_counts.items()
        }

        rounded_percentages = {
            class_name: round(percentage, 2)
            for class_name, percentage in class_percentages.items()
        }

        total_percentage = sum(rounded_percentages.values())
        difference = 1 - total_percentage
        if rounded_percentages:
            class_with_max_percentage = max(
                rounded_percentages, key=rounded_percentages.get
            )
            rounded_percentages[class_with_max_percentage] += difference

        return total_images, rounded_percentages
