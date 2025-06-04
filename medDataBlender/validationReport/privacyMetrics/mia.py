import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from typing import Tuple, List


class CombinedMIA_Dataset(Dataset):
    def __init__(self, real_loader: DataLoader, fake_loader: DataLoader) -> None:
        """
        Creates a dataset combining real and fake samples with labels.

        Args:
            real_loader (DataLoader): DataLoader containing real samples.
            fake_loader (DataLoader): DataLoader containing fake samples.
        """
        self.data = []
        self.labels = []

        # Load real data with label 1
        for real_images, _ in real_loader:
            self.data.append(real_images)
            self.labels.append(torch.ones(real_images.shape[0], dtype=torch.long))

        # Load fake data with label 0
        for fake_images, _ in fake_loader:
            self.data.append(fake_images)
            self.labels.append(torch.zeros(fake_images.shape[0], dtype=torch.long))

        # Concatenate all data into a single tensor
        self.data = torch.cat(self.data, dim=0)
        self.labels = torch.cat(self.labels, dim=0)

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns a sample and its corresponding label.

        Args:
            idx (int): Index of the sample.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Sample and label.
        """
        return self.data[idx], self.labels[idx]


class MIA:
    def __init__(
        self, classifier: object, mia_loader: DataLoader, batch_size: int = 32
    ) -> None:
        """
        Initializes the Membership Inference Attack (MIA) class.

        Args:
            classifier (object): A classifier model that performs inference.
            real_loader (DataLoader): DataLoader containing real samples.
            fake_loader (DataLoader): DataLoader containing fake samples.
            batch_size (int, optional): Batch size for DataLoader. Defaults to 32.
        """
        self.classifier = classifier
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Sposta il modello sul device scelto
        self.classifier.model.to(self.device)

        # Crea un DataLoader unico per l'attacco MIA
        self.mia_loader = mia_loader

    def infer_membership(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the membership probability for each sample in the combined dataset.

        Returns:
            Tuple[np.ndarray, np.ndarray]: True labels and predicted membership probabilities.
        """
        true_labels: List[float] = []
        predicted_membership: List[float] = []

        self.classifier.model.eval()  # Modalità valutazione

        with torch.no_grad():
            for data, labels in self.mia_loader:
                data = data.to(self.device)
                labels = labels.to(self.device)

                outputs = self.classifier.model(data)
                probs = torch.softmax(outputs, dim=1)
                max_probs, _ = torch.max(probs, dim=1)

                predicted_membership.extend(
                    max_probs.cpu().numpy()
                )  # Porta su CPU per numpy
                true_labels.extend(labels.cpu().numpy())

        return np.array(true_labels), np.array(predicted_membership)

    def attack(self) -> float:
        """
        Performs the membership inference attack and computes accuracy.

        Returns:
            float: Accuracy of the attack.
        """
        true_labels, predicted_membership = self.infer_membership()

        # Accuratezza con soglia 0.5
        accuracy = accuracy_score(true_labels, predicted_membership > 0.5)
        return accuracy
