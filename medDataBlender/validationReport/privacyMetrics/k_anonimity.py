from collections import Counter
import torch
import hashlib
import numpy as np
from scipy.spatial.distance import cdist
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple


def extract_features(dataset: DataLoader, data_type: str) -> List[str]:
    """
    Extracts features from images or sequences for the entire dataset.
    """
    all_data = []
    for batch in dataset:
        data, _ = batch  # Ignore labels
        all_data.append(data)

    all_data = torch.cat(all_data, dim=0)  # Merge all batches

    if data_type == "mri":
        return compute_image_features(all_data)
    elif data_type == "ecg":
        return compute_sequence_features(all_data)
    else:
        raise ValueError("Unsupported data type")


def compute_image_features(images: torch.Tensor) -> List[str]:
    """
    Extracts features from images based on the average pixel distance across the dataset.
    """
    images_flat = images.view(images.size(0), -1).numpy()
    pairwise_distances = cdist(images_flat, images_flat, metric="euclidean")
    mean_pixel_distances = np.mean(pairwise_distances, axis=1)

    return [
        hashlib.sha256(str(dist).encode()).hexdigest()[:10]
        for dist in mean_pixel_distances
    ]


def compute_sequence_features(sequences: torch.Tensor) -> List[str]:
    """
    Extracts features from sequences based on mean and standard deviation across the dataset.
    """
    sequences_flat = sequences.view(sequences.size(0), -1).numpy()
    stats = [(np.mean(seq), np.std(seq)) for seq in sequences_flat]

    return [hashlib.sha256(str(stat).encode()).hexdigest()[:10] for stat in stats]


def compute_k_anonymity(
    real_dataloader: DataLoader, fake_dataloader: DataLoader, data_type: str
) -> float:
    """
    Computes a privacy score based on k-Anonymity.
    Returns a float between 0.0 (no privacy) and 1.0 (perfect privacy).
    """
    real_features = extract_features(real_dataloader, data_type)
    fake_features = extract_features(fake_dataloader, data_type)

    feature_counts = Counter(real_features)
    total_fake = len(fake_features)
    k_values = [feature_counts[f] for f in fake_features]

    count_k0 = k_values.count(0) if k_values else total_fake
    score = count_k0 / total_fake if total_fake > 0 else 1.0

    return float(score)
