from typing import List, Tuple
import torch
import numpy as np
from scipy.spatial.distance import cdist
from skimage.metrics import structural_similarity as ssim
from .SyntheticDataEvaluator import SyntheticDataEvaluator


def calculate_ssim(
    real_images: np.ndarray, fake_images: np.ndarray, win_size: int = 5
) -> float:
    """
    Calculates the Structural Similarity Index (SSIM) between two sets of images, real and generated.

    :param real_images: A numpy array of real images with shape [batch_size, height, width].
    :param fake_images: A numpy array of generated images with shape [batch_size, height, width].
    :param win_size: The size of the window for SSIM calculation. If the images are too small, reduce win_size.

    :return: The mean SSIM value for the batch of images.
    """
    batch_ssim = [
        ssim(
            real_images[i],
            fake_images[i],
            win_size=win_size,
            data_range=real_images[i].max() - real_images[i].min(),
        )
        for i in range(real_images.shape[0])
    ]

    return float(np.mean(batch_ssim))


def calculate_mmd(real_images: np.ndarray, fake_images: np.ndarray) -> float:
    """
    Compute the Maximum Mean Discrepancy (MMD) between two sets of images (real and fake).

    Parameters:
        real_images (np.ndarray): A batch of real images of shape (batch_size, H, W) or (batch_size, C, H, W).
        fake_images (np.ndarray): A batch of generated (fake) images of shape (batch_size, H, W) or (batch_size, C, H, W).

    Returns:
        float: The computed MMD value as a scalar.
    """
    # Flatten images into vectors
    real_images = real_images.reshape(real_images.shape[0], -1).astype(np.float32)
    fake_images = fake_images.reshape(fake_images.shape[0], -1).astype(np.float32)

    # Define the RBF kernel parameter (gamma)
    gamma = 1.0 / real_images.shape[1]

    # Compute the kernel matrices
    K_xx = np.exp(-gamma * cdist(real_images, real_images, metric="sqeuclidean")).mean()
    K_yy = np.exp(-gamma * cdist(fake_images, fake_images, metric="sqeuclidean")).mean()
    K_xy = np.exp(-gamma * cdist(real_images, fake_images, metric="sqeuclidean")).mean()

    # Calculate MMD (Maximum Mean Discrepancy)
    mmd = K_xx + K_yy - 2 * K_xy

    return float(mmd)


def _is_denormalized(images: torch.Tensor) -> bool:
    return images.dtype == torch.uint8 and torch.all((images >= 0) & (images <= 255))


def _ensure_denormalized(
    data: List[Tuple[torch.Tensor, torch.Tensor]],
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    processed = []
    for images, labels in data:
        if not _is_denormalized(images):
            images = ((images + 1) * 127.5).clamp(0, 255).type(torch.uint8).cpu()
        else:
            images = images.type(torch.uint8).cpu()
        processed.append((images, labels))
    return processed


class ImageEvaluator(SyntheticDataEvaluator):
    def __init__(
        self,
        real_data: List[Tuple[torch.Tensor, torch.Tensor]],
        synthetic_data: List[Tuple[torch.Tensor, torch.Tensor]],
    ):
        super().__init__(real_data, synthetic_data)
        # self.real_data = _ensure_denormalized(self.real_data)
        # self.synthetic_data = _ensure_denormalized(self.synthetic_data)

    def evaluate(self) -> dict:
        ssim_scores = []
        mmd_scores = []
        for (real_images, _), (synthetic_images, _) in zip(
            self.real_data, self.synthetic_data
        ):
            real_images_np = real_images.numpy()
            synthetic_images_np = synthetic_images.numpy()

            # Check solo sui canali e dimensioni spatiali, non sulla batch size
            if real_images_np.shape[1:] != synthetic_images_np.shape[1:]:
                raise ValueError(
                    "Real and synthetic images must have the same channel and spatial dimensions."
                )

            # Squeeze se è monocanale
            if real_images_np.shape[1] == 1:
                real_images_np = np.squeeze(real_images_np, axis=1)
                synthetic_images_np = np.squeeze(synthetic_images_np, axis=1)

            # Allinea la dimensione del batch
            min_batch = min(real_images_np.shape[0], synthetic_images_np.shape[0])
            real_images_np = real_images_np[:min_batch]
            synthetic_images_np = synthetic_images_np[:min_batch]

            # Calcolo delle metriche
            ssim_value = calculate_ssim(real_images_np, synthetic_images_np)
            mmd_value = calculate_mmd(real_images_np, synthetic_images_np)

            ssim_scores.append(ssim_value)
            mmd_scores.append(mmd_value)

        return {
            "Avg SSIM": np.mean(ssim_scores),
            "Avg MMD": np.mean(mmd_scores),
        }
