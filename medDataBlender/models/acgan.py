import os
import time
from typing import Dict

import cv2  # OpenCV for image resizing
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image, ImageFilter

from .mri import Discriminator, Generator
from .evaluator import ImageEvaluator
from .baseModel import BaseModel, weights_init
import random
import torch.nn.utils as utils
import io
from urllib.parse import urlparse


def combine_vectors(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Combines two input vectors along the second dimension.

    Parameters:
        x (torch.Tensor): First vector of shape (n_samples, ?), typically the noise vector (n_samples, z_dim).
        y (torch.Tensor): Second vector of shape (n_samples, ?), typically a one-hot encoded class vector (n_samples, n_classes).

    Returns:
        torch.Tensor: The concatenated tensor of shape (n_samples, ?), converted to float type.
    """
    combined = torch.cat((x.float(), y.float()), 1)
    return combined


class Acgan(BaseModel):
    def __init__(
        self,
        eta,
        epochs,
        weight_decay,
        latent_space,
        image_shape,
        kernel_size,
        n_classes=2,
        lambda_gp=10,
        checkpoint_epoch=50,
        base_dir="saved_models_brain_MRI",
        use_s3=False,
    ):
        """
        Initializes the GAN model with the provided parameters.

        :param eta: Learning rate for the optimizers (typically used for the generator and discriminator).
        :param epochs: The number of training epochs to run.
        :param weight_decay: Regularization term (L2 penalty) for the optimizer to prevent overfitting.
        :param latent_space: The size of the latent space used as input to the generator (typically a vector of random noise).
        :param image_shape: The shape of the images (e.g., channels, height, width).
        :param kernel_size: The size of the convolutional kernel used in the model's layers.
        :param n_classes: The number of classes in the conditional GAN setup (default is 2).
        :param lambda_gp: Coefficient for the gradient penalty term (used in Wasserstein GAN with Gradient Penalty).
        :param checkpoint_epoch: The frequency (in terms of epochs) at which the model checkpoints are saved.
        :param base_dir: The base directory where the models will be saved during and after training.

        Initializes the necessary components like optimizers, loss functions, device setup, and directories.
        """

        super().__init__(
            epochs, n_classes, base_dir, checkpoint_epoch, weight_decay, use_s3
        )

        # Store hyperparameters and configuration
        self.eta = eta  # Learning rate
        self.latent_space = latent_space  # Size of the latent space for random noise
        self.image_shape = (
            image_shape  # Shape of the generated images (height, width, channels)
        )
        self.kernel_size = kernel_size  # Size of convolutional kernels in the model
        # Gradient penalty coefficient used in Wasserstein GAN with Gradient Penalty (WGAN-GP)
        self.lambda_gp = lambda_gp
        # Loss function used for training; here, it’s binary cross-entropy with logits
        self.criterion = nn.BCEWithLogitsLoss()

    def _gradient_penalty(
        self,
        real_samples: torch.Tensor,
        fake_samples: torch.Tensor,
        one_hot_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculates the gradient penalty for a Conditional GAN.

        Args:
            real_samples (torch.Tensor): Tensor of real samples with shape (batch_size, channels, height, width).
            fake_samples (torch.Tensor): Tensor of generated samples with shape (batch_size, channels, height, width).
            one_hot_labels (torch.Tensor): Tensor of one-hot encoded labels with shape (batch_size, num_classes).

        Returns:
            torch.Tensor: The gradient penalty value.
        """
        batch_size = real_samples.size(0)

        # Sample epsilon from the uniform distribution
        epsilon = torch.rand(batch_size, 1, 1, 1, device=self.device)
        epsilon = epsilon.expand_as(real_samples)

        # Interpolate between real and fake samples
        interpolated_images = (
            epsilon * real_samples + (1 - epsilon) * fake_samples
        ).requires_grad_(True)

        # Prepare conditional labels for the interpolated images
        image_one_hot_labels = one_hot_labels[:, :, None, None]
        image_one_hot_labels = image_one_hot_labels.repeat(
            1, 1, real_samples.size(2), real_samples.size(3)
        )

        # Combine the interpolated images with the labels
        interpolated_images_and_labels = combine_vectors(
            interpolated_images, image_one_hot_labels
        )

        # Compute the logits for the interpolated images
        d_interpolates = self.discriminator_model(interpolated_images_and_labels)
        grad_outputs = torch.ones_like(d_interpolates, device=self.device)

        # Compute the gradient
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolated_images,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        # Compute the gradient norm
        gradients = gradients.view(batch_size, -1)
        grad_norm = gradients.norm(2, dim=1)

        # Compute the gradient penalty
        gradient_penalty = torch.mean((grad_norm - 1) ** 2)

        return gradient_penalty

    def _generator_loss(self, fake_preds: torch.Tensor) -> torch.Tensor:
        """
        Computes the loss for the generator.

        Args:
            fake_preds (torch.Tensor): Tensor of predictions from the discriminator for fake samples.

        Returns:
            torch.Tensor: The generator loss.
        """
        return self.criterion(fake_preds, torch.ones_like(fake_preds).to(self.device))

    def _discriminator_loss(
        self, real_output: torch.Tensor, fake_output: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the loss for the discriminator.

        Args:
            real_output (torch.Tensor): Tensor of discriminator's output for real samples.
            fake_output (torch.Tensor): Tensor of discriminator's output for fake samples.

        Returns:
            torch.Tensor: The discriminator loss.
        """
        real_loss = self.criterion(
            real_output, torch.ones_like(real_output).to(self.device)
        )
        fake_loss = self.criterion(
            fake_output, torch.zeros_like(fake_output).to(self.device)
        )
        return real_loss + fake_loss

    def _samples(
        self,
        noise: torch.Tensor,
        labels: torch.Tensor,
        e: int,
        num_batch: int,
        save_samples: bool = True,
    ) -> torch.Tensor:
        """
        Generates and optionally saves sample images during training.

        Args:
            noise (torch.Tensor): Noise vector for the generator.
            labels (torch.Tensor): Conditional labels.
            e (int): Epoch number.
            num_batch (int): Batch number.
            save_samples (bool): Whether to save generated samples.

        Returns:
            torch.Tensor: The generated fake images.
        """

        # Generate images
        fake_images = self.generator_model(noise)
        images = ((fake_images + 1) * 127.5).clamp(0, 255).type(torch.uint8).cpu()
        ys = labels.cpu()

        if save_samples:
            out_dir = f"{self.base_dir}/training_output_samples"
            self.create_directory(out_dir)
            out_dir = f"{out_dir}/epoch_{e}"
            self.create_directory(out_dir)

            # Prepare the plot
            plt.figure(figsize=(16, 4))
            for i in range(8):
                plt.subplot(2, 4, i + 1)
                img = images[i].permute(1, 2, 0).numpy()
                img_resized = cv2.resize(img, (496, 248), interpolation=cv2.INTER_AREA)
                plt.imshow(img_resized, cmap="gray")
                plt.title(ys[i].item())
                plt.axis("off")
            plt.subplots_adjust(wspace=0.1, hspace=0.3)

            filename = f"synt_{e}_{num_batch}.png"
            if self.use_s3:
                parsed = urlparse(out_dir)
                bucket = parsed.netloc
                prefix = parsed.path.lstrip("/")

                buffer = io.BytesIO()
                plt.savefig(buffer, format="png")
                buffer.seek(0)
                s3_key = os.path.join(prefix, filename)
                self.s3_client.upload_fileobj(buffer, bucket, s3_key)
                plt.close()
            else:
                out_path = os.path.join(out_dir, filename)
                plt.savefig(out_path)
                plt.close()

        return fake_images

    def _generator(self):
        """
        Initializes the Generator model for the GAN.

        The Generator class is a neural network that takes a latent vector (random noise) and transforms it
        through fully connected and transposed convolution layers to generate synthetic images.

        Parameters:
            None
        """

        # Initialize the Generator model with the specified input dimensions and latent space
        self.generator_model = Generator(
            self.image_shape[1], self.latent_space + self.n_classes
        ).to(self.device)
        self.generator_model = self.generator_model.apply(weights_init)

    def _discriminator(self):
        """
        Initializes the Discriminator model for the GAN.

        The Discriminator class is a neural network that takes an image and classifies it as real or fake.
        The network consists of three convolutional layers followed by a fully connected layer to output a scalar value.
        """

        # Initialize the Discriminator model with the specified image size and channels
        self.discriminator_model = Discriminator(
            self.image_shape[1], self.image_shape[0] + self.n_classes
        ).to(self.device)
        self.discriminator_model = self.discriminator_model.apply(weights_init)

    def build(self):
        """
        Builds the generator and discriminator model, sets up optimizers,
        and applies learning rate schedulers for both models.

        Input:
            - self: An instance of the class that contains the method.
                It is expected to have the following attributes:
                    - `eta`: Learning rate for both the generator and discriminator (float).
                    - `weight_decay`: Weight decay for the optimizer (float).
                    - `discriminator_model`: The discriminator model (torch.nn.Module).
                    - `generator_model`: The generator model (torch.nn.Module).

        Output:
            None: This method does not return anything.
        """

        # Initialize the generator and discriminator models
        self._generator()
        self._discriminator()

        # Set up the optimizer for the discriminator
        self.d_optimizer = optim.Adam(
            self.discriminator_model.parameters(),
            lr=self.eta,
            betas=(0.5, 0.999),
            weight_decay=self.weight_decay,
        )

        # Set up the optimizer for the generator
        self.g_optimizer = optim.Adam(
            self.generator_model.parameters(),
            lr=self.eta,
            betas=(0.5, 0.999),
            weight_decay=self.weight_decay * 0.5,
        )

        # Use ReduceLROnPlateau instead of ExponentialLR
        self.d_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.d_optimizer,
            mode="max",  # We want to maximize SSIM
            factor=0.5,  # Reduce LR by half
            patience=5,  # Wait 5 epochs of no improvement
            verbose=True,
            threshold=1e-4,
        )

        self.g_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.g_optimizer,
            mode="max",
            factor=0.5,
            patience=5,
            verbose=True,
            threshold=1e-4,
        )

    def train_algorithm(self, dataloader):
        """
        Train the model using the provided data loader.

        :param dataloader: (torch.utils.data.DataLoader) The data loader providing batches of training data.

        :return: None
        """

        if self.base_dir is None:
            print("without base dir it is possible to work only in generation mode")
            return

        print("Start training..")

        num_batches = 0  # Counter for the number of batches
        clip_value = 1.0

        for epoch in range(self.epochs):
            epoch_start_time = time.time()
            d_loss_epoch = 0.0
            g_loss_epoch = 0.0
            ssim_epoch = 0.0
            fid_epoch = 0
            mmd_epoch = 0

            for batch_idx, (real_images, real_labels) in enumerate(dataloader):
                batch_start_time = time.time()

                real_images, real_labels = real_images.to(self.device), real_labels.to(
                    self.device
                )
                one_hot_labels = (
                    F.one_hot(real_labels, num_classes=self.n_classes)
                    .float()
                    .to(self.device)
                )

                batch_size = real_images.size(0)
                image_one_hot_labels = one_hot_labels[:, :, None, None].repeat(
                    1, 1, self.image_shape[1], self.image_shape[1]
                )
                # ---------------- Train Discriminator (more steps in early epochs) ----------------
                d_steps = 5 if epoch < 5 else 1

                noise = torch.randn(batch_size, self.latent_space, device=self.device)
                noise_and_labels = combine_vectors(noise, one_hot_labels)

                d_loss_avg_step = 0.0
                for _ in range(d_steps):
                    fake_images = self.generator_model(noise_and_labels)

                    real_image_and_labels = combine_vectors(
                        real_images, image_one_hot_labels
                    )
                    fake_image_and_labels = combine_vectors(
                        fake_images, image_one_hot_labels
                    )

                    real_preds = self.discriminator_model(real_image_and_labels)
                    fake_preds = self.discriminator_model(
                        fake_image_and_labels.detach()
                    )

                    gp = self._gradient_penalty(
                        real_images, fake_images, one_hot_labels
                    )

                    self.d_optimizer.zero_grad()
                    d_loss = (
                        self._discriminator_loss(real_preds, fake_preds)
                        + self.lambda_gp * gp
                    )
                    d_loss.backward()
                    utils.clip_grad_norm_(
                        self.discriminator_model.parameters(), clip_value
                    )
                    self.d_optimizer.step()
                    d_loss_avg_step += d_loss.item()

                # ---------------- Train Generator ----------------
                # Il noise rimane invariato durante la parte del Generatore
                fake_images = self.generator_model(noise_and_labels)
                fake_image_and_labels = combine_vectors(
                    fake_images, image_one_hot_labels
                )

                fake_preds = self.discriminator_model(fake_image_and_labels)

                self.g_optimizer.zero_grad()
                g_loss = self._generator_loss(fake_preds)
                g_loss.backward()
                utils.clip_grad_norm_(self.generator_model.parameters(), clip_value)

                self.g_optimizer.step()

                # Accumulate metrics
                d_loss_epoch += d_loss_avg_step
                g_loss_epoch += g_loss.item()
                num_batches += 1

                eval = ImageEvaluator(
                    [(real_images, real_labels)], [(fake_images, real_labels)]
                )
                batch_metrics = eval.evaluate()
                del eval

                ssim_value = batch_metrics["Avg SSIM"]
                ssim_epoch += ssim_value
                mmd_value = batch_metrics["Avg MMD"]
                mmd_epoch += mmd_value

                batch_end_time = time.time()
                batch_time = batch_end_time - batch_start_time

                if (batch_idx + 1) % 50 == 0:
                    print(
                        f"Epoch [{epoch + 1}/{self.epochs}], Batch [{batch_idx + 1}/{len(dataloader)}], "
                        f"D_Loss: {d_loss_avg_step:.4f}, G_Loss: {g_loss.item():.4f}, "
                        f"SSIM: {ssim_value:.4f}, MMD: {mmd_value:.4f}, "
                        f"Batch Time: {batch_time:.4f} sec"
                    )

            # Update learning rates using ReduceLROnPlateau
            epoch_end_time = time.time()
            epoch_time = epoch_end_time - epoch_start_time
            avg_d_loss = d_loss_epoch / len(dataloader)
            avg_g_loss = g_loss_epoch / len(dataloader)
            avg_ssim = ssim_epoch / len(dataloader)

            self.d_scheduler.step(avg_d_loss)
            self.g_scheduler.step(-avg_ssim)

            print(
                f"Epoch [{epoch + 1}/{self.epochs}] Complete. Avg D_Loss: {avg_d_loss:.4f}, Avg G_Loss: {avg_g_loss:.4f}, "
                f"Epoch Time: {epoch_time:.4f} sec"
            )

            if epoch % self.checkpoint_epoch == 0 and self.base_dir is not None:
                with torch.no_grad():
                    self._save_models(epoch)

                    # Seleziona casualmente l'1% dei batch dal dataloader
                    batch_percentage = 0.01
                    total_batches = len(dataloader)
                    selected_batches = random.sample(
                        range(total_batches),
                        max(1, int(batch_percentage * total_batches)),
                    )

                    # Liste per le metriche
                    ssim_scores = []
                    mmd_scores = []
                    all_synthetic_images = []
                    all_real_images = []
                    all_labels = []

                    # Salva campioni di immagini e i modelli

                    for idx, (real_images, real_labels) in enumerate(dataloader):
                        if idx not in selected_batches:
                            continue

                        # Genera immagini sintetiche con le stesse etichette
                        sample_noise = torch.randn(
                            real_images.size(0), self.latent_space, device=self.device
                        )
                        one_hot_labels = (
                            F.one_hot(real_labels, num_classes=self.n_classes)
                            .float()
                            .to(self.device)
                        )

                        noise_and_labels = combine_vectors(sample_noise, one_hot_labels)
                        fake_imgs = self._samples(
                            noise_and_labels, real_labels, epoch, idx
                        )
                        all_synthetic_images.append(fake_imgs)
                        all_real_images.append(real_images)
                        all_labels.append(real_labels)

                    all_real_images = torch.cat(all_real_images, dim=0)
                    all_labels = torch.cat(all_labels, dim=0)
                    all_synthetic_images = torch.cat(all_synthetic_images, dim=0)

                    # Valuta le immagini sintetiche rispetto a quelle reali
                    eval = ImageEvaluator(
                        [(all_real_images, all_labels)],  # Dati reali
                        [(all_synthetic_images, all_labels)],  # Dati sintetici
                    )
                    metrics = eval.evaluate()

                    # Calcola la media delle metriche
                    avg_ssim = metrics["Avg SSIM"]
                    avg_mmd = metrics["Avg MMD"]

                    # Stampa i risultati
                    print(
                        f"Evaluation - Epoch {epoch}: Avg SSIM: {avg_ssim:.4f}, Avg MMD: {avg_mmd:.4f}"
                    )

    def generate_and_save_samples_by_class(
        self,
        num_samples: int,
        save_dir: str,
        label_percentage: Dict[str, float],
        batch_size: int = 100,
    ):
        self.create_directory(save_dir)
        if self.use_s3:
            parsed = urlparse(save_dir)
            bucket = parsed.netloc
            prefix = parsed.path.lstrip("/")

        class_samples = {
            class_name: int(num_samples * percentage)
            for class_name, percentage in label_percentage.items()
        }
        total_allocated = sum(class_samples.values())
        if total_allocated < num_samples:
            diff = num_samples - total_allocated
            first_key = next(iter(class_samples))
            class_samples[first_key] += diff

        class_dirs = {}
        for class_name in label_percentage:
            class_dir = os.path.join(save_dir, class_name)
            self.create_directory(class_dir)
            class_dirs[class_name] = class_dir

        generated_samples = 0
        while generated_samples < num_samples:
            batch_samples = {}
            remaining_total = num_samples - generated_samples
            for class_name in class_samples:
                remaining = class_samples[class_name]
                if remaining > 0:
                    to_generate = min(batch_size, remaining, remaining_total)
                    batch_samples[class_name] = to_generate
                    class_samples[class_name] -= to_generate
                    remaining_total -= to_generate
                if remaining_total <= 0:
                    break

            if sum(batch_samples.values()) == 0:
                break

            total_batch_size = sum(batch_samples.values())
            noise = torch.randn(total_batch_size, self.latent_space, device=self.device)

            one_hot_labels = []
            class_indices = list(label_percentage.keys())
            for idx, class_name in enumerate(class_indices):
                n = batch_samples.get(class_name, 0)
                if n > 0:
                    one_hot = torch.zeros(n, self.n_classes, device=self.device)
                    one_hot[:, idx] = 1
                    one_hot_labels.append(one_hot)
            one_hot_labels = torch.cat(one_hot_labels, dim=0)
            noise_and_labels = torch.cat((noise, one_hot_labels), dim=1)

            self.generator_model.eval()
            with torch.no_grad():
                fake_images = self.generator_model(noise_and_labels)

            fake_images = ((fake_images + 1) * 127.5).clamp(0, 255).type(torch.uint8)
            fake_images = fake_images.cpu().permute(0, 2, 3, 1).numpy()
            if fake_images.shape[-1] == 1:
                fake_images = fake_images.squeeze(-1)

            image_idx = 0
            for class_name, num_samples_for_class in batch_samples.items():
                class_dir = class_dirs[class_name]
                for i in range(num_samples_for_class):
                    img_array = fake_images[image_idx]
                    if img_array.ndim == 2:
                        denoised = cv2.medianBlur(img_array, ksize=3)
                    else:
                        denoised = cv2.medianBlur(
                            cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY), ksize=3
                        )
                    img = Image.fromarray(denoised)
                    img = img.resize((496, 248), resample=Image.Resampling.BICUBIC)
                    img = img.filter(ImageFilter.SHARPEN)

                    filename = f"generated_image_{generated_samples}.png"
                    if self.use_s3:

                        buffer = io.BytesIO()
                        img.save(buffer, format="PNG")
                        buffer.seek(0)
                        s3_key = os.path.join(prefix, class_name, filename)
                        self.s3_client.upload_fileobj(buffer, bucket, s3_key)
                    else:
                        img_save_path = os.path.join(class_dir, filename)
                        img.save(img_save_path)

                    generated_samples += 1
                    image_idx += 1

            print(f"{generated_samples}/{num_samples} images generated and saved.")

        print(f"All {num_samples} samples have been generated and saved.")
