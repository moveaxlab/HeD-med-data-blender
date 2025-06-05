from .baseModel import BaseModel, weights_init
from .ecg import AdamW, Generator, Discriminator
from .evaluator import ECGQualityMetrics
import torch.nn as nn
import torch
import os
from typing import Dict
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import pandas as pd
import random
import io
from urllib.parse import urlparse


class LinearLrDecay(object):
    def __init__(self, optimizer, start_lr, end_lr, decay_start_step, decay_end_step):

        assert start_lr > end_lr
        self.optimizer = optimizer
        self.delta = (start_lr - end_lr) / (decay_end_step - decay_start_step)
        self.decay_start_step = decay_start_step
        self.decay_end_step = decay_end_step
        self.start_lr = start_lr
        self.end_lr = end_lr

    def step(self, current_step):
        if current_step <= self.decay_start_step:
            lr = self.start_lr
        elif current_step >= self.decay_end_step:
            lr = self.end_lr
        else:
            lr = self.start_lr - self.delta * (current_step - self.decay_start_step)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr
        return lr


def gradient_penalty(y, x, device):
    """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
    weight = torch.ones(y.size()).to(device)
    dydx = torch.autograd.grad(
        outputs=y,
        inputs=x,
        grad_outputs=weight,
        retain_graph=True,
        create_graph=True,
        only_inputs=True,
    )[0]

    dydx = dydx.view(dydx.size(0), -1)
    dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
    return torch.mean((dydx_l2norm - 1) ** 2)


class TTSGan(BaseModel):
    def __init__(
        self,
        seq_len: int = 187,
        epochs: int = 200,
        channels: int = 1,
        latent_dim: int = 100,
        data_embed_dim: int = 10,
        label_embed_dim: int = 10,
        depth: int = 3,
        num_heads: int = 5,
        forward_drop_rate: float = 0.5,
        attn_drop_rate: float = 0.5,
        patch_size: int = 1,
        data_emb_size: int = 50,
        label_emb_size: int = 10,
        n_classes: int = 2,
        checkpoint_epoch: int = 50,
        weight_decay: float = 1e-3,
        max_iter: int = 500000,
        g_lr: float = 0.0001,
        d_lr: float = 0.0003,
        n_critic: int = 1,
        lr_deacy: bool = False,
        base_dir: str = "saved_models_ecg_PTB",
        use_s3=False,
        s3_region_name=None,
        s3_endpoint_url=None,
        s3_access_key_id=None,
        s3_secret_access_key=None
    ):
        """
        Initializes the TTSGan model with specified parameters.

        Attributes:
            seq_len (int): Length of the input sequence.
            epochs (int): Number of training epochs.
            channels (int): Number of channels in the input data.
            latent_dim (int): Dimensionality of the latent space.
            data_embed_dim (int): Size of the data embedding.
            label_embed_dim (int): Size of the label embedding.
            depth (int): Depth of the Transformer-based architecture.
            num_heads (int): Number of attention heads.
            forward_drop_rate (float): Dropout rate in the feedforward layers.
            attn_drop_rate (float): Dropout rate in the attention layers.
            patch_size (int): Patch size for data embedding.
            data_emb_size (int): Size of the data embedding space.
            label_emb_size (int): Size of the label embedding space.
            n_classes (int): Number of output classes.
            checkpoint_epoch (int): Interval for saving model checkpoints.
            weight_decay (float): Weight decay for optimizers.
            max_iter (int): Maximum number of iterations for training.
            g_lr (float): Learning rate for the generator.
            d_lr (float): Learning rate for the discriminator.
            n_critic (int): Number of discriminator updates per generator update.
            lr_deacy (bool): Whether to apply learning rate decay.
            base_dir (str): Directory to save model checkpoints.

        """

        super().__init__(
            epochs=epochs, n_classes=n_classes, base_dir=base_dir, checkpoint_epoch=checkpoint_epoch, weight_decay=weight_decay, use_s3=use_s3,
            s3_region_name=s3_region_name, s3_endpoint_url=s3_endpoint_url, s3_access_key_id=s3_access_key_id, s3_secret_access_key=s3_secret_access_key
        )

        self.lr_deacy = lr_deacy
        self.seq_len = seq_len
        self.channels = channels
        self.latent_dim = latent_dim
        self.data_embed_dim = data_embed_dim
        self.label_embed_dim = label_embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.forward_drop_rate = forward_drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.patch_size = patch_size
        self.data_emb_size = data_emb_size
        self.label_emb_size = label_emb_size
        self.g_lr = g_lr
        self.d_lr = d_lr
        self.max_iter = max_iter
        self.n_critic = n_critic

    def build(self) -> None:
        """
        Constructs the generator and discriminator models and initializes their optimizers and learning rate schedulers.

        Returns:
            None
        """
        # Initialize the generator and discriminator
        self._generator()
        self._discriminator()

        # Define optimizers for both generator and discriminator
        self.g_optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.generator_model.parameters()),
            self.g_lr,
            weight_decay=self.weight_decay,
        )

        self.d_optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.discriminator_model.parameters()),
            self.d_lr,
            weight_decay=self.weight_decay,
        )

        # Define learning rate schedulers
        self.g_scheduler = LinearLrDecay(
            self.g_optimizer, self.g_lr, 0.0, 0, self.max_iter * self.n_critic
        )
        self.d_scheduler = LinearLrDecay(
            self.d_optimizer, self.d_lr, 0.0, 0, self.max_iter * self.n_critic
        )

    def _generator(self) -> None:
        """
        Initializes the Generator model for the GAN.

        The Generator model takes a latent vector (random noise) and transforms it into a synthetic ECG signal
        using a neural network architecture with attention-based mechanisms.

        Returns:
            None
        """
        self.generator_model = Generator(
            self.seq_len,
            self.channels,
            self.n_classes,
            self.latent_dim,
            self.data_embed_dim,
            self.label_embed_dim,
            self.depth,
            self.num_heads,
            self.forward_drop_rate,
            self.attn_drop_rate,
        ).to(self.device)

        # Apply weight initialization
        self.generator_model = self.generator_model.apply(weights_init)

    def _discriminator(self) -> None:
        """
        Initializes the Discriminator model for the GAN.

        The Discriminator model takes an input signal and determines whether it is real or generated by the Generator.
        The network consists of convolutional layers and fully connected layers for classification.

        Returns:
            None
        """
        self.discriminator_model = Discriminator(
            self.channels,
            self.patch_size,
            self.data_emb_size,
            self.label_emb_size,
            self.seq_len,
            self.depth,
            self.n_classes,
        ).to(self.device)

        # Apply weight initialization
        self.discriminator_model = self.discriminator_model.apply(weights_init)

    def _samples(
        self,
        noise: torch.Tensor,
        labels: torch.Tensor,
        e: int,
        num_batch: int,
        save_samples: bool = True,
    ) -> torch.Tensor:
        """
        Generates and saves samples of synthetic ECG signals from the generator.
        Saves the signals as CSV and individual plots.

        Args:
            noise (torch.Tensor): A tensor of random noise used as input to the generator.
            labels (torch.Tensor): A tensor of labels corresponding to each sample.
            e (int): The current epoch number.
            num_batch (int): Current batch number during training.
            save_samples (bool): Whether to save samples to disk or S3.

        Returns:
            torch.Tensor: The generated synthetic signals.
        """

        # Generazione segnali
        signals = self.generator_model(noise, labels)
        signals_numpy = signals.cpu().detach().numpy()
        ys = labels.cpu().numpy()

        data = []
        selected_signals = {label: 0 for label in np.unique(ys)}

        save_dir = f"{self.base_dir}/training_samples"
        self.create_directory(save_dir)

        if save_samples:
            epoch_dir = f"{save_dir}/epoch_{e}"
            self.create_directory(epoch_dir)

            for idx, (signal, label) in enumerate(zip(signals_numpy, ys)):
                row = np.append(signal.flatten(), label)
                data.append(row)
                selected_signals[label] += 1

                # Salvataggio del plot
                plt.figure(figsize=(6, 2))
                plt.plot(signal[0][0])
                plt.axis("off")
                plt.title(f"Label: {label}")

                filename = f"{idx}_{label}.png"
                if self.use_s3:
                    parsed = urlparse(epoch_dir)
                    bucket = parsed.netloc
                    prefix = parsed.path.lstrip("/")

                    buffer = io.BytesIO()
                    plt.savefig(buffer, format="png")
                    buffer.seek(0)
                    s3_key = os.path.join(prefix, filename)
                    self.s3_client.upload_fileobj(buffer, bucket, s3_key)
                    plt.close()
                else:
                    plot_path = os.path.join(epoch_dir, filename)
                    plt.savefig(plot_path)
                    plt.close()

            # Salvataggio CSV
            csv_filename = "synthetic_ecg.csv"
            csv_path = os.path.join(epoch_dir, csv_filename)

            if self.use_s3:
                parsed = urlparse(epoch_dir)
                bucket = parsed.netloc
                prefix = parsed.path.lstrip("/")
                buffer = io.StringIO()
                pd.DataFrame(data).to_csv(buffer, index=False, header=False)
                buffer.seek(0)
                s3_key = os.path.join(prefix, csv_filename)
                self.s3_client.put_object(
                    Bucket=bucket, Key=s3_key, Body=buffer.getvalue()
                )
            else:
                pd.DataFrame(data).to_csv(csv_path, index=False, header=False)

        return signals

    # Define the train_algorithm method with type annotations
    def train_algorithm(self, dataloader: torch.utils.data.DataLoader) -> None:
        """
        Trains the GAN model (Discriminator and Generator) for a number of epochs.

        input:
            dataloader (torch.utils.data.DataLoader): DataLoader object that provides
                batches of training data (real images and their labels).

        Returns:
            None: This method trains the model in-place, it does not return anything.
        """

        if self.base_dir is None:
            print("without base dir it is possible to work only in generation mode")
            return

        global_steps = 0
        cls_criterion = (
            nn.CrossEntropyLoss()
        )  # Cross entropy loss for classification task
        lambda_cls = 1  # Weight for classification loss in the discriminator
        lambda_gp = 10  # Weight for gradient penalty term in the discriminator

        print("Start Training")

        # Initialize learning rate schedulers if they are enabled
        lr_schedulers = (self.g_scheduler, self.d_scheduler) if self.lr_deacy else None

        # Loop over epochs
        for epoch in range(0, self.epochs):
            epoch_start_time = time.time()  # Track the start time for the epoch
            d_loss_epoch = 0.0  # Discriminator loss for the entire epoch
            g_loss_epoch = 0.0  # Generator loss for the entire epoch

            # Loop over batches in the dataloader
            for iter_idx, (real_imgs, real_img_labels) in enumerate(tqdm(dataloader)):
                batch_start_time = time.time()  # Track the start time for the batch

                # Move data to the specified device (e.g., GPU or CPU)
                real_imgs = real_imgs.to(self.device, dtype=torch.float32)
                real_img_labels = real_img_labels.to(self.device, dtype=torch.long)

                # Generate random noise for the generator's input
                noise = torch.randn(
                    real_imgs.shape[0], self.latent_dim, device=self.device
                )
                fake_img_labels = torch.randint(
                    0, 5, (real_imgs.shape[0],), device=self.device
                )
                # Train Discriminator
                self.discriminator_model.zero_grad()  # Reset gradients for the discriminator
                r_out_adv, r_out_cls = self.discriminator_model(
                    real_imgs
                )  # Real image outputs
                fake_imgs = self.generator_model(
                    noise, fake_img_labels
                )  # Generate fake images
                f_out_adv, f_out_cls = self.discriminator_model(
                    fake_imgs
                )  # Fake image outputs

                # Compute the gradient penalty for the discriminator
                alpha = torch.rand(real_imgs.size(0), 1, 1, 1, device=self.device)
                x_hat = (
                    alpha * real_imgs.data + (1 - alpha) * fake_imgs.data
                ).requires_grad_(True)
                out_src, _ = self.discriminator_model(x_hat)
                d_loss_gp = gradient_penalty(out_src, x_hat, self.device)

                # Compute the discriminator loss (adversarial + classification loss)
                d_real_loss = -torch.mean(r_out_adv)
                d_fake_loss = torch.mean(f_out_adv)
                d_adv_loss = d_real_loss + d_fake_loss
                d_cls_loss = cls_criterion(r_out_cls, real_img_labels)
                d_loss = d_adv_loss + lambda_cls * d_cls_loss + lambda_gp * d_loss_gp

                d_loss.backward()  # Backpropagate discriminator loss
                torch.nn.utils.clip_grad_norm_(
                    self.discriminator_model.parameters(), 5.0
                )  # Gradient clipping
                self.d_optimizer.step()  # Update discriminator parameters

                # Train Generator
                self.generator_model.zero_grad()  # Reset gradients for the generator
                gen_imgs = self.generator_model(
                    noise, fake_img_labels
                )  # Generate fake images
                g_out_adv, g_out_cls = self.discriminator_model(
                    gen_imgs
                )  # Discriminator outputs for fake images
                g_loss_adv = -torch.mean(g_out_adv)  # Generator adversarial loss
                g_loss_cls = cls_criterion(
                    g_out_cls, fake_img_labels
                )  # Generator classification loss
                g_loss = g_loss_adv + lambda_cls * g_loss_cls

                g_loss.backward()  # Backpropagate generator loss
                torch.nn.utils.clip_grad_norm_(
                    self.generator_model.parameters(), 5.0
                )  # Gradient clipping
                self.g_optimizer.step()  # Update generator parameters

                # Track loss for the entire epoch
                d_loss_epoch += d_loss.item()
                g_loss_epoch += g_loss.item()

                batch_end_time = time.time()  # Track the end time for the batch
                batch_time = batch_end_time - batch_start_time

                # Print statistics every 50 batches
                if (iter_idx + 1) % 50 == 0:
                    print(
                        f"Epoch [{epoch + 1}/{self.epochs}], Batch [{iter_idx + 1}/{len(dataloader)}], "
                        f"D_Loss: {d_loss.item():.4f}, G_Loss: {g_loss.item():.4f}, "
                        f"Batch Time: {batch_time:.4f} sec"
                    )

            # Update learning rate schedulers if they are enabled
            if lr_schedulers:
                self.g_lr = self.g_scheduler.step(global_steps)
                self.d_lr = self.d_scheduler.step(global_steps)

            global_steps += 1

            # Track and print the epoch's time and average loss values
            epoch_end_time = time.time()
            epoch_time = epoch_end_time - epoch_start_time
            avg_d_loss = d_loss_epoch / len(dataloader)
            avg_g_loss = g_loss_epoch / len(dataloader)
            print(
                f"Epoch [{epoch + 1}/{self.epochs}] Complete. Avg D_Loss: {avg_d_loss:.4f}, Avg G_Loss: {avg_g_loss:.4f}, "
                f"Epoch Time: {epoch_time:.4f} sec"
            )

            # generate samples every 5 epochs
            if (
                epoch % self.checkpoint_epoch == 0
                and self.base_dir is not None
                and epoch != 0
            ):
                self._save_models(epoch)

                with torch.no_grad():
                    # Imposta la percentuale di batch da selezionare casualmente
                    batch_percentage = 0.005
                    total_batches = len(dataloader)

                    # Seleziona il batch casualmente
                    selected_batches = random.sample(
                        range(total_batches), int(batch_percentage * total_batches)
                    )

                    # Liste per raccogliere segnali sintetici e label
                    all_noises = []
                    all_reals = []
                    all_labels = []

                    # Estrazione di segnali e etichette per ogni batch selezionato
                    for idx, (real_batch, real_batch_labels) in enumerate(dataloader):
                        if idx not in selected_batches:
                            continue

                        # Generazione segnali sintetici con le stesse etichette
                        all_noises.append(
                            torch.randn(
                                real_batch.size(0), self.latent_dim, device=self.device
                            )
                        )
                        all_reals.append(real_batch)
                        all_labels.append(real_batch_labels)

                    # Concatena tutti i tensori accumulati
                    all_reals = torch.cat(all_reals, dim=0)
                    all_labels = torch.cat(all_labels, dim=0)
                    all_noises = torch.cat(all_noises, dim=0)

                    # Salva i dati generati
                    synthetic_signals = self._samples(all_noises, all_labels, epoch)

                    # Passa i dati reali e sintetici come liste di tuple (segnali, etichette)

                    eval_ecg = ECGQualityMetrics(
                        [(all_reals, all_labels)],  # Dati reali come lista di tuple
                        [
                            (synthetic_signals, all_labels)
                        ],  # Dati sintetici come lista di tuple
                    )

                    metrics = eval_ecg.evaluate()
                    del eval_ecg

                    # Calcola i valori medi delle metriche
                    mean_mae = metrics["MAE"]
                    mean_pearson = metrics["Pearson"]
                    mean_spearman = metrics["Spearman"]
                    mean_dtw = metrics["DTW"]

                    # Stampa i risultati medi
                    print(
                        f"Epoch {epoch}: MAE: {mean_mae:.4f}, Pearson: {mean_pearson:.4f}, "
                        f"Spearman: {mean_spearman:.4f}, DTW: {mean_dtw:.4f}"
                    )

    def generate_and_save_samples_by_class(
        self,
        num_samples: int,
        save_dir: str,
        label_percentage: Dict[str, float],
        batch_size: int = 100,
    ):
        """
        Generates and saves synthetic ECG signals by class according to specified label percentages.

        :param num_samples: The total number of samples to generate.
        :param save_dir: Directory where the generated ECG signals will be saved.
        :param label_percentage: A dictionary with class names as keys and their respective percentage in the range [0, 100].
        :param batch_size: The batch size used to generate signals in batches.

        :return: None. The signals are saved in the provided directory.
        """
        import os
        import torch
        import numpy as np
        import pandas as pd

        os.makedirs(save_dir, exist_ok=True)

        # Create a mapping from class names to integers
        class_to_int = {
            class_name: idx for idx, class_name in enumerate(label_percentage.keys())
        }

        # Calculate the number of signals per class based on the percentages
        class_samples = {
            class_name: int(num_samples * (percentage / 100))
            for class_name, percentage in label_percentage.items()
        }

        # List to store all generated signals for CSV saving
        all_signals = []

        generated_samples = 0
        while generated_samples < num_samples:
            # Calculate how many signals to generate per class in this batch
            batch_samples = {class_name: 0 for class_name in label_percentage}
            for class_name, remaining in class_samples.items():
                if remaining > 0:
                    batch_samples[class_name] = min(batch_size, remaining)

            total_batch_size = sum(batch_samples.values())
            if total_batch_size == 0:
                break  # No more signals to generate

            # Generate noise and labels
            noise = torch.randn(total_batch_size, self.latent_dim, device=self.device)
            label_list = []
            for class_name, num_samples_for_class in batch_samples.items():
                label_list.extend([class_to_int[class_name]] * num_samples_for_class)

            labels = torch.tensor(label_list, dtype=torch.long, device=self.device)

            # Generate synthetic signals
            self.generator_model.eval()
            with torch.no_grad():
                fake_signals = self.generator_model(noise, labels)

            fake_signals = fake_signals.cpu().detach().numpy()

            # Save each signal with the label as the last value
            signal_idx = 0
            for class_name, num_samples_for_class in batch_samples.items():
                if num_samples_for_class > 0:
                    for _ in range(num_samples_for_class):
                        signal = fake_signals[signal_idx][0][
                            0
                        ]  # Shape assumed to be (1, 1, N)
                        flattened_signal = signal.flatten()
                        signal_with_label = np.append(
                            flattened_signal, class_to_int[class_name]
                        )
                        all_signals.append(signal_with_label)
                        signal_idx += 1

            generated_samples += total_batch_size
            print(f"{generated_samples}/{num_samples} ECG signals generated.")

        # Convert to DataFrame and save to CSV
        signals_df = pd.DataFrame(all_signals)
        csv_save_path = os.path.join(save_dir, "generated_ecg_signals.csv")
        signals_df.to_csv(csv_save_path, index=False)

        print(
            f"All {num_samples} samples have been generated and saved in CSV at {csv_save_path}."
        )

    def generate_and_save_plot_by_class(
        self,
        num_samples: int,
        save_dir: str,
        label_percentage: Dict[str, float],
        batch_size: int = 100,
    ):
        """
        Generates and saves synthetic ECG signals as plot images and txt files by class.
        Supports saving either locally or to S3 depending on `self.use_s3`.

        Args:
            num_samples (int): Total number of samples to generate.
            save_dir (str): Base directory or S3 path for saving.
            label_percentage (Dict[str, float]): Class distribution in percentages.
            batch_size (int): Batch size for generation.
        """

        self.create_directory(save_dir)
        if self.use_s3:
            parsed = urlparse(save_dir)
            bucket = parsed.netloc
            prefix = parsed.path.lstrip("/")

        # Mappa classi in interi
        class_to_int = {cls: idx for idx, cls in enumerate(label_percentage.keys())}

        # Numero di segnali per classe
        class_samples = {
            cls: int(num_samples * (pct / 100)) for cls, pct in label_percentage.items()
        }
        # Corregge se c'è un delta
        total_allocated = sum(class_samples.values())
        if total_allocated < num_samples:
            diff = num_samples - total_allocated
            first_key = next(iter(class_samples))
            class_samples[first_key] += diff

        # Directory per classe
        class_dirs = {}
        for cls in label_percentage:
            class_dir = os.path.join(save_dir, cls)
            self.create_directory(class_dir)
            class_dirs[cls] = class_dir

        generated_samples = 0
        while generated_samples < num_samples:
            # Determina quanti segnali per classe generare in questo batch
            batch_samples = {}
            remaining_total = num_samples - generated_samples
            for cls in class_samples:
                remaining = class_samples[cls]
                if remaining > 0:
                    to_generate = min(batch_size, remaining, remaining_total)
                    batch_samples[cls] = to_generate
                    class_samples[cls] -= to_generate
                    remaining_total -= to_generate
                if remaining_total <= 0:
                    break

            if sum(batch_samples.values()) == 0:
                break

            # Genera input e labels
            total_batch_size = sum(batch_samples.values())
            noise = torch.randn(total_batch_size, self.latent_dim, device=self.device)
            label_list = []
            for cls, n in batch_samples.items():
                label_list.extend([class_to_int[cls]] * n)
            labels = torch.tensor(label_list, dtype=torch.long, device=self.device)

            # Generazione segnali
            self.generator_model.eval()
            with torch.no_grad():
                signals = self.generator_model(noise, labels)
            signals = signals.cpu().detach().numpy()

            # Salvataggio
            signal_idx = 0
            for cls, n in batch_samples.items():
                class_dir = class_dirs[cls]
                for i in range(n):
                    signal = signals[signal_idx][0][0]  # (1, 1, N)

                    txt_filename = f"generated_signal_{generated_samples}.txt"
                    png_filename = f"generated_signal_{generated_samples}.png"

                    if self.use_s3:
                        # Save .txt
                        txt_buffer = io.StringIO()
                        np.savetxt(txt_buffer, signal)
                        txt_buffer.seek(0)
                        self.s3_client.put_object(
                            Bucket=bucket,
                            Key=os.path.join(prefix, cls, txt_filename),
                            Body=txt_buffer.getvalue(),
                        )

                        # Save plot .png
                        plt.figure(figsize=(10, 4))
                        plt.plot(signal)
                        plt.title(f"Label: {cls}")
                        plt.xlabel("Time Step")
                        plt.ylabel("Amplitude")
                        png_buffer = io.BytesIO()
                        plt.savefig(png_buffer, format="png")
                        png_buffer.seek(0)
                        self.s3_client.upload_fileobj(
                            png_buffer, bucket, os.path.join(prefix, cls, png_filename)
                        )
                        plt.close()

                    else:
                        # Save locally
                        np.savetxt(os.path.join(class_dir, txt_filename), signal)

                        plt.figure(figsize=(10, 4))
                        plt.plot(signal)
                        plt.title(f"Label: {cls}")
                        plt.xlabel("Time Step")
                        plt.ylabel("Amplitude")
                        plt.savefig(os.path.join(class_dir, png_filename))
                        plt.close()

                    generated_samples += 1
                    signal_idx += 1

            print(f"{generated_samples}/{num_samples} ECG signals generated and saved.")

        print(f"All {num_samples} samples have been generated and saved.")

