import torch
import os
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict
from abc import ABC, abstractmethod
import boto3
from urllib.parse import urlparse
import tempfile
import zipfile
import io


def weights_init(m: nn.Module) -> None:
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)


class BaseModel(ABC):
    def __init__(
        self,
        epochs,
        n_classes,
        base_dir,
        checkpoint_epoch,
        weight_decay,
        use_s3: bool = False,
        device=None,
        s3_region_name=None,
        s3_endpoint_url=None,
        s3_access_key_id=None,
        s3_secret_access_key=None,
    ):
        self.epochs = epochs
        self.n_classes = n_classes
        self.base_dir = base_dir
        self.checkpoint_epoch = checkpoint_epoch
        self.weight_decay = weight_decay
        self.use_s3 = use_s3

        self.discriminator_model = None
        self.generator_model = None

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.d_optimizer = None
        self.g_optimizer = None
        self.d_scheduler = None
        self.g_scheduler = None

        if self.base_dir is not None:
            self.create_directory(self.base_dir)
            if self.use_s3:
                self.s3_endpoint_url = s3_endpoint_url
                self.s3_region_name = (s3_region_name,)
                self.s3_access_key_id = (s3_access_key_id,)
                self.s3_secret_access_key = (s3_secret_access_key,)
                self.s3_client = boto3.client(
                    "s3",
                    aws_access_key_id=self.s3_access_key_id,
                    aws_secret_access_key=self.s3_secret_access_key,
                    region_name=self.s3_region_name,
                    endpoint_url=self.s3_endpoint_url,
                )

    def create_directory(self, path: str, is_dir: bool = True) -> None:
        """
        Creates a path either locally or on S3, depending on self.use_s3.

        Args:
            path (str): Local path or S3 URI (e.g., '/tmp/dir' or 's3://bucket-name/dir/')
            is_dir (bool): If True, ensures the path is treated as a directory (ends with '/')
        """
        if self.use_s3:
            parsed = urlparse(path)
            if parsed.scheme != "s3":
                raise ValueError(
                    f"`path` must be a valid S3 URI if use_s3=True (got '{path}')"
                )
            bucket = parsed.netloc
            key = parsed.path.lstrip("/")
            if is_dir and not key.endswith("/"):
                key += "/"

            s3 = boto3.client(
                "s3",
                aws_access_key_id=self.s3_access_key_id,
                aws_secret_access_key=self.s3_secret_access_key,
                region_name=self.s3_region_name,
                endpoint_url=self.s3_endpoint_url,
            )

            try:
                s3.put_object(Bucket=bucket, Key=key)
                print(f"S3 directory '{path}' created or already exists.")
            except Exception as e:
                raise RuntimeError(f"Error while creating path on S3: {e}")
        else:
            if is_dir:
                os.makedirs(path, exist_ok=True)
                print(f"Local directory '{path}' created or already exists.")
            else:
                parent_dir = os.path.dirname(path)
                os.makedirs(parent_dir, exist_ok=True)
                print(f"Parent directory '{parent_dir}' created or already exists.")

    def _save_models(self, epoch: int) -> None:
        save_path = os.path.join(self.base_dir, "saved_weights", f"epoch_{epoch}")
        self.save_models_to_path(save_path)

    def save_models_to_path(self, save_path: str) -> None:
        if self.use_s3:
            parsed = urlparse(save_path)
            bucket = parsed.netloc
            prefix = parsed.path.lstrip("/")
            s3 = boto3.client("s3")

            with tempfile.TemporaryDirectory() as tmp_dir:
                zip_path = os.path.join(tmp_dir, "models.zip")
                with zipfile.ZipFile(zip_path, "w") as zipf:
                    gen_path = os.path.join(tmp_dir, "generator.pth")
                    disc_path = os.path.join(tmp_dir, "discriminator.pth")

                    torch.save(self.generator_model.state_dict(), gen_path)
                    torch.save(self.discriminator_model.state_dict(), disc_path)

                    zipf.write(gen_path, "generator.pth")
                    zipf.write(disc_path, "discriminator.pth")

                s3.upload_file(zip_path, bucket, f"{prefix}/models.zip")
                print(f"Models uploaded to s3://{bucket}/{prefix}/models.zip")
        else:
            os.makedirs(save_path, exist_ok=True)
            torch.save(self.generator_model.state_dict(), f"{save_path}/generator.pth")
            torch.save(
                self.discriminator_model.state_dict(), f"{save_path}/discriminator.pth"
            )
            print(f"Models saved to {save_path}")

    def load_weights_from_path(self, path_dir: str, device: str = None) -> None:
        if device is None:
            device = self.device

        if self.use_s3:
            parsed = urlparse(path_dir)
            bucket = parsed.netloc
            prefix = parsed.path.lstrip("/")

            s3 = boto3.client(
                "s3",
                aws_access_key_id=self.s3_access_key_id,
                aws_secret_access_key=self.s3_secret_access_key,
                region_name=self.s3_region_name,
                endpoint_url=self.s3_endpoint_url,
            )

            print(f"Scaricamento da S3: bucket={bucket}, prefix={prefix}")

            gen_obj = s3.get_object(Bucket=bucket, Key=f"{prefix}/generator.pth")
            disc_obj = s3.get_object(Bucket=bucket, Key=f"{prefix}/discriminator.pth")

            gen_state = torch.load(
                io.BytesIO(gen_obj["Body"].read()), map_location=device
            )
            disc_state = torch.load(
                io.BytesIO(disc_obj["Body"].read()), map_location=device
            )

            self.generator_model.load_state_dict(gen_state)
            self.discriminator_model.load_state_dict(disc_state)

        else:
            gen_path = os.path.join(path_dir, "generator.pth")
            disc_path = os.path.join(path_dir, "discriminator.pth")

            self.generator_model.load_state_dict(
                torch.load(gen_path, map_location=device)
            )
            self.discriminator_model.load_state_dict(
                torch.load(disc_path, map_location=device)
            )

        self.generator_model.to(device)
        self.discriminator_model.to(device)
        print(f"Models loaded from {'S3' if self.use_s3 else path_dir} onto {device}")

    def update_models_weights(
        self, discriminator_weights: dict, generator_weights: dict, device: str = None
    ) -> None:
        if device is None:
            device = self.device

        self.generator_model.load_state_dict(generator_weights)
        self.discriminator_model.load_state_dict(discriminator_weights)
        self.generator_model.to(device)
        self.discriminator_model.to(device)
        print(f"Models weights updated and moved to {device}")

    def get_model_memory_usage(self, data_shape, n_samples, batch_size=64):
        total_memory = 0.0

        gen_params = sum(
            p.numel() * p.element_size() for p in self.generator_model.parameters()
        )
        gen_buffers = sum(
            b.numel() * b.element_size() for b in self.generator_model.buffers()
        )
        gen_memory = (gen_params + gen_buffers) / (1024**2)
        total_memory += gen_memory
        print(f"- Generator Memory: {gen_memory:.2f} MB")

        disc_params = sum(
            p.numel() * p.element_size() for p in self.discriminator_model.parameters()
        )
        disc_buffers = sum(
            b.numel() * b.element_size() for b in self.discriminator_model.buffers()
        )
        disc_memory = (disc_params + disc_buffers) / (1024**2)
        total_memory += disc_memory
        print(f"- Discriminator Memory: {disc_memory:.2f} MB")

        optim_memory = 0.0
        for optimizer in [self.d_optimizer, self.g_optimizer]:
            if optimizer is not None:
                for param_group in optimizer.param_groups:
                    for param in param_group["params"]:
                        optim_memory += param.numel() * param.element_size()
                        if param.grad is not None:
                            optim_memory += (
                                param.grad.numel() * param.grad.element_size()
                            )

        optim_memory /= 1024**2
        total_memory += optim_memory
        print(f"- Optimizers Memory: {optim_memory:.2f} MB")

        batch_data = torch.zeros(batch_size, *data_shape, device=self.device)
        batch_memory = (batch_data.numel() * batch_data.element_size()) / (1024**2)
        total_memory += batch_memory
        print(f"- Batch Memory: {batch_memory:.2f} MB")

        dataset_memory = (
            n_samples * batch_data.numel() * batch_data.element_size()
        ) / (1024**2)
        print(f"- Entire Dataset (theoretical): {dataset_memory:.2f} MB")

        print(f" Total Estimated Memory: {total_memory:.2f} MB")
        return total_memory

    @abstractmethod
    def build(self) -> None:
        pass

    @abstractmethod
    def train_algorithm(self, dataloader: DataLoader) -> None:
        pass

    @abstractmethod
    def generate_and_save_samples_by_class(
        self,
        num_samples: int,
        save_dir: str,
        label_percentage: Dict[str, float],
        batch_size: int,
    ) -> None:
        pass

    @abstractmethod
    def _samples(
        self,
        noise: torch.Tensor,
        labels: torch.Tensor,
        e: int,
        num_batch: int,
        save_samples: bool = True,
    ) -> torch.Tensor:
        pass
