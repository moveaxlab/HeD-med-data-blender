import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, kernel_size=4, padding=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size, stride=1, padding=padding
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.project = None
        if stride != 1 or in_channels != out_channels:
            self.project = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, padding=0
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.project is not None:
            identity = self.project(identity)

        # Allineamento forzato delle dimensioni spaziali
        if identity.shape[2:] != out.shape[2:]:
            identity = F.interpolate(
                identity, size=out.shape[2:], mode="bilinear", align_corners=False
            )

        return out + identity


class Discriminator(nn.Module):
    def __init__(self, image_size=64, feature_channels=1):
        super(Discriminator, self).__init__()

        self.residual_blocks = nn.Sequential(
            ResidualBlock(feature_channels, 32, stride=2),
            ResidualBlock(32, 64, stride=2),
            ResidualBlock(64, 128, stride=2),
        )

        # Dummy forward per determinare output shape dinamicamente
        dummy_input = torch.zeros(1, feature_channels, image_size, image_size)
        dummy_output = self.residual_blocks(dummy_input)
        self.flat_features = dummy_output.view(1, -1).shape[1]

        self.fc = nn.Linear(self.flat_features, 1)

    def forward(self, x):
        x = self.residual_blocks(x)
        x = rearrange(x, "b c h w -> b (c h w)")
        x = self.fc(x)
        return x


class GeneratorResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(self, image_size=128, latent_dim=256, image_channels=1):
        super().__init__()
        self.latent_dim = latent_dim
        self.init_size = image_size // 8  # 128 -> 16

        self.fc = nn.Linear(latent_dim, self.init_size * self.init_size * latent_dim)

        self.upsample1 = nn.Sequential(
            nn.BatchNorm2d(latent_dim),
            nn.ConvTranspose2d(
                latent_dim, 256, kernel_size=4, stride=2, padding=1
            ),  # 16 -> 32
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.res1 = GeneratorResidualBlock(256)

        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(
                256, 128, kernel_size=4, stride=2, padding=1
            ),  # 32 -> 64
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.res2 = GeneratorResidualBlock(128)

        self.upsample3 = nn.Sequential(
            nn.ConvTranspose2d(
                128, 64, kernel_size=4, stride=2, padding=1
            ),  # 64 -> 128
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.res3 = GeneratorResidualBlock(64)

        self.final_conv = nn.Sequential(
            nn.Conv2d(64, image_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),  # Output in range [-1, 1]
        )

    def forward(self, z):
        out = self.fc(z)
        out = rearrange(
            out,
            "b (c h w) -> b c h w",
            c=self.latent_dim,
            h=self.init_size,
            w=self.init_size,
        )

        out = self.upsample1(out)
        out = self.res1(out)

        out = self.upsample2(out)
        out = self.res2(out)

        out = self.upsample3(out)
        out = self.res3(out)

        img = self.final_conv(out)
        return img
