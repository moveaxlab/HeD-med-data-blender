import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class Generator(nn.Module):
    """
    The Generator model for the GAN.

    It consists of a fully connected layer followed by several convolutional transpose layers
    to progressively upsample the input latent vector into an image. The image is output after a
    final convolutional layer with Tanh activation.

    Parameters:
        input_dim (int): The input dimension (height or width) of the final generated image.
        latent (int): The size of the latent space (dimensionality of the random noise vector).
        image_channels (int): The number of channels in the generated image (default is 1 for grayscale images).
    """

    def __init__(self, input_dim, latent, image_channels=1):
        """
        Initializes the layers of the Generator.

        Parameters:
            input_dim (int): The height/width of the generated image.
            latent (int): The size of the latent vector (random noise vector).
            image_channels (int): The number of channels in the generated image (default is 1).

        Returns:
            None
        """
        super(Generator, self).__init__()
        self.latent_size = latent
        self.input_dim = input_dim
        # Calculate initial size after downsampling
        self.init_size = input_dim // 8  # The final size after 3 upsampling steps

        self.fc = nn.Linear(latent, self.init_size * self.init_size * latent)

        # Convolutional blocks for upsampling
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(self.latent_size),
            # First upsampling (from 16x16 to 32x32)
            nn.ConvTranspose2d(
                self.latent_size, 256, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # Second upsampling (from 32x32 to 64x64)
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # Third upsampling (from 64x64 to 128x128)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # Final layer to refine image to 256x256
            nn.Conv2d(64, image_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),  # Activation function to output values in the range [-1, 1]
        )

    def forward(self, z):
        """
        Forward pass through the Generator.

        Parameters:
            z (Tensor): A tensor representing the latent vector with shape (batch_size, latent_size).

        Returns:
            Tensor: A tensor representing the generated image with shape (batch_size, image_channels, height, width).
        """
        out = self.fc(z)
        # Reshape the output tensor to be passed through the ConvTranspose2d layers using einops
        out = rearrange(
            out,
            "b (c h w) -> b c h w",
            c=self.latent_size,
            h=self.init_size,
            w=self.init_size,
        )
        # Pass through the convolutional layers
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    """
    The Discriminator model for the GAN.

    It consists of three convolutional layers, each followed by batch normalization,
    LeakyReLU activation, and dropout. The output is passed through a fully connected layer
    to classify the input image as real or fake.

    Parameters:
        image_size (int): The size of the input image (height or width).
        feature_channels (int): The number of input channels (e.g., 3 for RGB images).
    """

    def __init__(self, image_size, feature_channels):
        """
        Initializes the layers of the Discriminator.

        Parameters:
            image_size (int): The height/width of the input image.
            feature_channels (int): The number of channels in the input image (e.g., 3 for RGB).

        Returns:
            None
        """
        super(Discriminator, self).__init__()
        final_size = (
            image_size // 8
        )  # The final size after 3 convolution layers with stride=2

        # First convolution layer: takes input with `feature_channels` channels and outputs 32 channels
        self.conv1 = nn.Conv2d(feature_channels, 32, 4, 2, 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.dropout1 = nn.Dropout(0.4)

        # Second convolution layer: takes 32 channels and outputs 64 channels
        self.conv2 = nn.Conv2d(32, 64, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.dropout2 = nn.Dropout(0.4)

        # Third convolution layer: takes 64 channels and outputs 128 channels
        self.conv3 = nn.Conv2d(64, 128, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(128)
        self.dropout3 = nn.Dropout(0.4)

        # Fully connected layer that outputs a single scalar value (real/fake)
        self.fc = nn.Linear(128 * final_size * final_size, 1)

    def forward(self, x):
        """
        Forward pass through the Discriminator.

        Parameters:
            x (Tensor): A tensor representing a batch of images of shape
                        (batch_size, feature_channels, height, width).

        Returns:
            Tensor: A tensor of shape (batch_size, 1) where each value is the classification score
                    (real/fake) for each input image in the batch.
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x, 0.2)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x, 0.2)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.leaky_relu(x, 0.2)
        x = self.dropout3(x)

        # Flatten the output to feed into the fully connected layer using einops
        x = rearrange(x, "b c h w -> b (c h w)")
        x = self.fc(x)  # Output: scalar classification score (real/fake)

        return x
