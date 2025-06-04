# -----------------------------------------------------------------------------
# ResNetECGClassifier
#
# This 1D CNN-based classifier is designed for processing ECG time-series signals.
# It is used in two primary contexts:
#
# 1. **Membership Inference Attacks (MIA)**:
#    The classifier can be trained on real or generated data to assess
#    the ability of an adversary to distinguish between training and non-training samples.
#
# 2. **Machine Learning Quality Evaluation**:
#    It is also used to assess the utility of synthetic ECG data by training the classifier
#    on generated data and evaluating performance on real test data (or vice versa).
#
# The network consists of several convolutional and pooling layers followed by
# fully connected layers tailored for ECG signals of fixed length (e.g., 187 samples).
# -----------------------------------------------------------------------------


import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from scipy.linalg import sqrtm


def calculate_fid(real_features, fake_features):
    """
    Compute Fréchet Inception Distance (FID) score.
    """
    mu_r, sigma_r = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
    mu_g, sigma_g = np.mean(fake_features, axis=0), np.cov(fake_features, rowvar=False)

    mean_diff = np.sum((mu_r - mu_g) ** 2)
    cov_mean = sqrtm(sigma_r @ sigma_g)

    if np.iscomplexobj(cov_mean):
        cov_mean = cov_mean.real  # Remove imaginary component if any

    fid = mean_diff + np.trace(sigma_r + sigma_g - 2 * cov_mean)
    return fid


class ResNetMRIClassifier:
    def __init__(self, num_classes, lr=0.001, num_epochs=10, save_dir="resNet_weights"):
        """
        Initialize the classifier with training and testing DataLoaders.
        """

        self.num_classes = num_classes
        self.lr = lr
        self.num_epochs = num_epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model()
        self.save_dir = save_dir

    def _build_model(self):
        """
        Build a ResNet18 model adapted for grayscale images and number of classes.
        Uses BCEWithLogitsLoss for binary classification (1 output unit),
        and CrossEntropyLoss for multi-class (>=2 classes).
        """
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)

        # Adatta il primo layer a immagini in scala di grigi (1 canale)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        model.fc = nn.Linear(model.fc.in_features, self.num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

        return model.to(self.device)

    def train(self, train_loader):
        """
        Train the model and save checkpoints.
        """
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0

            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f" Epoch [{epoch + 1}/{self.num_epochs}] - Loss: {total_loss:.4f}")

            # Save model checkpoint
            model_path = os.path.join(self.save_dir, f"model_epoch_{epoch + 1}.pth")
            torch.save(self.model.state_dict(), model_path)
            print(f" Model saved at {model_path}")

    def evaluate(self, test_loader):
        """
        Evaluate the model's performance on the test dataset.
        """
        self.model.eval()
        correct = 0
        total = 0
        class_correct = torch.zeros(self.num_classes, dtype=torch.int64).to(self.device)
        class_total = torch.zeros(self.num_classes, dtype=torch.int64).to(self.device)

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)

                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                for i in range(labels.size(0)):
                    label = labels[i]
                    class_correct[label] += (predicted[i] == label).item()
                    class_total[label] += 1

        acc = 100 * correct / total
        print(f" Test Accuracy: {acc:.2f}%")

        print("Class-wise accuracy:")
        for i in range(self.num_classes):
            if class_total[i] > 0:
                class_acc = 100 * class_correct[i] / class_total[i]
                print(f"   Class {i}: {class_acc:.2f}%")
            else:
                print(f"   Class {i}: No examples in test data")

        return acc

    def save_model(self, model_path):
        """
        Save the trained model.
        """
        torch.save(self.model.state_dict(), model_path)
        print(f" Model saved at {model_path}")

    def load_model(self, model_path):
        """
        Load a trained model.
        """
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        print(f" Model loaded from {model_path}")

    def calculate_average_fid(self, real_loader, fake_loader):
        """
        Compute the normalized average Fréchet Inception Distance (FID) score over all batches.
        The score is normalized between 0.0 and 1.0 using the maximum FID computed on black vs white images.
        """
        fid_scores = []

        # Calcola la massima FID tra immagini nere e bianche della stessa dimensione
        def estimate_fid_max(example_shape):
            """
            Calculate the maximum FID by comparing black and white images,
            assuming images are in the same normalized range as the real data.
            """
            with torch.no_grad():
                # Create black and white images, assuming [-1, 1] normalization
                black_images = torch.full(
                    example_shape, -1.0, device=self.device
                )  # Normalized to -1
                white_images = torch.full(
                    example_shape, 1.0, device=self.device
                )  # Normalized to 1

                # Extract features
                black_features = self.extract_features(black_images)
                white_features = self.extract_features(white_images)

            return calculate_fid(black_features, white_features)

        self.model.eval()
        example_shape = None

        with torch.no_grad():
            for real_images, fake_images in zip(real_loader, fake_loader):
                real_images, fake_images = real_images[0].to(self.device), fake_images[
                    0
                ].to(self.device)

                if example_shape is None:
                    example_shape = real_images.shape

                real_features = self.extract_features(real_images)
                fake_features = self.extract_features(fake_images)

                fid_score = calculate_fid(real_features, fake_features)
                fid_scores.append(fid_score)

        average_fid = np.mean(fid_scores)
        max_fid = estimate_fid_max(example_shape)
        print(max_fid)
        print(average_fid)
        normalized_fid = np.clip(average_fid / max_fid, 0.0, 1.0)
        print(f"Normalized Average FID score: {normalized_fid:.4f}")
        return normalized_fid

    def extract_features(self, images):
        """
        Extract features from the penultimate layer of the model (before the fully connected layer).
        """
        # Disable the fully connected layers (classifier) for feature extraction
        feature_extractor = nn.Sequential(*list(self.model.children())[:-1])

        # Get features from the penultimate layer
        with torch.no_grad():
            features = feature_extractor(images)
            features = features.view(features.size(0), -1)  # Flatten the features
        return features.cpu().numpy()
