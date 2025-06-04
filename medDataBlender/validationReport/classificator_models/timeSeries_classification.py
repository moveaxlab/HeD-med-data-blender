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
import torch
import torch.nn as nn
import torch.optim as optim


class ResNetECGClassifier:
    def __init__(self, num_classes, lr=0.001, num_epochs=10, save_dir="ecg_weights"):
        self.num_classes = num_classes
        self.lr = lr
        self.num_epochs = num_epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_dir = save_dir
        self.model = self._build_model()

    def _build_model(self):
        class ECGNet(nn.Module):
            def __init__(self, num_classes):
                super(ECGNet, self).__init__()
                self.net = nn.Sequential(
                    nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=2),
                    nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=2),
                    nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=2),
                    nn.Conv1d(128, 256, kernel_size=5, stride=1, padding=2),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=2),
                    nn.Conv1d(256, 512, kernel_size=5, stride=1, padding=2),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=2),
                )
                self.flatten = nn.Flatten()
                self.classifier = nn.Sequential(
                    nn.Linear(512 * 5, 1024),  # per segnali di lunghezza 187
                    nn.ReLU(),
                    nn.Linear(1024, num_classes),
                )

            def forward(self, x):
                # Assicura la forma corretta: [B, 1, 187]
                if x.ndim == 4 and x.shape[2] == 1:
                    x = x.squeeze(2)  # [B, 1, 1, 187] → [B, 1, 187]
                x = self.net(x)
                x = self.flatten(x)
                x = self.classifier(x)
                return x

        return ECGNet(self.num_classes).to(self.device)

    def train(self, train_loader):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0

            for signals, labels in train_loader:
                signals, labels = signals.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(signals)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f" Epoch [{epoch + 1}/{self.num_epochs}] - Loss: {total_loss:.4f}")
            model_path = os.path.join(self.save_dir, f"ecg_epoch_{epoch + 1}.pth")
            torch.save(self.model.state_dict(), model_path)
            print(f" Model saved at {model_path}")

    def evaluate(self, test_loader):
        self.model.eval()
        correct = 0
        total = 0
        class_correct = torch.zeros(self.num_classes).to(self.device)
        class_total = torch.zeros(self.num_classes).to(self.device)

        with torch.no_grad():
            for signals, labels in test_loader:
                signals, labels = signals.to(self.device), labels.to(self.device)
                outputs = self.model(signals)
                _, predicted = torch.max(outputs, 1)

                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                for i in range(labels.size(0)):
                    class_correct[labels[i]] += (predicted[i] == labels[i]).item()
                    class_total[labels[i]] += 1

        acc = 100 * correct / total
        print(f" Test Accuracy: {acc:.2f}%")

        print("Class-wise accuracy:")
        for i in range(self.num_classes):
            if class_total[i] > 0:
                class_acc = 100 * class_correct[i] / class_total[i]
                print(f"   Class {i}: {class_acc:.2f}%")
            else:
                print(f"   Class {i}: No samples in test data")

        return acc

    def save_model(self, model_path):
        torch.save(self.model.state_dict(), model_path)
        print(f" Model saved at {model_path}")

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        print(f" Model loaded from {model_path}")
