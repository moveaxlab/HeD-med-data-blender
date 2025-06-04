import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


class ECGQualityMetrics:
    def __init__(self, real_loader, fake_loader):
        # Data loaders for real and generated (fake) ECG signals
        self.real_loader = real_loader
        self.fake_loader = fake_loader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def compute_metrics(self, real_signals, fake_signals):
        pearsons = []
        spearmans = []
        dtws = []

        for i, (real, fake) in enumerate(zip(real_signals, fake_signals)):
            # Flatten signals in case they are multidimensional
            real = np.array(real).flatten()
            fake = np.array(fake).flatten()

            # Pearson Correlation Coefficient measures linear correlation
            try:
                pearson, _ = pearsonr(real, fake)
            except ValueError as e:
                print(f"[Pearson Error @ sample {i}]: {e}")
                pearson = 0.0
            pearsons.append(pearson)

            # Spearman Rank Correlation measures monotonic relationship (non-linear)
            try:
                spearman, _ = spearmanr(real, fake)
            except ValueError as e:
                print(f"[Spearman Error @ sample {i}]: {e}")
                spearman = 0.0
            spearmans.append(spearman)

            # Dynamic Time Warping (DTW) allows comparison of time series that may vary in time or speed
            # fastdtw provides an approximate but efficient implementation
            try:
                real_list = real.tolist()
                fake_list = fake.tolist()
                dtw_distance, _ = fastdtw(
                    real_list, fake_list, dist=lambda x, y: abs(x - y)
                )
                # Normalize DTW by the length of the signal to account for varying lengths
                normalized_dtw = dtw_distance / len(real_list)
            except (ValueError, TypeError, ZeroDivisionError) as e:
                print(f"[DTW Error @ sample {i}]: {e}")
                normalized_dtw = 0.0
            dtws.append(normalized_dtw)

        return {"Pearson": pearsons, "Spearman": spearmans, "DTW": dtws}

    def evaluate(self):
        real_signals = []
        fake_signals = []

        # Collect all real signals from the data loader
        for batch in self.real_loader:
            signals, _ = batch
            real_signals.append(signals.numpy())

        # Collect all fake signals from the data loader
        for batch in self.fake_loader:
            signals, _ = batch
            fake_signals.append(signals.numpy())

        # Concatenate batches into full arrays
        real_signals = np.concatenate(real_signals, axis=0)
        fake_signals = np.concatenate(fake_signals, axis=0)

        # If no data is found, skip evaluation
        if len(real_signals) == 0 or len(fake_signals) == 0:
            print("Skipping evaluation due to missing data")
            return None

        # Ensure real and fake signal arrays have the same number of samples
        min_len = min(len(real_signals), len(fake_signals))
        real_signals = real_signals[:min_len]
        fake_signals = fake_signals[:min_len]

        # Compute per-sample metrics
        results = self.compute_metrics(real_signals, fake_signals)

        # Aggregate results by computing the mean of each metric
        aggregated_results = {
            "Pearson": float(np.mean(results["Pearson"])),
            "Spearman": float(np.mean(results["Spearman"])),
            "DTW": float(np.mean(results["DTW"])),
        }

        return aggregated_results
