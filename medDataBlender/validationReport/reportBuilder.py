from torch.utils.data import DataLoader

from medDataBlender.models.dataLoaders import OasisMRI_ReadDataset, Mitbih_ReadDataset
from medDataBlender.models.evaluator import ImageEvaluator, ECGQualityMetrics
from .privacyMetrics import MIA, CombinedMIA_Dataset, compute_k_anonymity
from .classificator_models import ResNetMRIClassifier, ResNetECGClassifier


class ReportBuilder:
    def __init__(
        self,
        base_dir,
        real_dir,
        fake_dir,
        real_test_dir,
        fake_test_dir,
        batch_size=16,
        model_path_mia=None,
        model_path_fake=None,
        num_classes=2,
        lr=0.001,
        num_epochs=10,
        data_type="ECG",  # or "MRI"
        ecg_shape=(1, 1, 187),
        mri_shape=(256, 256),
    ):
        self.base_dir = base_dir
        self.real_dir = real_dir
        self.fake_dir = fake_dir
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.lr = lr
        self.num_epochs = num_epochs
        self.model_path_mia = model_path_mia
        self.model_path_fake = model_path_fake
        self.data_type = data_type.lower()
        self.ecg_shape = ecg_shape
        self.mri_shape = mri_shape

        self.real_test_loader = self.create_dataloader(real_test_dir)
        self.fake_test_loader = self.create_dataloader(fake_test_dir)

        mia_dataset_test = CombinedMIA_Dataset(
            self.real_test_loader, self.fake_test_loader
        )
        self.mia_loader_test = DataLoader(
            mia_dataset_test, batch_size=batch_size, shuffle=True
        )

        self.real_loader = None
        self.fake_loader = None
        self.mia_loader_train = None

        if not model_path_fake or not model_path_mia:
            self.real_loader = self.create_dataloader(real_dir)
            self.fake_loader = self.create_dataloader(fake_dir)
            if not model_path_mia:
                mia_dataset_train = CombinedMIA_Dataset(
                    self.real_loader, self.fake_loader
                )
                self.mia_loader_train = DataLoader(
                    mia_dataset_train, batch_size=batch_size, shuffle=True
                )

    def create_dataloader(self, data_dir):
        if self.data_type == "mri":
            dataset_reader = OasisMRI_ReadDataset(
                dataset_path=data_dir,
                labels=["Non_Demented", "Very_mild_Dementia"],
                data_shape=self.mri_shape,
                batch_size=self.batch_size,
            )
        elif self.data_type == "ecg":
            dataset_reader = Mitbih_ReadDataset(
                dataset_path=data_dir,
                labels=[
                    "Non-Ectopic Beats",
                    "Superventrical Ectopic",
                    "Ventricular Beats",
                    "Unknown",
                    "Fusion Beats",
                ],
                data_shape=self.ecg_shape,
                batch_size=self.batch_size,
            )
        else:
            raise ValueError(f"Unsupported data_type '{self.data_type}'")

        loader, _ = dataset_reader.load_data()
        return loader

    def initialize_classifier(
        self,
        datat_type,
        model_path=None,
        dataloader=None,
        isMIA=False,
        save_dir="classifier_weights",
    ):
        n_classes = 2 if isMIA else self.num_classes

        if datat_type == "mri":
            classifier = ResNetMRIClassifier(
                n_classes, self.lr, self.num_epochs, save_dir
            )
        elif datat_type == "ecg":
            classifier = ResNetECGClassifier(
                n_classes, self.lr, self.num_epochs, save_dir
            )
        else:
            raise ValueError(f"Unsupported data_type '{datat_type}'")

        if model_path:
            classifier.load_model(model_path)
            return classifier
        else:
            classifier.train(dataloader)
            return classifier

    def compute_metrics(self):
        metrics = {}

        classifier = self.initialize_classifier(
            datat_type=self.data_type,
            model_path=self.model_path_fake,
            dataloader=self.fake_loader,
            save_dir=self.base_dir + "/classifier_" + self.data_type,
        )

        metrics["ML Accuracy"] = classifier.evaluate(self.real_test_loader)

        mia_classifier = self.initialize_classifier(
            datat_type=self.data_type,
            model_path=self.model_path_mia,
            dataloader=self.mia_loader_train,
            save_dir=self.base_dir + "/mia_attack_model_" + self.data_type,
            isMIA=True,
        )

        mia = MIA(mia_classifier, self.mia_loader_test)
        metrics["MIA Accuracy"] = mia.attack()

        k_anonymity_distribution = compute_k_anonymity(
            self.real_test_loader, self.fake_test_loader, self.data_type
        )

        metrics["k-Anonymity"] = k_anonymity_distribution

        # Conversione dei dataloader in liste [(x, y), ...]
        real_data_list = [(x, y) for x, y in self.real_test_loader]
        fake_data_list = [(x, y) for x, y in self.fake_test_loader]

        # Usa la nuova struttura astratta
        if self.data_type == "mri":
            evaluator = ImageEvaluator(real_data_list, fake_data_list)
            metrics.update(evaluator.evaluate())

            metrics["FID"] = classifier.calculate_average_fid(
                self.real_test_loader, self.fake_test_loader
            )

        elif self.data_type == "ecg":
            evaluator = ECGQualityMetrics(real_data_list, fake_data_list)
            metrics.update(evaluator.evaluate())

        return metrics
