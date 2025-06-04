import io
import os
import shutil
import zipfile
import tempfile
import boto3
from abc import ABC, abstractmethod
from typing import Tuple, Dict, List, Type
from torch.utils.data import DataLoader, Dataset
from urllib.parse import urlparse


class BaseDatasetReader(ABC):
    """
    Classe astratta per la lettura di dataset con immagini e classi.
    Supporta lettura da S3 se is_s3=True.
    """

    def __init__(
        self,
        dataset_path: str,
        labels: List[str],
        data_shape: Tuple[int, ...],
        batch_size: int,
        dataset_class: Type[Dataset],
        is_s3: bool = False,
        s3_region_name=None,
        s3_access_key_id=None,
        s3_secret_access_key=None,
        s3_endpoint_url=None,
    ):
        """
        Inizializzazione comune a tutti i dataset reader.

        :param dataset_path: Percorso alla directory del dataset o s3://bucket/path/to/file.zip.
        :param labels: Etichette/classi.
        :param data_shape: Shape dei dati (es. immagini: [C, H, W]).
        :param batch_size: Dimensione del batch per il DataLoader.
        :param dataset_class: Classe Dataset da istanziare.
        :param is_s3: Indica se caricare da S3.
        """
        self.dataset_path = dataset_path
        self.labels = labels
        self.data_shape = data_shape
        self.batch_size = batch_size
        self.class_counts: Dict[str, int] = {}
        self.dataset_class = dataset_class
        self.is_s3 = is_s3

        if self.is_s3:
            self.s3_region_name = s3_region_name
            self.s3_access_key_id = s3_access_key_id
            self.s3_secret_access_key = s3_secret_access_key
            self.s3_endpoint_url = s3_endpoint_url
            self.dataset_path = self._download_and_extract_from_s3(self.dataset_path)

    @abstractmethod
    def load_data(self) -> Tuple[DataLoader, Dict[str, int]]:
        """
        Caricamento del dataset, ritorna un DataLoader e i conteggi per classe.
        """
        pass

    @abstractmethod
    def compute_class_distribution(self) -> Tuple[int, Dict[str, float]]:
        """
        Calcolo della distribuzione delle classi.
        Ritorna numero totale di campioni e distribuzione percentuale per classe.
        """
        pass

    def print_class_counts(self):
        """
        Stampa dei conteggi per ciascuna classe.
        """
        print("Conteggio per classe:")
        for label, count in self.class_counts.items():
            print(f"Classe '{label}': {count}")

    def _download_and_extract_from_s3(self, s3_path: str) -> str:
        """
        Scarica ed estrae un file .zip da S3 in una directory temporanea locale.

        :param s3_path: Percorso in stile s3://bucket-name/path/to/file.zip
        :return: Percorso locale alla cartella estratta
        """
        parsed = urlparse(s3_path)
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")

        s3 = boto3.client(
            "s3",
            aws_access_key_id=self.s3_access_key_id,
            aws_secret_access_key=self.s3_secret_access_key,
            region_name=self.s3_region_name,
            endpoint_url=self.s3_endpoint_url,
        )

        print(f"Scaricamento da S3: bucket={bucket}, key={key}")
        obj = s3.get_object(Bucket=bucket, Key=key)
        zip_content = obj["Body"].read()

        temp_dir = tempfile.mkdtemp()
        with zipfile.ZipFile(io.BytesIO(zip_content)) as zf:
            zf.extractall(temp_dir)

        print(f"Dataset estratto in: {temp_dir}")
        return temp_dir

    def cleanup_temp_dir(self):
        """
        Rimuove in modo sicuro una directory temporanea e tutto il suo contenuto.

        """
        if self.is_s3:
            temp_dir = self.dataset_path

            if os.path.exists(temp_dir) and os.path.isdir(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                    print(f"Directory temporanea rimossa: {temp_dir}")
                except Exception as e:
                    print(f"Errore durante la rimozione della temp_dir: {e}")
            else:
                print(f"Directory temporanea non trovata o già rimossa: {temp_dir}")
        else:
            print("non sono state create dir temporanee")
