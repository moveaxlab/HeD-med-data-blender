import socket
import os
import pickle
from pathlib import Path
from zipfile import ZipFile
from medDataBlender.validationReport import ReportBuilder


class ClientReport:
    def __init__(
        self,
        data_type,
        real_test_dir,
        save_dir,
        port=5000,
        host="localhost",
        client_id=0,
    ):
        self.data_type = data_type
        self.real_test_dir = real_test_dir
        self.save_dir = Path(save_dir)
        self.port = port
        self.host = host
        self.client_id = client_id

        self.model_path_fake = self.save_dir / "fake_model.pth"
        self.model_path_mia = self.save_dir / "mia_model.pth"
        self.fake_data_zip = self.save_dir / "fake_data.zip"
        self.fake_data_dir = self.save_dir / "fake_data"

        self.save_dir.mkdir(exist_ok=True)
        self.fake_data_dir.mkdir(exist_ok=True)

    def receive_file(self, sock, save_path):
        size = int.from_bytes(sock.recv(8), "big")
        print(
            f"[CLIENT {self.client_id}] Receiving file {save_path.name} ({size} bytes)"
        )
        received = 0
        with open(save_path, "wb") as f:
            while received < size:
                chunk = sock.recv(min(4096, size - received))
                if not chunk:
                    break
                f.write(chunk)
                received += len(chunk)
                if received % (1024 * 1024) < 4096:
                    print(
                        f"[CLIENT {self.client_id}] Received {received}/{size} bytes..."
                    )
        print(f"[CLIENT {self.client_id}] Finished receiving {save_path.name}")

    def extract_zip(self, zip_path, extract_to):
        print(f"[CLIENT {self.client_id}] Extracting zip file {zip_path}")
        with ZipFile(zip_path, "r") as zip_ref:
            zip_contents = zip_ref.namelist()
            total_files = len(zip_contents)
            print(f"[CLIENT {self.client_id}] Zip contains {total_files} files")

            for idx, name in enumerate(zip_contents, 1):
                zip_ref.extract(name, extract_to)
                if idx % max(1, total_files // 10) == 0 or idx == total_files:
                    percent = (idx / total_files) * 100
                    print(
                        f"[CLIENT {self.client_id}] Extraction progress: {percent:.0f}%"
                    )

        print(f"[CLIENT {self.client_id}] Extraction completed to {extract_to}")

    def send_metrics(self, sock, metrics):
        data = pickle.dumps(metrics)
        sock.sendall(len(data).to_bytes(8, "big"))
        sock.sendall(data)
        print(f"[CLIENT {self.client_id}] Metrics sent ({len(data)} bytes)")

    def run(self):
        sock = socket.socket()
        sock.connect((self.host, self.port))
        print(f"[CLIENT {self.client_id}] Connected to server.")

        self.receive_file(sock, self.model_path_fake)
        self.receive_file(sock, self.model_path_mia)
        self.receive_file(sock, self.fake_data_zip)

        self.extract_zip(self.fake_data_zip, self.fake_data_dir)

        builder = ReportBuilder(
            base_dir=str(self.save_dir),
            real_dir=self.real_test_dir,
            fake_dir=str(self.fake_data_dir),
            real_test_dir=self.real_test_dir,
            fake_test_dir=str(self.fake_data_dir),
            model_path_fake=str(self.model_path_fake),
            model_path_mia=str(self.model_path_mia),
            data_type=self.data_type,
        )

        metrics = builder.compute_metrics()
        print(f"[CLIENT {self.client_id}] Computed metrics: {metrics}")

        self.send_metrics(sock, metrics)
        sock.close()
        print(f"[CLIENT {self.client_id}] Connection closed.")
