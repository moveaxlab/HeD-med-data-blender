import socket
import pickle
import os
import zipfile
import threading


class ServerReport:
    def __init__(
        self, model_path_fake, model_path_mia, fake_data_dir, num_clients=1, port=5000
    ):
        self.model_path_fake = model_path_fake
        self.model_path_mia = model_path_mia
        self.fake_data_dir = fake_data_dir
        self.num_clients = num_clients
        self.port = port
        self.metrics_list = []
        self.temp_zip_path = "temp_fake_data.zip"
        self.lock = threading.Lock()  # Per sincronizzare accesso a metrics_list

    def send_file(self, client_socket, file_path):
        with open(file_path, "rb") as f:
            data = f.read()
        client_socket.sendall(len(data).to_bytes(8, "big"))
        client_socket.sendall(data)

    def receive_metrics(self, client_socket):
        size = int.from_bytes(client_socket.recv(8), "big")
        data = b""
        while len(data) < size:
            data += client_socket.recv(min(4096, size - len(data)))
        return pickle.loads(data)

    def aggregate_metrics(self):
        aggregated = {}
        if not self.metrics_list:
            return aggregated
        keys = self.metrics_list[0].keys()
        for key in keys:
            aggregated[key] = sum(m[key] for m in self.metrics_list) / len(
                self.metrics_list
            )
        return aggregated

    def zip_folder(self, folder_path, zip_path):
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(folder_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, start=folder_path)
                    zipf.write(file_path, arcname=arcname)
        print(f"[SERVER] Zipped folder '{folder_path}' to '{zip_path}'")

    def handle_client(self, client_socket, addr, client_id):
        print(f"[SERVER] Handling client {client_id} at {addr}")
        try:
            self.send_file(client_socket, self.model_path_fake)
            self.send_file(client_socket, self.model_path_mia)
            self.send_file(client_socket, self.temp_zip_path)

            metrics = self.receive_metrics(client_socket)
            print(f"[SERVER] Metrics from client {client_id}: {metrics}")
            with self.lock:
                self.metrics_list.append(metrics)
        finally:
            client_socket.close()
            print(f"[SERVER] Closed connection with client {client_id}")

    def run(self):
        # Crea zip della directory dei dati fake
        self.zip_folder(self.fake_data_dir, self.temp_zip_path)

        server_socket = socket.socket()
        server_socket.bind(("0.0.0.0", self.port))
        server_socket.listen(self.num_clients)
        print(f"[SERVER] Listening on port {self.port}")

        threads = []
        try:
            for i in range(self.num_clients):
                client_socket, addr = server_socket.accept()
                print(f"[SERVER] Connected to client {i + 1} at {addr}")
                thread = threading.Thread(
                    target=self.handle_client, args=(client_socket, addr, i + 1)
                )
                thread.start()
                threads.append(thread)

            for thread in threads:
                thread.join()

        finally:
            server_socket.close()

        aggregated = self.aggregate_metrics()
        print("[SERVER] Aggregated metrics:", aggregated)

        if os.path.exists(self.temp_zip_path):
            os.remove(self.temp_zip_path)
            print(f"[SERVER] Removed temporary file '{self.temp_zip_path}'")

        return aggregated
