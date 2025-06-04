import argparse
import os
import pickle
import socket
import time
import lz4.frame  # Fast compression
import logging
from typing import Dict, Optional
import torch

from medDataBlender.models import BaseModel
from medDataBlender.models.dataLoaders import BaseDatasetReader
from torch.utils.data import DataLoader

# -------------------------
# CONFIGURATION
# -------------------------

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

NUM_ROUNDS = 5
CHUNK_SIZE = 64 * 1024  # 64 KB


# -------------------------
# NETWORK UTILS
# -------------------------


def create_socket_and_connect(host: str, port: int) -> socket.socket:
    """
    Create and connect a socket to the server, retrying on failure.

    Args:
        host (str): Hostname or IP address of the server.
        port (int): Port number to connect to.

    Returns:
        socket.socket: Connected socket.
    """
    while True:
        try:
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.connect((host, port))
            logging.info(f"Connection established with {host}:{port}")
            return client_socket
        except ConnectionRefusedError:
            logging.warning("Connection refused. Retrying in 5 seconds...")
            time.sleep(5)
        except Exception as e:
            logging.error(f"Connection error: {e}")
            time.sleep(10)


# -------------------------
# COMMUNICATION FUNCTIONS
# -------------------------


def receive_weights_from_server(
    connection: socket.socket,
) -> Optional[Dict[str, Dict[str, torch.Tensor]]]:
    """
    Receives model weights from the server over a socket.

    Args:
        connection (socket.socket): Connected socket.

    Returns:
        Optional[Dict[str, Dict[str, torch.Tensor]]]: Deserialized model weights, or None if failed.
    """
    try:
        # Read the first 8 bytes for payload size
        data_length_bytes = connection.recv(8)
        if not data_length_bytes:
            raise ValueError("No data received for size.")

        data_length = int.from_bytes(data_length_bytes, "big")
        logging.info(f"Receiving: {data_length} bytes expected...")

        # Receive payload in chunks
        data = b""
        received_bytes = 0
        while received_bytes < data_length:
            packet = connection.recv(min(CHUNK_SIZE, data_length - received_bytes))
            if not packet:
                raise ConnectionError("Connection lost while receiving data.")
            data += packet
            received_bytes += len(packet)
            logging.info(
                f"Received: {received_bytes}/{data_length} bytes ({(received_bytes / data_length) * 100:.2f}%)"
            )

        if received_bytes != data_length:
            raise ValueError(
                f"Incomplete data received ({received_bytes}/{data_length} bytes)."
            )

        # Decompress and deserialize
        decompressed_data = lz4.frame.decompress(data)
        weights = pickle.loads(decompressed_data)
        logging.info("Weights received and deserialized successfully.")
        return weights

    except Exception as e:
        logging.error(f"Error receiving weights: {e}")
        return None

    finally:
        connection.close()


def send_weights_to_server(
    client_socket: socket.socket,
    client_id: int,
    generator: torch.nn.Module,
    discriminator: torch.nn.Module,
) -> None:
    """
    Sends local model weights to the federated server.

    Args:
        client_socket (socket.socket): Connected socket.
        client_id (int): Unique identifier for the client.
        generator (torch.nn.Module): Local generator model.
        discriminator (torch.nn.Module): Local discriminator model.

    Returns:
        None
    """
    try:
        # Pack model weights
        weights = {
            "generator_weights": generator.state_dict(),
            "discriminator_weights": discriminator.state_dict(),
        }

        payload = pickle.dumps({"client_id": client_id, "weights": weights})
        compressed_payload = lz4.frame.compress(payload)

        # Send size first
        client_socket.sendall(len(compressed_payload).to_bytes(8, "big"))

        # Send payload in chunks
        sent_bytes = 0
        for i in range(0, len(compressed_payload), CHUNK_SIZE):
            chunk = compressed_payload[i : i + CHUNK_SIZE]
            client_socket.sendall(chunk)
            sent_bytes += len(chunk)
            logging.info(
                f"Client {client_id}: sent {sent_bytes}/{len(compressed_payload)} bytes "
                f"({(sent_bytes / len(compressed_payload)) * 100:.2f}%)"
            )

        logging.info(
            f"Client {client_id}: weights sent to server ({len(compressed_payload)} bytes)."
        )

    except Exception as e:
        logging.error(f"Error sending weights to server: {e}")
    finally:
        client_socket.close()


# -------------------------
# TRAINING LOOP
# -------------------------


def train_over_runs(
    model: BaseModel,
    train_loader: DataLoader,
    device: torch.device,
    client_id: int,
    host: str,
    send_port: int,
    receive_port: int,
) -> None:
    """
    Orchestrates the federated training process for a client.

    Args:
        model (BaseModel): Local GAN model.
        train_loader (BaseDatasetReader): Training data loader.
        device (torch.device): Computation device (CPU/GPU).
        client_id (int): Unique client identifier.
        host (str): Server address.
        send_port (int): Port to send weights to server.
        receive_port (int): Port to receive weights from server.

    Returns:
        None
    """
    print(f"Client {client_id} process started.")

    # Move models to device
    generator = model.generator_model.to(device)
    discriminator = model.discriminator_model.to(device)

    # Initial weights from server
    with create_socket_and_connect(host, send_port) as client_socket:
        weights = receive_weights_from_server(client_socket)
    generator.load_state_dict(weights["generator_weights"])
    discriminator.load_state_dict(weights["discriminator_weights"])

    # Start federated learning rounds
    for round_num in range(NUM_ROUNDS):
        print(f"Client {client_id} - Round {round_num} started.")

        # Train locally (to be implemented)
        # model.train_algorithm(train_loader)

        # Send updated weights
        with create_socket_and_connect(host, receive_port) as client_socket_send:
            send_weights_to_server(
                client_socket_send, client_id, generator, discriminator
            )

        # Receive aggregated weights
        with create_socket_and_connect(host, send_port) as client_socket_receive:
            weights = receive_weights_from_server(client_socket_receive)

        generator.load_state_dict(weights["generator_weights"])
        discriminator.load_state_dict(weights["discriminator_weights"])

        # Clean up
        del weights
        torch.cuda.empty_cache()

    print(f"Client {client_id} completed all {NUM_ROUNDS} training rounds.")
