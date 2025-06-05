import socket
import pickle
import torch
import threading
import lz4.frame  # Fast compression
import logging
import queue

from typing import Type

from medDataBlender.models.evaluator import SyntheticDataEvaluator
from medDataBlender.models import BaseModel
from torch.utils.data import DataLoader
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

NUM_ROUNDS = 5
CHUNK_SIZE = 64 * 1024  # 64 KB

# -------------------------
# CLIENT COMMUNICATION
# -------------------------


def send_weights_to_client(
    connection: socket.socket,
    generator: torch.nn.Module,
    discriminator: torch.nn.Module,
) -> None:
    """
    Sends the model weights (generator and discriminator) to a client over a network connection.

    Args:
        connection (socket.socket): The socket connection to the client.
        generator (torch.nn.Module): The generator model.
        discriminator (torch.nn.Module): The discriminator model.

    Returns:
        None
    """
    try:
        weights = {
            "generator_weights": generator.state_dict(),
            "discriminator_weights": discriminator.state_dict(),
        }

        serialized_weights = pickle.dumps(weights)
        compressed_weights = lz4.frame.compress(serialized_weights)

        # Send size of payload first (8 bytes, big-endian)
        connection.sendall(len(compressed_weights).to_bytes(8, "big"))

        # Send compressed weights in chunks
        for i in range(0, len(compressed_weights), CHUNK_SIZE):
            connection.sendall(compressed_weights[i : i + CHUNK_SIZE])

        logging.info(f"Weights sent to client {connection.getpeername()}")
    except Exception as e:
        logging.error(f"Error sending weights: {e}")
    finally:
        connection.close()


def send_weights_to_clients(
    host: str,
    send_port: int,
    generator: torch.nn.Module,
    discriminator: torch.nn.Module,
    num_clients: int,
) -> None:
    """
    Sends model weights to multiple clients.

    Args:
        host (str): The host address of the server.
        send_port (int): Port used to send weights.
        generator (torch.nn.Module): Generator model.
        discriminator (torch.nn.Module): Discriminator model.
        num_clients (int): Number of expected clients.

    Returns:
        None
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((host, send_port))
        server_socket.listen()
        logging.info(f"Server listening to send weights on port {send_port}...")

        threads = []
        for _ in range(num_clients):
            connection, address = server_socket.accept()
            logging.info(f"Connection established with {address}")
            thread = threading.Thread(
                target=send_weights_to_client,
                args=(connection, generator, discriminator),
            )
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()


def handle_client(
    connection: socket.socket, received_weights: dict, lock: threading.Lock
):
    try:
        connection.settimeout(10)
        size_bytes = connection.recv(8)
        if not size_bytes:
            raise ConnectionError("No size bytes received")
        total_size = int.from_bytes(size_bytes, "big")
        logging.info(
            f"Ricevuto dimensione payload: {total_size} byte da {connection.getpeername()}"
        )

        data = b""
        received_len = 0
        last_percent_reported = -1

        while received_len < total_size:
            chunk = connection.recv(min(CHUNK_SIZE, total_size - received_len))
            if not chunk:
                raise ConnectionError("Connection lost while receiving data")
            data += chunk
            received_len += len(chunk)

            # Calcola percentuale
            percent = int((received_len / total_size) * 100)
            if percent != last_percent_reported and percent % 5 == 0:  # stampa ogni 5%
                logging.info(
                    f"Ricezione da {connection.getpeername()}: {percent}% completato"
                )
                last_percent_reported = percent

            # invia ack chunk ricevuto (1 byte)
            connection.sendall(b"\x01")

        decompressed = lz4.frame.decompress(data)
        received_data = pickle.loads(decompressed)

        client_id = received_data["client_id"]
        client_weights = received_data["weights"]

        with lock:
            received_weights[client_id] = client_weights

        logging.info(f"Pesi ricevuti con successo dal client {client_id}")

    except Exception as e:
        logging.error(f"Errore nella ricezione da client: {e}")

    finally:
        connection.close()


def receive_weights_from_clients(host: str, port: int, num_clients: int) -> dict:
    received_weights = {}
    lock = threading.Lock()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((host, port))
        server_socket.listen()
        logging.info(f"Server in ascolto su {host}:{port}...")

        with ThreadPoolExecutor(max_workers=num_clients) as executor:
            futures = []
            for _ in range(num_clients):
                conn, addr = server_socket.accept()
                logging.info(f"Connessione accettata da {addr}")
                futures.append(
                    executor.submit(handle_client, conn, received_weights, lock)
                )

            for future in futures:
                future.result()  # aspetta completamento thread

    return received_weights


# -------------------------
# AGGREGATION ALGORITHMS
# -------------------------


def FedPIDAvg(
    client_weights: list[dict],
    actual_costs: list[float],
    previous_costs: list[float],
    dataset_sizes: list[int],
    alpha: float = 0.45,
    beta: float = 0.45,
    gamma: float = 0.1,
) -> dict:
    """
    Aggregates model weights using the FedPID strategy.

    Args:
        client_weights (list): List of client weight dictionaries.
        actual_costs (list): Actual costs for each client.
        previous_costs (list): Previous costs for each client.
        dataset_sizes (list): Size of each client's dataset.
        alpha (float): Coefficient for cost delta.
        beta (float): Coefficient for cumulative cost.
        gamma (float): Coefficient for dataset size.

    Returns:
        dict: Aggregated weights.
    """
    num_clients = len(client_weights)
    assert len(actual_costs) == num_clients
    assert len(previous_costs) == num_clients
    assert len(dataset_sizes) == num_clients

    weights = []
    for i in range(num_clients):
        delta_cost = abs(previous_costs[i] - actual_costs[i])
        sum_cost = sum(previous_costs)
        dataset_size = dataset_sizes[i]
        weight = (alpha * delta_cost) + (beta * sum_cost) + (gamma * dataset_size)
        weights.append(weight)

    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]

    aggregated_weights = {}
    for key in client_weights[0].keys():
        aggregated_weights[key] = torch.zeros_like(client_weights[0][key])
        for i in range(num_clients):
            aggregated_weights[key] += weights[i] * client_weights[i][key]

    return aggregated_weights


def fedavg(weights_dict: dict) -> tuple[dict, dict]:
    """
    Aggregates model weights using standard Federated Averaging (FedAvg).

    Args:
        weights_dict (dict): Dictionary containing weights from all clients.

    Returns:
        tuple: Aggregated (discriminator_weights, generator_weights)
    """
    if not weights_dict:
        raise ValueError("The weights dictionary is empty")

    first_client_id = next(iter(weights_dict))

    agg_discriminator_weights = {
        k: torch.zeros_like(v, dtype=torch.float32)
        for k, v in weights_dict[first_client_id]["discriminator_weights"].items()
    }

    agg_generator_weights = {
        k: torch.zeros_like(v, dtype=torch.float32)
        for k, v in weights_dict[first_client_id]["generator_weights"].items()
    }

    num_clients = len(weights_dict)

    for client_weights in weights_dict.values():
        for layer, weights in client_weights["discriminator_weights"].items():
            agg_discriminator_weights[layer] += weights
        for layer, weights in client_weights["generator_weights"].items():
            agg_generator_weights[layer] += weights

    for layer in agg_discriminator_weights:
        agg_discriminator_weights[layer] /= num_clients
    for layer in agg_generator_weights:
        agg_generator_weights[layer] /= num_clients

    return agg_discriminator_weights, agg_generator_weights


# -------------------------
# FEDERATED LEARNING SERVER
# -------------------------


def server(
    model: BaseModel,
    test_dataloader: DataLoader,
    evaluator: Type[SyntheticDataEvaluator],
    host: str = "localhost",
    send_port: int = 5000,
    receive_port: int = 5001,
    num_clients: int = 3,
    num_rounds: int = NUM_ROUNDS,
    base_dir: str = "",
    weights_path: str = None,
) -> None:
    """
    Launches the Federated Learning server that coordinates communication, aggregation, and evaluation.

    Args:
        model (BaseModel): The GAN model instance with generator/discriminator.
        test_dataloader (BaseDatasetReader): Ground truth test dataloader for evaluation.
        evaluator (Type[SyntheticDataEvaluator]): Evaluator class for computing metrics.
        host (str): Server host address.
        send_port (int): Port for sending weights to clients.
        receive_port (int): Port for receiving weights from clients.
        num_clients (int): Number of participating clients.
        num_rounds (int): Number of communication rounds.
        base_dir (str): Directory to save weights/samples.
        weights_path (str): Optional path to pretrained model weights.

    Returns:
        None
    """
    print("....................................................")

    if weights_path:
        model.load_weights_from_path(weights_path)

    # Initial weights distribution
    """
    send_weights_to_clients(
        host, send_port, model.generator_model, model.discriminator_model, num_clients
    )
    """

    for j in range(num_rounds):
        print(f"\n STARTING ROUND {j}")

        # Step 1: Receive client model updates
        received_weights = receive_weights_from_clients(host, receive_port, num_clients)

        if not received_weights:
            print(f" Error: No weights received in round {j}!")
            continue

        # Step 2: Aggregate weights
        discriminator_weights, generator_weights = fedavg(received_weights)

        # Step 3: Update global model and distribute it
        print(f"Aggregating weights for run_{j}")
        model.update_models_weights(discriminator_weights, generator_weights)

        save_dir = f"{base_dir}/aggregated_weights_run_{j}"
        model.save_models_to_path(save_dir)

        send_weights_to_clients(
            host,
            send_port,
            model.generator_model,
            model.discriminator_model,
            num_clients,
        )
        """
        eval_server = evaluator(test_dataloader, synthetic_dataloader)
        print(eval_server.evaluate())
        del eval_server
        """

