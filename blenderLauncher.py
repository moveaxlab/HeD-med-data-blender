import argparse
import sys
import torch

from medDataBlender.fedLearning import train_over_runs, server
from medDataBlender.models import Acgan, TTSGan
from medDataBlender.models.dataLoaders import OasisMRI_ReadDataset, Mitbih_ReadDataset
from medDataBlender.models.evaluator import ImageEvaluator, TimeSeriesEvaluator
from medDataBlender import (
    setup_ecg_args,
    setup_mri_args,
    setup_client_args,
    setup_common_args,
    setup_server_args,
)


def load_weights_or_fail(model, weights_path, only_generation):
    if weights_path:
        model.load_weights_from_path(weights_path)
        print(f"Weights loaded from {weights_path}")
    elif only_generation:
        print("Error: --only_generation is True but no --weights_path was provided.")
        sys.exit(1)
    else:
        print("No weights provided. Training will start from scratch.")


def define_model_and_loader(args):
    data_type = args.data_type

    s3_region_name = ""
    s3_access_key_id = ""
    s3_secret_access_key = ""
    s3_endpoint_url = ""

    if args.is_s3:
        s3_region_name = args.s3_region_name
        s3_access_key_id = args.s3_access_key_id
        s3_secret_access_key = args.s3_secret_access_key
        s3_endpoint_url = args.s3_endpoint_url

    if not args.data_path:
        print("No data_path provided, returning models only with no data loading.")
        dataloader = None
        tot_img = None
        label_percentage = args.num_samples_per_classes
    else:
        if data_type == "mri":
            print("Starting MRI pipeline...")
            dataset_reader = OasisMRI_ReadDataset(
                dataset_path=args.data_path,
                labels=args.classes,
                data_shape=args.image_shape,
                batch_size=args.batch_size,
                is_s3=args.is_s3,
                s3_region_name=s3_region_name,
                s3_access_key_id=s3_access_key_id,
                s3_secret_access_key=s3_secret_access_key,
                s3_endpoint_url=s3_endpoint_url,
            )
        elif data_type == "ecg":
            print("Starting ECG pipeline...")

            dataset_reader = Mitbih_ReadDataset(
                dataset_path=args.data_path,
                labels=args.classes,
                data_shape=(1, args.channels, args.seq_len),
                batch_size=args.batch_size,
                is_s3=args.is_s3,
                s3_region_name=s3_region_name,
                s3_access_key_id=s3_access_key_id,
                s3_secret_access_key=s3_secret_access_key,
                s3_endpoint_url=s3_endpoint_url,
            )
        else:
            raise ValueError(f"Unsupported data_type '{data_type}'")

        dataloader, label_counter = dataset_reader.load_data()
        tot_img, label_percentage = dataset_reader.compute_class_distribution()
        dataset_reader.cleanup_temp_dir()

    # Creazione del modello
    if data_type == "mri":
        model = Acgan(
            eta=args.eta,
            epochs=args.epochs,
            weight_decay=args.weight_decay,
            latent_space=args.latent_dim,
            image_shape=[args.channels, *args.image_shape],
            kernel_size=5,
            n_classes=len(label_percentage) if label_percentage else 0,
            checkpoint_epoch=args.checkpoint_epoch,
            base_dir=args.base_dir,
            use_s3=args.is_s3,
        )
    else:  # ECG
        model = TTSGan(
            seq_len=args.seq_len,
            channels=args.channels,
            n_classes=len(label_percentage) if label_percentage else 0,
            latent_dim=args.latent_dim,
            data_embed_dim=args.data_embed_dim,
            label_embed_dim=args.label_embed_dim,
            depth=args.depth,
            num_heads=args.num_heads,
            forward_drop_rate=args.forward_drop_rate,
            attn_drop_rate=args.attn_drop_rate,
            base_dir=args.base_dir,
            use_s3=args.is_s3,
        )

    model.build()

    return model, dataloader, tot_img, label_percentage


def main():
    parser = argparse.ArgumentParser(description="medDataBlender framework")
    subparsers = parser.add_subparsers(dest="run_mode", required=True)

    # Crea subparser e salvali in una lista per iterarli dopo
    all_subparsers = []

    # Centralized
    centralized_parser = subparsers.add_parser("centralized")
    all_subparsers.append(centralized_parser)

    # Server
    server_parser = subparsers.add_parser("server")
    all_subparsers.append(server_parser)

    # Client
    client_parser = subparsers.add_parser("client")
    all_subparsers.append(client_parser)

    # === STEP 2: Parse parziale per sapere il tipo di dato ===
    for sp in all_subparsers:
        setup_common_args(sp)
    partial_args, _ = parser.parse_known_args()
    selected_data_type = partial_args.data_type

    # === STEP 3: Aggiunta di tutti gli argomenti comuni e specifici ai subparser ===

    def setup_data_args(subparser):
        if selected_data_type == "mri":
            setup_mri_args(subparser)
        else:
            setup_ecg_args(subparser)

    # Applica argomenti a ogni subparser
    for sp in all_subparsers:
        setup_data_args(sp)

    # Aggiungi anche argomenti specifici per subparser
    setup_server_args(server_parser)
    setup_client_args(client_parser)
    # centralized non ha altri argomenti oltre quelli comuni + data_type

    # === STEP 4: Parse finale ===
    args = parser.parse_args()

    if args.only_generation and not args.data_path:
        if not args.num_samples_per_classes:
            print(
                "Error: --only_generation is True and --data_path is not provided, but --num_samples_per_classes is "
                "missing."
            )
            sys.exit(1)

        # Verifica che num_samples_per_classes sia un dizionario
        if not isinstance(args.num_samples_per_classes, dict):
            print(
                "Error: --num_samples_per_classes must be a dictionary with class names as keys and percentages as values."
            )
            sys.exit(1)

        # Verifica che tutte le classi siano presenti nel dizionario
        missing_classes = [
            cls for cls in args.classes if cls not in args.num_samples_per_classes
        ]
        if missing_classes:
            print(
                f"Error: The following classes are missing in --num_samples_per_classes: {', '.join(missing_classes)}."
            )
            sys.exit(1)

        # Verifica che le percentuali siano nel formato corretto (numeri tra 0 e 100)
        invalid_percentages = [
            (cls, perc)
            for cls, perc in args.num_samples_per_classes.items()
            if not (0 <= perc <= 100)
        ]
        if invalid_percentages:
            print(
                f"Error: Invalid percentages found for the following classes: {', '.join([cls for cls, _ in invalid_percentages])}. Percentages must be between 0 and 100."
            )
            sys.exit(1)

        # Verifica che la somma delle percentuali sia uguale a 100
        total_percentage = sum(args.num_samples_per_classes.values())
        if total_percentage != 1:
            print(
                f"Error: The total percentage of samples must equal 100%. Current total: {total_percentage}%."
            )
            sys.exit(1)
    elif not args.only_generation and not args.data_path:
        print(
            "Error: --only_generation is False and --data_path is not provided, you cannot train the model without data"
        )
        sys.exit(1)

    model, dataloader, tot_img, label_percentage = define_model_and_loader(args)

    if dataloader:
        data_shape = next(iter(dataloader))[0].shape[1:]
        model.get_model_memory_usage(
            (1, *data_shape), n_samples=tot_img, batch_size=args.batch_size
        )

    if args.run_mode == "centralized":
        load_weights_or_fail(model, args.weights_path, args.only_generation)
        if not args.only_generation:
            model.train_algorithm(dataloader)
        if args.out_dir_samples is not None:
            model.generate_and_save_samples_by_class(
                num_samples=args.num_samples,
                save_dir=args.out_dir_samples,
                label_percentage=label_percentage,
                batch_size=args.batch_size,
            )

    elif args.run_mode == "server":
        print("Starting server...")
        evaluator_cls = (
            ImageEvaluator if args.data_type == "mri" else TimeSeriesEvaluator
        )
        server(
            model=model,
            test_dataloader=dataloader,
            evaluator=evaluator_cls,
            host=args.host,
            send_port=args.send_port,
            receive_port=args.receive_port,
            num_clients=args.num_clients,
            num_rounds=args.num_rounds,
            base_dir=args.base_dir,
            weights_path=args.weights_path,
        )

    elif args.run_mode == "client":
        print(f"Client {args.client_id} started.")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_over_runs(
            model=model,
            train_loader=dataloader,
            device=device,
            client_id=args.client_id,
            host=args.host,
            send_port=args.send_port,
            receive_port=args.receive_port,
        )


if __name__ == "__main__":
    main()
