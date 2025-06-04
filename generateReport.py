import argparse
from medDataBlender.validationReport import (
    ReportBuilder,
    add_report_args,
    add_db_args,
    add_server_args,
    add_client_args,
    MetricsDB,
)
from medDataBlender.validationReport.distributedReport import ServerReport
from medDataBlender.validationReport.distributedReport import ClientReport


# Check if all required database parameters are provided to allow writing to the DB
def should_write_to_db(args):
    return all(
        [
            args.db_host,
            args.db_name,
            args.db_user,
            args.db_password,
            args.db_table,
        ]
    )


# Attempt to write the computed metrics to the database
def write_metrics_to_db(metrics, args):
    try:
        db = MetricsDB(
            dbname=args.db_name,
            user=args.db_user,
            password=args.db_password,
            host=args.db_host,
            port=args.db_port,
        )
        db.insert_entry(metrics, args.data_type)  # Insert metrics into the table
        db.close()
        print("Metrics written to database successfully.")
    except Exception as e:
        print(f"Failed to write metrics to database: {e}")


# Add centralized mode arguments (standard evaluation mode)
def add_centralized_args(parser):
    add_report_args(parser)
    add_db_args(parser)


# Add server mode arguments (for distributed evaluation)
def add_server_mode_args(parser):
    add_server_args(parser)
    add_db_args(parser)


# Add client mode arguments (for distributed evaluation)
def add_client_mode_args(parser):
    add_client_args(parser)


# Add arguments common to all modes (centralized/server/client)
def add_common_args(parser):
    parser.add_argument(
        "--data_type",
        type=str,
        choices=["mri", "ecg"],
        required=True,
        help="Type of data to evaluate (mri or ecg)",
    )


# Main function handling CLI arguments and dispatching based on selected mode
def main():
    parser = argparse.ArgumentParser(description="medDataBlender framework")

    # Define subcommands for different run modes
    subparsers = parser.add_subparsers(dest="run_mode", required=True)

    # Subparsers for each mode
    centralized_parser = subparsers.add_parser("centralized")
    server_parser = subparsers.add_parser("server")
    client_parser = subparsers.add_parser("client")

    # Add mode-specific arguments
    add_report_args(centralized_parser)  # real_dir, fake_dir, model_path, etc.
    add_common_args(centralized_parser)
    add_db_args(centralized_parser)  # database connection arguments

    add_server_args(server_parser)
    add_common_args(server_parser)
    add_db_args(server_parser)  # optional DB for server

    add_client_args(client_parser)
    add_common_args(client_parser)

    # Parse command line arguments
    args = parser.parse_args()

    metrics = None  # Will store metrics if applicable

    # Run centralized evaluation
    if args.run_mode == "centralized":
        report = ReportBuilder(
            base_dir=args.base_dir,
            real_dir=args.real_dir,
            fake_dir=args.fake_dir,
            real_test_dir=args.real_test_dir,
            fake_test_dir=args.fake_test_dir,
            batch_size=args.batch_size,
            model_path_mia=args.model_path_mia,
            model_path_fake=args.model_path_fake,
            num_classes=args.num_classes,
            lr=args.lr,
            num_epochs=args.num_epochs,
            data_type=args.data_type,
        )
        metrics = report.compute_metrics()  # Compute evaluation metrics

    # Run in server mode (collect metrics from clients)
    elif args.run_mode == "server":
        server = ServerReport(
            model_path_fake=args.server_model_path_fake,
            model_path_mia=args.server_model_path_mia,
            fake_data_dir=args.server_fake_data,  # Path to directory with fake data
            num_clients=args.server_num_clients,
            port=args.server_port,
        )
        metrics = server.run()  # Starts server and waits for clients

    # Run in client mode (send evaluation to server)
    elif args.run_mode == "client":
        client = ClientReport(
            data_type=args.data_type,
            real_test_dir=args.client_real_test_dir,
            save_dir=args.client_save_dir,
            port=args.client_port,
            host=args.client_host,
            client_id=args.client_id,
        )
        client.run()  # Client computes and sends metrics
        return  # Clients do not write to DB

    # Write metrics to database if available and DB config is valid
    if metrics:
        if should_write_to_db(args):
            write_metrics_to_db(metrics, args)
        else:
            # Print metrics to console if DB writing is not configured
            print("Computed Metrics:")
            for key, value in metrics.items():
                print(f"{key}: {value}")


# Entry point of the script
if __name__ == "__main__":
    main()
