def add_report_args(parser):
    parser.add_argument(
        "--real_dir", type=str, required=True, help="Path to real training data"
    )
    parser.add_argument(
        "--fake_dir", type=str, required=True, help="Path to fake training data"
    )
    parser.add_argument(
        "--real_test_dir", type=str, required=True, help="Path to real test data"
    )
    parser.add_argument(
        "--fake_test_dir", type=str, required=True, help="Path to fake test data"
    )
    parser.add_argument(
        "--model_path_fake",
        type=str,
        required=False,
        help="Path to the trained model on fake data",
    )
    parser.add_argument(
        "--model_path_mia", type=str, required=False, help="Path to MIA attack model"
    )
    parser.add_argument(
        "--base_dir", type=str, required=True, help="Base directory for output and logs"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of classes")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")


def add_db_args(parser):
    parser.add_argument(
        "--db_host", type=str, default=None, help="Database host address"
    )
    parser.add_argument(
        "--db_port", type=int, default=3306, help="Database port (default: 3306)"
    )
    parser.add_argument(
        "--db_name", type=str, default=None, help="Name of the database"
    )
    parser.add_argument("--db_user", type=str, default=None, help="Database username")
    parser.add_argument(
        "--db_password", type=str, default=None, help="Database password"
    )
    parser.add_argument(
        "--db_table", type=str, default=None, help="Database table to store results"
    )


def add_server_args(parser):
    parser.add_argument("--server_model_path_fake", type=str, required=True)
    parser.add_argument("--server_model_path_mia", type=str, required=True)
    parser.add_argument("--server_fake_data", type=str, required=True)
    parser.add_argument("--server_num_clients", type=int, required=True)
    parser.add_argument("--server_port", type=int, default=5000)


def add_client_args(parser):
    parser.add_argument("--client_real_test_dir", type=str, required=True)
    parser.add_argument("--client_save_dir", type=str, required=True)
    parser.add_argument("--client_id", type=int, required=True)
    parser.add_argument("--client_host", type=str, default="localhost")
    parser.add_argument("--client_port", type=int, default=5000)
