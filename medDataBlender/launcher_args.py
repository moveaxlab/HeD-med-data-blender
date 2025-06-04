def parse_class_percentages(value):
    # Dividere l'input da spazi
    items = value.split(",")
    result = {}
    for item in items:
        # Ogni item sarà in formato 'classe:percentuale%'
        cls, pct = item.split(":")
        result[cls.strip()] = (
            float(pct.strip("%")) / 100.0
        )  # Converte la percentuale in un valore tra 0 e 1
    return result


def setup_common_args(parser):
    parser.add_argument(
        "--data_type",
        type=str,
        required=True,
        choices=["mri", "ecg"],
        help="The type of data. Options are 'mri' or 'ecg'.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=False,
        default=None,
        help="Path to the dataset. If not provided, generation-only mode is assumed.",
    )
    parser.add_argument(
        "--is_s3",
        type=bool,
        default=False,
        help="Set this flag if the dataset is stored in an S3 bucket (zip file). Default is False.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=10, help="Batch size to use during training."
    )
    parser.add_argument(
        "--weights_path",
        type=str,
        default=None,
        help="Path to load pre-trained weights.",
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default=None,
        help="Base directory for saving model and other outputs.",
    )
    parser.add_argument(
        "--out_dir_samples",
        type=str,
        default=None,
        help="Directory where generated samples will be saved.",
    )
    parser.add_argument(
        "--num_samples", type=int, default=100, help="Number of samples to generate."
    )
    parser.add_argument(
        "--only_generation",
        type=bool,
        default=False,
        help="Flag to indicate whether to run only generation (no training).",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=1,
        help="Number of channels in the input data (e.g., 1 for grayscale, 3 for RGB).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of epochs for training the model.",
    )
    parser.add_argument(
        "--num_samples_per_classes",
        type=parse_class_percentages,
        default=None,
        help="Specify the classes and their sample percentages, in the format 'class1:percentage%, class2:percentage%'.",
    )
    parser.add_argument(
        "--s3_endpoint_url",
        type=str,
        default="http://localhost:4566",
        help="Endpoint URL for the S3 service (e.g., LocalStack endpoint).",
    )
    parser.add_argument(
        "--s3_region_name",
        type=str,
        default="us-east-1",
        help="Region name for the S3 bucket.",
    )
    parser.add_argument(
        "--s3_access_key_id",
        type=str,
        default="test",
        help="Access key ID for S3.",
    )
    parser.add_argument(
        "--s3_secret_access_key",
        type=str,
        default="test",
        help="Secret access key for S3.",
    )


def setup_mri_args(parser):
    parser.add_argument(
        "--image_shape",
        type=tuple,
        default=(256, 256),
        help="Shape of the MRI images (height, width).",
    )
    parser.add_argument(
        "--eta", type=float, default=0.0002, help="Learning rate for the model."
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=6e-9,
        help="Weight decay for the optimizer.",
    )
    parser.add_argument(
        "--checkpoint_epoch",
        type=int,
        default=1,
        help="Epoch interval for saving checkpoints.",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=["Non_Demented", "Very_mild_Dementia"],
        help="List of classes in the dataset.",
    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=256,
        help="Dimensionality of the latent space in MRI",
    )


def setup_ecg_args(parser):
    parser.add_argument(
        "--seq_len", type=int, default=187, help="Length of the ECG sequence."
    )
    parser.add_argument(
        "--data_embed_dim",
        type=int,
        default=10,
        help="Dimension of the data embedding.",
    )
    parser.add_argument(
        "--label_embed_dim",
        type=int,
        default=10,
        help="Dimension of the label embedding.",
    )
    parser.add_argument(
        "--depth", type=int, default=3, help="Depth of the neural network."
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=5,
        help="Number of attention heads for the model.",
    )
    parser.add_argument(
        "--forward_drop_rate",
        type=float,
        default=0.5,
        help="Dropout rate for the forward pass.",
    )
    parser.add_argument(
        "--attn_drop_rate",
        type=float,
        default=0.5,
        help="Dropout rate for the attention mechanism.",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=[
            "Non-Ectopic Beats",
            "Superventrical Ectopic",
            "Ventricular Beats",
            "Unknown",
            "Fusion Beats",
        ],
        help="List of classes in the dataset.",
    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=100,
        help="Dimensionality of the latent space in ECG",
    )


def setup_server_args(parser):
    parser.add_argument(
        "--host", type=str, default="localhost", help="Host address for the server."
    )
    parser.add_argument(
        "--send_port",
        type=int,
        default=5000,
        help="Port used by the server to send data.",
    )
    parser.add_argument(
        "--receive_port",
        type=int,
        default=5001,
        help="Port used by the server to receive data.",
    )
    parser.add_argument(
        "--num_clients",
        type=int,
        default=3,
        help="Number of clients to connect to the server.",
    )
    parser.add_argument(
        "--num_rounds",
        type=int,
        default=5,
        help="Number of rounds for federated learning.",
    )


def setup_client_args(parser):
    parser.add_argument(
        "--client_id",
        type=int,
        required=True,
        help="Client ID for this client instance.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host address to connect to the server.",
    )
    parser.add_argument(
        "--send_port",
        type=int,
        default=5000,
        help="Port used by the client to send data.",
    )
    parser.add_argument(
        "--receive_port",
        type=int,
        default=5001,
        help="Port used by the client to receive data.",
    )
