"""
Configuration file for AWS EMR vs GCP Dataproc comparison experiments.
Adjust these settings based on your requirements and available resources.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# =============================================================================
# AWS EMR Configuration
# =============================================================================
AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET", "your-bucket-name")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
AWS_CONFIG = {
    "region": AWS_REGION,
    "cluster_name": "emr-ml-comparison-cluster",
    "release_label": "emr-6.15.0",
    "log_uri": f"s3://{AWS_S3_BUCKET}/logs/",
    "ec2_key_name": os.getenv("AWS_EC2_KEY_NAME") or None,
    # Instance configuration
    "master_instance_type": os.getenv("AWS_MASTER_INSTANCE_TYPE", "r8g.xlarge"),
    "core_instance_type": os.getenv("AWS_CORE_INSTANCE_TYPE", "r8g.xlarge"),
    "core_instance_count": int(os.getenv("AWS_CORE_INSTANCE_COUNT", "2")),
    # Data storage
    "s3_bucket": AWS_S3_BUCKET,
    "s3_data_path": f"s3://{AWS_S3_BUCKET}/data/",
    "s3_scripts_path": f"s3://{AWS_S3_BUCKET}/scripts/",
    "s3_results_path": f"s3://{AWS_S3_BUCKET}/results/",
}
# AWS Pricing
AWS_PRICING = {
    "m4.large": 0.10,
    "emr_fee_per_instance": 0.03,  # Additional EMR fee per EC2 instance hour
    "s3_storage": 0.023,  # USD per GB-month (first 50 TB)
    "s3_requests_put": 0.005,  # USD per 1000 PUT requests
}

# =============================================================================
# GCP Dataproc Configuration
# =============================================================================
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "your-project-id")
GCP_GCS_BUCKET = os.getenv("GCP_GCS_BUCKET", "your-bucket-name")
GCP_REGION = os.getenv("GCP_REGION", "us-east1")

GCP_CONFIG = {
    "project_id": GCP_PROJECT_ID,
    "region": GCP_REGION,
    "cluster_name": "dataproc-ml-comparison-cluster",
    "image_version": "2.1-debian11",  # Includes Spark 3.3.2
    # Instance configuration
    "master_machine_type": os.getenv("GCP_MASTER_MACHINE_TYPE", "n1-standard-4"),
    "worker_machine_type": os.getenv("GCP_WORKER_MACHINE_TYPE", "n1-standard-4"),
    "worker_count": int(os.getenv("GCP_WORKER_COUNT", "2")),
    # Data storage
    "gcs_bucket": GCP_GCS_BUCKET,
    "gcs_data_path": f"gs://{GCP_GCS_BUCKET}/data/",
    "gcs_scripts_path": f"gs://{GCP_GCS_BUCKET}/scripts/",
    "gcs_results_path": f"gs://{GCP_GCS_BUCKET}/results/",
}

# GCP Pricing
GCP_PRICING = {
    "n1-standard-2": 0.095,  # USD per hour
    "dataproc_fee_per_vcpu": 0.010,  # USD per vCPU hour
    "gcs_storage": 0.020,  # USD per GB-month (standard storage)
    "gcs_operations_class_a": 0.005,  # USD per 10K operations
}

# =============================================================================
# Machine Learning Configuration
# =============================================================================
ML_CONFIG = {
    "dataset": os.getenv("ML_DATASET", "fashion-mnist"),
    "epochs": int(os.getenv("ML_EPOCHS", "5")),
    "batch_size": int(os.getenv("ML_BATCH_SIZE", "128")),
    "model_type": "cnn",  # Convolutional Neural Network
    "test_split": 0.2,
}

# =============================================================================
# Benchmark Configuration
# =============================================================================
BENCHMARK_CONFIG = {
    # Dataset paths
    # Note: large dataset mode will sample 50% of rows from the file for faster processing
    "transactions_sample_path_aws": f"s3://{AWS_S3_BUCKET}/data/transactions/transactions_sample_1m.csv",
    "transactions_large_path_aws": f"s3://{AWS_S3_BUCKET}/data/transactions/transactions_large.csv",
    "transactions_full_path_aws": f"s3://{AWS_S3_BUCKET}/data/transactions/transactions.csv",
    "transactions_sample_path_gcp": f"gs://{GCP_GCS_BUCKET}/data/transactions/transactions_sample_1m.csv",
    "transactions_large_path_gcp": f"gs://{GCP_GCS_BUCKET}/data/transactions/transactions_large.csv",
    "transactions_full_path_gcp": f"gs://{GCP_GCS_BUCKET}/data/transactions/transactions.csv",
    # Benchmark output paths
    "benchmark_output_aws": f"s3://{AWS_S3_BUCKET}/benchmarks/",
    "benchmark_output_gcp": f"gs://{GCP_GCS_BUCKET}/benchmarks/",
    # Storage benchmark settings
    "storage_test_file_max_mb": 500,  # Max file size for upload tests
    # PySpark benchmark settings
    "dataset_modes": ["sample", "large", "full"],  # Available dataset modes
    "default_mode": "sample",  # Default mode for testing
    # Dataset sizes (in GB) for cost calculation
    "dataset_size_sample": 0.08,  # 1M rows ~81MB
    "dataset_size_large": 5.0,  # ~65M rows ~5GB (half of original large dataset)
    "dataset_size_full": 21.0,  # 262M rows ~21GB
}

# =============================================================================
# Experiment Configuration
# =============================================================================
EXPERIMENT_CONFIG = {
    "results_dir": os.path.join(os.path.dirname(__file__), "..", "data", "results"),
    "timeout_minutes": int(os.getenv("TIMEOUT_MINUTES", "60")),
    "retry_attempts": 3,
    "cleanup_on_failure": os.getenv("CLEANUP_ON_COMPLETE", "true").lower() == "true",
}

# =============================================================================
# Helper Functions
# =============================================================================


def validate_config():
    """
    Validate that required configuration values have been set.
    Raises ValueError if any required values are missing.
    """
    errors = []

    # Check AWS config
    if not os.getenv("AWS_S3_BUCKET") or "your-bucket-name" in AWS_CONFIG["s3_bucket"]:
        errors.append("AWS_S3_BUCKET environment variable must be set in .env file")

    if not os.getenv("AWS_ACCESS_KEY_ID"):
        errors.append("AWS_ACCESS_KEY_ID environment variable must be set in .env file")

    if not os.getenv("AWS_SECRET_ACCESS_KEY"):
        errors.append(
            "AWS_SECRET_ACCESS_KEY environment variable must be set in .env file"
        )

    # Check GCP config
    if not os.getenv("GCP_PROJECT_ID") or "your-project-id" in GCP_CONFIG["project_id"]:
        errors.append("GCP_PROJECT_ID environment variable must be set in .env file")

    if (
        not os.getenv("GCP_GCS_BUCKET")
        or "your-bucket-name" in GCP_CONFIG["gcs_bucket"]
    ):
        errors.append("GCP_GCS_BUCKET environment variable must be set in .env file")

    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        errors.append(
            "GOOGLE_APPLICATION_CREDENTIALS environment variable must be set in .env file"
        )

    if errors:
        raise ValueError(
            "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        )
