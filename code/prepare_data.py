import os
import argparse
import gzip
import numpy as np
import pandas as pd
import requests
import boto3
from google.cloud import storage
from config import AWS_CONFIG, GCP_CONFIG
from utils.logger_config import setup_logger

logger = setup_logger("data_preparation")


def download_file(url: str, filepath: str) -> None:
    """Download a file from URL to local path."""
    logger.info(f"Downloading {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(filepath, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    logger.info(f"Downloaded to {filepath}")


def load_mnist_images(filepath: str) -> np.ndarray:
    """Load MNIST image file (ubyte format)."""
    with gzip.open(filepath, "rb") as f:
        # Read header (first 16 bytes)
        num_images = int.from_bytes(f.read(4), "big")
        rows = int.from_bytes(f.read(4), "big")
        cols = int.from_bytes(f.read(4), "big")

        # Read image data
        data = np.frombuffer(f.read(), dtype=np.uint8)
        data = data.reshape(num_images, rows, cols)

    return data


def load_mnist_labels(filepath: str) -> np.ndarray:
    """Load MNIST label file (ubyte format)."""
    with gzip.open(filepath, "rb") as f:
        # Read label data
        labels = np.frombuffer(f.read(), dtype=np.uint8)

    return labels


def download_fashion_mnist(output_dir: str) -> tuple:
    """
    Download Fashion-MNIST dataset directly from source.

    Args:
        output_dir: Directory to save the data

    Returns:
        Tuple of (train_data, train_labels, test_data, test_labels)
    """
    logger.info("Downloading Fashion-MNIST dataset...")

    # Fashion-MNIST URLs
    base_url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
    files = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz",
    }

    # Create cache directory
    cache_dir = os.path.join(output_dir, ".fashion-mnist-cache")
    os.makedirs(cache_dir, exist_ok=True)

    # Download files if not already cached
    filepaths = {}
    for key, filename in files.items():
        filepath = os.path.join(cache_dir, filename)
        if not os.path.exists(filepath):
            url = base_url + filename
            download_file(url, filepath)
        else:
            logger.info(f"Using cached {filename}")
        filepaths[key] = filepath

    # Load data
    logger.info("Loading training images...")
    x_train = load_mnist_images(filepaths["train_images"])
    logger.info("Loading training labels...")
    y_train = load_mnist_labels(filepaths["train_labels"])
    logger.info("Loading test images...")
    x_test = load_mnist_images(filepaths["test_images"])
    logger.info("Loading test labels...")
    y_test = load_mnist_labels(filepaths["test_labels"])

    logger.info(f"Training set: {x_train.shape[0]} samples")
    logger.info(f"Test set: {x_test.shape[0]} samples")
    logger.info(f"Image shape: {x_train.shape[1:]} (28x28)")

    # Normalize pixel values to 0-1 range
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    return x_train, y_train, x_test, y_test


def convert_to_parquet(
    images: np.ndarray, labels: np.ndarray, output_path: str, dataset_type: str
) -> None:
    """
    Convert image data to Parquet format for Spark processing.

    Args:
        images: Image array (N, 28, 28)
        labels: Label array (N,)
        output_path: Path to save the parquet file
        dataset_type: 'train' or 'test'
    """
    logger.info(f"Converting {dataset_type} data to Parquet format...")

    # Flatten images from (N, 28, 28) to (N, 784)
    num_samples = images.shape[0]
    images_flat = images.reshape(num_samples, -1)

    # Create DataFrame
    # Column names: pixel0, pixel1, ..., pixel783, label
    pixel_columns = [f"pixel{i}" for i in range(784)]
    data_dict = {col: images_flat[:, i] for i, col in enumerate(pixel_columns)}
    data_dict["label"] = labels

    df = pd.DataFrame(data_dict)

    # Save as Parquet
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False, engine="pyarrow")

    logger.info(f"Saved {dataset_type} data to {output_path}")
    logger.info(f"  Shape: {df.shape}")
    logger.info(f"  Size: {os.path.getsize(output_path) / (1024 * 1024):.2f} MB")


def upload_to_s3(local_file: str, s3_key: str, bucket: str = None) -> str:
    """
    Upload a file to AWS S3.

    Args:
        local_file: Local file path
        s3_key: S3 object key
        bucket: S3 bucket name (defaults to config)

    Returns:
        S3 URI of uploaded file
    """
    bucket = bucket or AWS_CONFIG["s3_bucket"]
    logger.info(f"Uploading {local_file} to s3://{bucket}/{s3_key}")

    s3_client = boto3.client("s3", region_name=AWS_CONFIG["region"])

    try:
        s3_client.upload_file(local_file, bucket, s3_key)
        s3_uri = f"s3://{bucket}/{s3_key}"
        logger.info(f"Successfully uploaded to {s3_uri}")
        return s3_uri
    except Exception as e:
        logger.error(f"Failed to upload to S3: {str(e)}")
        raise


def upload_to_gcs(local_file: str, gcs_path: str, bucket: str = None) -> str:
    """
    Upload a file to Google Cloud Storage.

    Args:
        local_file: Local file path
        gcs_path: GCS object path
        bucket: GCS bucket name (defaults to config)

    Returns:
        GCS URI of uploaded file
    """
    bucket_name = bucket or GCP_CONFIG["gcs_bucket"]
    logger.info(f"Uploading {local_file} to gs://{bucket_name}/{gcs_path}")

    storage_client = storage.Client(project=GCP_CONFIG["project_id"])
    bucket_obj = storage_client.bucket(bucket_name)
    blob = bucket_obj.blob(gcs_path)

    try:
        blob.upload_from_filename(local_file)
        gcs_uri = f"gs://{bucket_name}/{gcs_path}"
        logger.info(f"Successfully uploaded to {gcs_uri}")
        return gcs_uri
    except Exception as e:
        logger.error(f"Failed to upload to GCS: {str(e)}")
        raise


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Prepare and upload dataset for cloud experiments"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="fashion-mnist",
        choices=["fashion-mnist"],
        help="Dataset to prepare",
    )
    parser.add_argument(
        "--output-dir", type=str, default="../data", help="Local output directory"
    )
    parser.add_argument("--upload-aws", action="store_true", help="Upload to AWS S3")
    parser.add_argument(
        "--upload-gcp", action="store_true", help="Upload to GCP Cloud Storage"
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Data Preparation Script")
    logger.info("=" * 60)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("=" * 60)

    # Create output directory
    data_dir = os.path.join(args.output_dir, args.dataset)
    os.makedirs(data_dir, exist_ok=True)

    # Download dataset
    if args.dataset == "fashion-mnist":
        x_train, y_train, x_test, y_test = download_fashion_mnist(data_dir)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    # Convert to Parquet
    train_parquet = os.path.join(data_dir, "train.parquet")
    test_parquet = os.path.join(data_dir, "test.parquet")

    convert_to_parquet(x_train, y_train, train_parquet, "train")
    convert_to_parquet(x_test, y_test, test_parquet, "test")

    # Upload to cloud storage
    uploaded_files = {
        "train": {"local": train_parquet},
        "test": {"local": test_parquet},
    }

    if args.upload_aws:
        logger.info("\nUploading to AWS S3...")
        try:
            for dataset_type in ["train", "test"]:
                s3_key = f"data/{args.dataset}/{dataset_type}.parquet"
                s3_uri = upload_to_s3(uploaded_files[dataset_type]["local"], s3_key)
                uploaded_files[dataset_type]["s3"] = s3_uri
        except Exception as e:
            logger.error(f"AWS upload failed: {str(e)}")

    if args.upload_gcp:
        logger.info("\nUploading to GCP Cloud Storage...")
        try:
            for dataset_type in ["train", "test"]:
                gcs_path = f"data/{args.dataset}/{dataset_type}.parquet"
                gcs_uri = upload_to_gcs(uploaded_files[dataset_type]["local"], gcs_path)
                uploaded_files[dataset_type]["gcs"] = gcs_uri
        except Exception as e:
            logger.error(f"GCP upload failed: {str(e)}")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Data Preparation Complete!")
    logger.info("=" * 60)
    logger.info("\nLocal files:")
    for dataset_type, paths in uploaded_files.items():
        logger.info(f"  {dataset_type}: {paths['local']}")

    if args.upload_aws:
        logger.info("\nAWS S3 files:")
        for dataset_type, paths in uploaded_files.items():
            if "s3" in paths:
                logger.info(f"  {dataset_type}: {paths['s3']}")

    if args.upload_gcp:
        logger.info("\nGCP Cloud Storage files:")
        for dataset_type, paths in uploaded_files.items():
            if "gcs" in paths:
                logger.info(f"  {dataset_type}: {paths['gcs']}")

    logger.info("\n" + "=" * 60)


if __name__ == "__main__":
    main()
