import argparse
import time
import json
from datetime import datetime

from pyspark.sql import SparkSession
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import col


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Distributed ML training job for cloud comparison"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="fashion-mnist",
        choices=["fashion-mnist", "cifar10"],
        help="Dataset to use for training",
    )
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=128, help="Batch size for training"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Cloud storage path to training data (S3 or GCS)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Cloud storage path for output results",
    )

    return parser.parse_args()


def load_fashion_mnist(spark, data_path):
    """
    Load Fashion-MNIST dataset from cloud storage.

    Args:
        spark: SparkSession
        data_path: Cloud storage path to dataset

    Returns:
        Training and test DataFrames
    """
    print(f"Loading Fashion-MNIST dataset from {data_path}")

    # Load training data
    train_df = spark.read.parquet(f"{data_path}/train.parquet")
    test_df = spark.read.parquet(f"{data_path}/test.parquet")

    print(f"Training samples: {train_df.count()}")
    print(f"Test samples: {test_df.count()}")

    return train_df, test_df


def prepare_features(df, num_features=784):
    """
    Prepare features for ML training.

    Args:
        df: Input DataFrame with pixel columns
        num_features: Number of feature columns (784 for Fashion-MNIST)

    Returns:
        DataFrame with assembled feature vector
    """
    # Assemble pixel columns into a feature vector
    feature_cols = [f"pixel{i}" for i in range(num_features)]

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

    return assembler.transform(df)


def train_model(train_df, test_df, epochs=5):
    """
    Train a neural network classifier.

    Args:
        train_df: Training DataFrame with features and label
        test_df: Test DataFrame
        epochs: Number of training epochs (simulated via iterations)

    Returns:
        Trained model and metrics
    """
    print("Training neural network classifier...")

    # Define network architecture
    # Fashion-MNIST: 784 input features -> hidden layers -> 10 output classes
    layers = [784, 256, 128, 10]

    # Create classifier
    classifier = MultilayerPerceptronClassifier(
        featuresCol="features",
        labelCol="label",
        layers=layers,
        maxIter=epochs * 10,  # Approximate epochs
        blockSize=128,
        seed=42,
    )

    # Train model
    start_time = time.time()
    model = classifier.fit(train_df)
    training_time = time.time() - start_time

    print(f"Training completed in {training_time:.2f} seconds")

    # Evaluate on test set
    predictions = model.transform(test_df)

    evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy"
    )

    accuracy = evaluator.evaluate(predictions)

    # Calculate additional metrics
    f1_evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="f1"
    )
    f1_score = f1_evaluator.evaluate(predictions)

    metrics = {
        "training_time_seconds": training_time,
        "accuracy": accuracy,
        "f1_score": f1_score,
        "epochs": epochs,
        "architecture": layers,
    }

    print(f"Model Accuracy: {accuracy:.4f}")
    print(f"Model F1 Score: {f1_score:.4f}")

    return model, metrics


def save_results(spark, output_path, metrics, job_metadata):
    """
    Save training results and metrics to cloud storage.

    Args:
        spark: SparkSession
        output_path: Cloud storage path for output
        metrics: Training metrics dictionary
        job_metadata: Job metadata (start time, end time, etc.)
    """
    print(f"Saving results to {output_path}")

    # Combine metrics and metadata
    results = {**metrics, **job_metadata}

    # Convert to JSON string
    results_json = json.dumps(results, indent=2, default=str)

    # Create a DataFrame with results and save
    results_df = spark.createDataFrame([(results_json,)], ["results"])
    results_df.coalesce(1).write.mode("overwrite").text(f"{output_path}/metrics")

    print("Results saved successfully")


def main():
    """Main execution function."""
    args = parse_args()

    print("=" * 60)
    print("Distributed ML Training Job")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Data path: {args.data_path}")
    print(f"Output path: {args.output_path}")
    print("=" * 60)

    # Record job start time
    job_start_time = time.time()
    job_start_timestamp = datetime.utcnow().isoformat()

    # Create Spark session
    spark = SparkSession.builder.appName("ML-Training-Comparison").getOrCreate()

    try:
        # Load dataset
        print("\n[1/4] Loading dataset...")
        if args.dataset == "fashion-mnist":
            train_df, test_df = load_fashion_mnist(spark, args.data_path)
        else:
            raise NotImplementedError(f"Dataset {args.dataset} not yet implemented")

        # Prepare features
        print("\n[2/4] Preparing features...")
        train_prepared = prepare_features(train_df)
        test_prepared = prepare_features(test_df)

        # Cache for better performance
        train_prepared.cache()
        test_prepared.cache()

        # Train model
        print("\n[3/4] Training model...")
        model, metrics = train_model(train_prepared, test_prepared, epochs=args.epochs)

        # Save results
        print("\n[4/4] Saving results...")
        job_end_time = time.time()
        job_end_timestamp = datetime.utcnow().isoformat()

        job_metadata = {
            "job_start_time": job_start_timestamp,
            "job_end_time": job_end_timestamp,
            "total_job_time_seconds": job_end_time - job_start_time,
            "dataset": args.dataset,
            "batch_size": args.batch_size,
        }

        save_results(spark, args.output_path, metrics, job_metadata)

        print("\n" + "=" * 60)
        print("Job completed successfully!")
        print(f"Total time: {job_end_time - job_start_time:.2f} seconds")
        print("=" * 60)

    except Exception as e:
        print(f"\nERROR: Job failed with exception: {str(e)}")
        import traceback

        traceback.print_exc()
        raise

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
