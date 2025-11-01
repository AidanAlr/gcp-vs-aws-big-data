import argparse
import sys
import os
from pathlib import Path
from datetime import datetime

from aws.emr_cluster import EMRCluster
from aws.emr_runner import EMRRunner
from utils.metrics_collector import MetricsCollector
from utils.cost_calculator import CostCalculator
from utils.logger_config import setup_logger
from config import AWS_CONFIG, EXPERIMENT_CONFIG, BENCHMARK_CONFIG

logger = setup_logger("aws_benchmark", log_file="../data/results/aws_benchmark.log")


def run_storage_benchmark(cleanup: bool = True, experiment_id: str = None) -> dict:
    """
    Run S3 storage performance benchmark.

    This runs locally (no cluster needed) to test S3 performance.
    """
    logger.info("=" * 60)
    logger.info("Starting AWS S3 Storage Benchmark")
    logger.info("=" * 60)

    try:
        # Import storage benchmark
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "benchmarks"))
        from storage_benchmark import StorageBenchmark

        # Use sample data file
        project_root = Path(__file__).parent.parent
        test_file = project_root / "data/transactions/transactions_sample_1m.csv"

        benchmark = StorageBenchmark(
            provider="aws",
            test_file_path=str(test_file),
            max_file_size_mb=BENCHMARK_CONFIG["storage_test_file_max_mb"],
        )

        results = benchmark.run_full_benchmark()

        # Save results
        results_dir = os.path.join(EXPERIMENT_CONFIG["results_dir"], "storage")
        os.makedirs(results_dir, exist_ok=True)  # Ensure directory exists

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_id = experiment_id or timestamp
        results_file = os.path.join(results_dir, f"storage_aws_{exp_id}.json")

        import json

        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Storage benchmark results saved to: {results_file}")
        logger.info("✓ Storage benchmark completed successfully!")

        return results

    except Exception as e:
        logger.error(f"Storage benchmark failed: {str(e)}")
        raise


def run_bigdata_benchmark(
    cleanup: bool = True,
    experiment_id: str = None,
    dataset_mode: str = "sample",
    cluster_id: str = None,
    emr_instance: EMRCluster = None,
) -> dict:
    """
    Run big data operations benchmark on transactions dataset.

    Args:
        cleanup: Whether to cleanup cluster after experiment
        experiment_id: Optional custom experiment ID
        dataset_mode: "sample" or "full"
        cluster_id: Optional existing cluster ID to reuse
        emr_instance: Optional existing EMRCluster instance to reuse
    """
    metrics = MetricsCollector(provider="aws", experiment_id=experiment_id)
    cost_calc = CostCalculator(provider="aws")

    # Track experiment metadata
    metrics.add_metadata("benchmark_type", "bigdata")
    metrics.add_metadata("dataset_mode", dataset_mode)
    metrics.add_metadata(
        "cluster_config",
        {
            "master_instance": AWS_CONFIG["master_instance_type"],
            "core_instance": AWS_CONFIG["core_instance_type"],
            "core_instance_count": AWS_CONFIG["core_instance_count"],
            "region": AWS_CONFIG["region"],
        },
    )

    created_new_cluster = False
    emr = emr_instance

    try:
        logger.info("=" * 60)
        logger.info(f"Starting AWS Big Data Benchmark ({dataset_mode} mode)")
        logger.info("=" * 60)

        metrics.start_timer("total_time")

        # Step 1: Create EMR Cluster (or reuse existing)
        provisioning_time = 0
        if cluster_id is None:
            logger.info("\n[1/6] Creating EMR cluster...")
            emr = EMRCluster(region=AWS_CONFIG["region"])

            metrics.start_timer("cluster_provisioning")
            cluster_id = emr.create_cluster()
            metrics.add_metadata("cluster_id", cluster_id)

            logger.info("Waiting for cluster to be ready...")
            if not emr.wait_for_cluster_ready(
                cluster_id, timeout_minutes=EXPERIMENT_CONFIG["timeout_minutes"]
            ):
                raise RuntimeError("Cluster failed to become ready")

            provisioning_time = metrics.stop_timer("cluster_provisioning")
            created_new_cluster = True
        else:
            logger.info(f"\n[1/6] Reusing existing cluster: {cluster_id}")
            metrics.add_metadata("cluster_id", cluster_id)
            metrics.add_metadata("reused_cluster", True)

        # Step 2: Upload Benchmark Script to S3
        logger.info("\n[2/6] Uploading benchmark script to S3...")
        runner = EMRRunner(cluster_id, region=AWS_CONFIG["region"])

        script_local_path = os.path.join(
            os.path.dirname(__file__), "spark_jobs/bigdata_benchmark_job.py"
        )
        script_s3_uri = runner.upload_script_to_s3(script_local_path)
        metrics.add_metadata("script_s3_uri", script_s3_uri)

        # Step 3: Submit Spark Benchmark Job
        logger.info("\n[3/6] Submitting Spark benchmark job...")

        # Select data path based on mode
        if dataset_mode == "sample":
            data_path = BENCHMARK_CONFIG["transactions_sample_path_aws"]
        elif dataset_mode == "large":
            data_path = BENCHMARK_CONFIG["transactions_large_path_aws"]
        else:
            data_path = BENCHMARK_CONFIG["transactions_full_path_aws"]

        output_path = (
            f"{BENCHMARK_CONFIG['benchmark_output_aws']}{metrics.experiment_id}"
        )

        job_args = [
            "--data-path",
            data_path,
            "--output-path",
            output_path,
            "--dataset-mode",
            dataset_mode,
        ]

        metrics.start_timer("benchmark_execution")
        step_id = runner.submit_spark_job(
            script_s3_uri=script_s3_uri,
            job_name=f"BigData-Benchmark-{metrics.experiment_id}",
            args=job_args,
        )

        metrics.add_metadata("step_id", step_id)

        # Step 4: Monitor Job Completion
        logger.info("\n[4/6] Monitoring benchmark execution...")

        if not runner.wait_for_step_completion(
            step_id, timeout_minutes=EXPERIMENT_CONFIG["timeout_minutes"]
        ):
            raise RuntimeError("Benchmark job failed or timed out")

        benchmark_time = metrics.stop_timer("benchmark_execution")

        # Step 5: Calculate Costs
        logger.info("\n[5/6] Calculating costs...")

        cluster_runtime = provisioning_time + benchmark_time
        # Get data size from config based on mode
        if dataset_mode == "sample":
            data_size_gb = BENCHMARK_CONFIG["dataset_size_sample"]
        elif dataset_mode == "large":
            data_size_gb = BENCHMARK_CONFIG["dataset_size_large"]
        else:
            data_size_gb = BENCHMARK_CONFIG["dataset_size_full"]

        costs = cost_calc.calculate_total_experiment_cost(
            cluster_runtime_seconds=cluster_runtime, data_size_gb=data_size_gb
        )

        for cost_component, amount in costs.items():
            if cost_component != "total":
                metrics.record_cost(cost_component, amount)

        cost_calc.print_cost_estimate(cluster_runtime, data_size_gb=data_size_gb)

        # Step 6: Cleanup (only if we created a new cluster)
        if cleanup and created_new_cluster:
            logger.info("\n[6/6] Cleaning up resources...")
            emr.terminate_cluster(cluster_id)
            logger.info("Cluster termination initiated")
        else:
            if created_new_cluster:
                logger.info("\n[6/6] Skipping cleanup")
                logger.warning(f"Remember to manually terminate cluster: {cluster_id}")
            else:
                logger.info("\n[6/6] Reusing cluster, skipping cleanup")

        total_time = metrics.stop_timer("total_time")

        # Save results
        logger.info("\n" + "=" * 60)
        logger.info("AWS Big Data Benchmark Completed Successfully!")
        logger.info("=" * 60)

        metrics.print_summary()

        pyspark_results_dir = os.path.join(EXPERIMENT_CONFIG["results_dir"], "pyspark")
        metrics_file = metrics.save_to_file(pyspark_results_dir)
        logger.info(f"Metrics saved to: {metrics_file}")

        return metrics.get_metrics()

    except Exception as e:
        logger.error(f"\nBenchmark failed: {str(e)}")
        metrics.record_error(str(e), error_type=type(e).__name__)

        if cleanup and created_new_cluster and cluster_id and emr:
            logger.warning("Attempting cleanup after failure...")
            try:
                emr.terminate_cluster(cluster_id)
            except Exception as cleanup_error:
                logger.error(f"Cleanup failed: {str(cleanup_error)}")

        try:
            pyspark_results_dir = os.path.join(
                EXPERIMENT_CONFIG["results_dir"], "pyspark"
            )
            metrics_file = metrics.save_to_file(pyspark_results_dir)
            logger.info(f"Metrics saved to: {metrics_file}")
        except Exception:
            pass

        raise


def run_all_benchmarks(
    cleanup: bool = True, experiment_id: str = None, dataset_mode: str = "sample"
) -> dict:
    """
    Run all benchmarks sequentially.
    """
    logger.info("=" * 60)
    logger.info("Running ALL AWS Benchmarks")
    logger.info("=" * 60)

    results = {"experiment_id": experiment_id, "benchmarks": {}}

    try:
        # 1. Storage Benchmark (no cluster needed)
        logger.info("\n[1/3] Running Storage Benchmark...")
        storage_results = run_storage_benchmark(
            cleanup=cleanup, experiment_id=f"{experiment_id}_storage"
        )
        results["benchmarks"]["storage"] = storage_results
        logger.info("✓ Storage benchmark complete\n")

        # 2. Big Data Benchmark (requires cluster)
        logger.info("\n[2/3] Running Big Data Benchmark...")
        bigdata_results = run_bigdata_benchmark(
            cleanup=cleanup,
            experiment_id=f"{experiment_id}_bigdata",
            dataset_mode=dataset_mode,
        )
        results["benchmarks"]["bigdata"] = bigdata_results
        logger.info("✓ Big data benchmark complete\n")

        # 3. ML Benchmark (requires cluster) - import from original
        logger.info("\n[3/3] Running ML Benchmark...")
        from run_aws_ml_experiment import run_aws_experiment

        ml_results = run_aws_experiment(
            cleanup=cleanup, experiment_id=f"{experiment_id}_ml"
        )
        results["benchmarks"]["ml"] = ml_results
        logger.info("✓ ML benchmark complete\n")

        logger.info("=" * 60)
        logger.info("✓ ALL AWS Benchmarks Completed Successfully!")
        logger.info("=" * 60)

        return results

    except Exception as e:
        logger.error(f"Benchmark suite failed: {str(e)}")
        raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run AWS EMR benchmarks")
    parser.add_argument(
        "--experiment-type",
        choices=["ml", "storage", "bigdata", "all"],
        default="ml",
        help="Type of experiment to run (default: ml)",
    )
    parser.add_argument(
        "--dataset-mode",
        choices=["sample", "large", "full"],
        default="sample",
        help="Dataset mode for bigdata benchmark: sample (1M rows), large (131M rows ~10GB), or full (262M rows ~21GB) (default: sample)",
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Do not cleanup resources after experiment",
    )
    parser.add_argument("--experiment-id", type=str, help="Custom experiment ID")

    args = parser.parse_args()

    try:
        cleanup = not args.no_cleanup

        if args.experiment_type == "storage":
            logger.info("Running STORAGE benchmark only")
            results = run_storage_benchmark(
                cleanup=cleanup, experiment_id=args.experiment_id
            )

        elif args.experiment_type == "bigdata":
            logger.info(f"Running BIG DATA benchmark ({args.dataset_mode} mode)")
            results = run_bigdata_benchmark(
                cleanup=cleanup,
                experiment_id=args.experiment_id,
                dataset_mode=args.dataset_mode,
            )

        elif args.experiment_type == "ml":
            logger.info("Running ML benchmark only")
            from run_aws_ml_experiment import run_aws_experiment

            results = run_aws_experiment(
                cleanup=cleanup, experiment_id=args.experiment_id
            )

        elif args.experiment_type == "all":
            logger.info(
                f"Running ALL benchmarks ({args.dataset_mode} mode for bigdata)"
            )
            results = run_all_benchmarks(
                cleanup=cleanup,
                experiment_id=args.experiment_id,
                dataset_mode=args.dataset_mode,
            )

        logger.info("\n✓ Benchmark completed successfully!")
        sys.exit(0)

    except Exception as e:
        logger.error(f"\n✗ Benchmark failed: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
