import argparse
import sys
import os
from datetime import datetime

from run_aws_benchmark import run_storage_benchmark, run_bigdata_benchmark
from run_aws_ml_experiment import run_aws_experiment
from utils.metrics_aggregator import MetricsAggregator
from utils.logger_config import setup_logger
from config import EXPERIMENT_CONFIG

logger = setup_logger(
    "aws_multi_benchmark", log_file="../data/results/aws_multi_benchmark.log"
)


def run_ml_multi(num_runs: int = 10, cleanup: bool = True) -> dict:
    """
    Run ML benchmark multiple times and aggregate results.

    Args:
        num_runs: Number of times to run the benchmark
        cleanup: Whether to cleanup resources after each run

    Returns:
        Aggregated metrics dictionary
    """
    logger.info("=" * 80)
    logger.info(f"Running AWS ML Benchmark - {num_runs} iterations")
    logger.info("=" * 80)

    aggregator = MetricsAggregator(benchmark_type="ml", provider="aws")

    for i in range(1, num_runs + 1):
        logger.info(f"\n>>> ML Run {i}/{num_runs}")
        try:
            experiment_id = f"ml_run{i:02d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            metrics = run_aws_experiment(cleanup=cleanup, experiment_id=experiment_id)
            aggregator.add_run(metrics)
            logger.info(f"✓ ML Run {i}/{num_runs} completed successfully")
        except Exception as e:
            logger.error(f"✗ ML Run {i}/{num_runs} failed: {str(e)}")
            # Continue with remaining runs

    # Aggregate and print summary
    aggregator.aggregate()
    aggregator.print_summary()

    return aggregator.get_metrics()


def run_storage_multi(num_runs: int = 10, cleanup: bool = True) -> dict:
    """
    Run storage benchmark multiple times and aggregate results.

    Args:
        num_runs: Number of times to run the benchmark
        cleanup: Whether to cleanup resources (not applicable for storage)

    Returns:
        Aggregated metrics dictionary
    """
    logger.info("=" * 80)
    logger.info(f"Running AWS Storage Benchmark - {num_runs} iterations")
    logger.info("=" * 80)

    aggregator = MetricsAggregator(benchmark_type="storage", provider="aws")

    for i in range(1, num_runs + 1):
        logger.info(f"\n>>> Storage Run {i}/{num_runs}")
        try:
            experiment_id = (
                f"storage_run{i:02d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            metrics = run_storage_benchmark(
                cleanup=cleanup, experiment_id=experiment_id
            )
            aggregator.add_run(metrics)
            logger.info(f"✓ Storage Run {i}/{num_runs} completed successfully")
        except Exception as e:
            logger.error(f"✗ Storage Run {i}/{num_runs} failed: {str(e)}")
            # Continue with remaining runs

    # Aggregate and print summary
    aggregator.aggregate()
    aggregator.print_summary()

    return aggregator.get_metrics()


def run_pyspark_multi(dataset_mode: str, num_runs: int, cleanup: bool = True) -> dict:
    """
    Run PySpark benchmark multiple times and aggregate results.
    Uses the same cluster for all runs to reduce provisioning overhead.

    Args:
        dataset_mode: Dataset mode (sample, large, or full)
        num_runs: Number of times to run the benchmark
        cleanup: Whether to cleanup resources after all runs

    Returns:
        Aggregated metrics dictionary
    """
    from aws.emr_cluster import EMRCluster
    from config import AWS_CONFIG, EXPERIMENT_CONFIG

    logger.info("=" * 80)
    logger.info(
        f"Running AWS PySpark Benchmark ({dataset_mode}) - {num_runs} iterations"
    )
    logger.info("=" * 80)

    aggregator = MetricsAggregator(
        benchmark_type="pyspark", provider="aws", dataset_mode=dataset_mode
    )

    # Create cluster once for all runs
    logger.info("\n>>> Creating shared cluster for all runs...")
    emr = EMRCluster(region=AWS_CONFIG["region"])
    cluster_id = None

    try:
        cluster_id = emr.create_cluster()
        logger.info(f"Cluster created: {cluster_id}")
        logger.info("Waiting for cluster to be ready...")
        if not emr.wait_for_cluster_ready(
            cluster_id, timeout_minutes=EXPERIMENT_CONFIG["timeout_minutes"]
        ):
            raise RuntimeError("Cluster failed to become ready")
        logger.info("✓ Cluster is ready for benchmarks\n")

        # Run all benchmarks on the same cluster
        for i in range(1, num_runs + 1):
            logger.info(f"\n>>> PySpark {dataset_mode} Run {i}/{num_runs}")
            try:
                experiment_id = f"pyspark_{dataset_mode}_run{i:02d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                metrics = run_bigdata_benchmark(
                    cleanup=False,  # Don't cleanup between runs
                    experiment_id=experiment_id,
                    dataset_mode=dataset_mode,
                    cluster_id=cluster_id,
                    emr_instance=emr
                )
                aggregator.add_run(metrics)
                logger.info(
                    f"✓ PySpark {dataset_mode} Run {i}/{num_runs} completed successfully"
                )
            except Exception as e:
                logger.error(
                    f"✗ PySpark {dataset_mode} Run {i}/{num_runs} failed: {str(e)}"
                )
                # Continue with remaining runs

    finally:
        # Cleanup cluster after all runs
        if cleanup and cluster_id:
            logger.info("\n>>> Cleaning up shared cluster...")
            try:
                emr.terminate_cluster(cluster_id)
                logger.info("✓ Cluster termination initiated")
            except Exception as e:
                logger.error(f"✗ Failed to terminate cluster: {str(e)}")

    # Aggregate and print summary
    aggregator.aggregate()
    aggregator.print_summary()

    return aggregator.get_metrics()


def run_all_multi(
    ml_runs: int = 10,
    storage_runs: int = 10,
    pyspark_sample_runs: int = 10,
    pyspark_large_runs: int = 5,
    cleanup: bool = True,
) -> dict:
    """
    Run all benchmarks with multiple iterations.

    Args:
        ml_runs: Number of ML benchmark runs
        storage_runs: Number of storage benchmark runs
        pyspark_sample_runs: Number of PySpark sample runs
        pyspark_large_runs: Number of PySpark large runs
        cleanup: Whether to cleanup resources after each run

    Returns:
        Dictionary with all aggregated results
    """
    start_time = datetime.now()
    logger.info("=" * 80)
    logger.info("Starting AWS Complete Multi-Run Benchmark Suite")
    logger.info(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)
    logger.info(f"ML Runs: {ml_runs}")
    logger.info(f"Storage Runs: {storage_runs}")
    logger.info(f"PySpark Sample Runs: {pyspark_sample_runs}")
    logger.info(f"PySpark Large Runs: {pyspark_large_runs}")
    logger.info("=" * 80)

    results = {
        "provider": "aws",
        "start_time": start_time.isoformat(),
        "benchmarks": {},
    }
    # Run PySpark sample benchmarks
    logger.info("\n[1/4] Running PySpark Sample Benchmarks...")
    try:
        pyspark_sample_results = run_pyspark_multi(
            dataset_mode="sample", num_runs=pyspark_sample_runs, cleanup=cleanup
        )
        results["benchmarks"]["pyspark_sample"] = pyspark_sample_results
        logger.info("✓ PySpark sample benchmarks complete\n")
    except Exception as e:
        logger.error(f"✗ PySpark sample benchmarks failed: {str(e)}")
        results["benchmarks"]["pyspark_sample"] = {"error": str(e)}

    # Run PySpark large benchmarks
    logger.info("\n[2/4] Running PySpark Large Benchmarks...")
    try:
        pyspark_large_results = run_pyspark_multi(
            dataset_mode="large", num_runs=pyspark_large_runs, cleanup=cleanup
        )
        results["benchmarks"]["pyspark_large"] = pyspark_large_results
        logger.info("✓ PySpark large benchmarks complete\n")
    except Exception as e:
        logger.error(f"✗ PySpark large benchmarks failed: {str(e)}")
        results["benchmarks"]["pyspark_large"] = {"error": str(e)}

    # Run ML benchmarks
    logger.info("\n[3/4] Running ML Benchmarks...")
    try:
        ml_results = run_ml_multi(num_runs=ml_runs, cleanup=cleanup)
        results["benchmarks"]["ml"] = ml_results
        logger.info("✓ ML benchmarks complete\n")
    except Exception as e:
        logger.error(f"✗ ML benchmarks failed: {str(e)}")
        results["benchmarks"]["ml"] = {"error": str(e)}

    # Run storage benchmarks
    logger.info("\n[4/4] Running Storage Benchmarks...")
    try:
        storage_results = run_storage_multi(num_runs=storage_runs, cleanup=cleanup)
        results["benchmarks"]["storage"] = storage_results
        logger.info("✓ Storage benchmarks complete\n")
    except Exception as e:
        logger.error(f"✗ Storage benchmarks failed: {str(e)}")
        results["benchmarks"]["storage"] = {"error": str(e)}

    # Calculate total time
    end_time = datetime.now()
    total_duration = (end_time - start_time).total_seconds()
    results["end_time"] = end_time.isoformat()
    results["total_duration_seconds"] = total_duration

    logger.info("=" * 80)
    logger.info("AWS Complete Multi-Run Benchmark Suite Finished!")
    logger.info(f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Total Duration: {total_duration / 3600:.2f} hours")
    logger.info("=" * 80)

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run AWS benchmarks with multiple iterations"
    )
    parser.add_argument(
        "--benchmark-type",
        choices=["ml", "storage", "pyspark-sample", "pyspark-large", "all"],
        default="all",
        help="Type of benchmark to run (default: all)",
    )
    parser.add_argument(
        "--ml-runs",
        type=int,
        default=10,
        help="Number of ML benchmark runs (default: 10)",
    )
    parser.add_argument(
        "--storage-runs",
        type=int,
        default=10,
        help="Number of storage benchmark runs (default: 10)",
    )
    parser.add_argument(
        "--pyspark-sample-runs",
        type=int,
        default=10,
        help="Number of PySpark sample benchmark runs (default: 10)",
    )
    parser.add_argument(
        "--pyspark-large-runs",
        type=int,
        default=5,
        help="Number of PySpark large benchmark runs (default: 5)",
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Do not cleanup resources after experiments",
    )

    args = parser.parse_args()

    try:
        cleanup = not args.no_cleanup

        if args.benchmark_type == "ml":
            logger.info("Running ML benchmarks only")
            results = run_ml_multi(num_runs=args.ml_runs, cleanup=cleanup)

        elif args.benchmark_type == "storage":
            logger.info("Running Storage benchmarks only")
            results = run_storage_multi(num_runs=args.storage_runs, cleanup=cleanup)

        elif args.benchmark_type == "pyspark-sample":
            logger.info("Running PySpark sample benchmarks only")
            results = run_pyspark_multi(
                dataset_mode="sample",
                num_runs=args.pyspark_sample_runs,
                cleanup=cleanup,
            )

        elif args.benchmark_type == "pyspark-large":
            logger.info("Running PySpark large benchmarks only")
            results = run_pyspark_multi(
                dataset_mode="large", num_runs=args.pyspark_large_runs, cleanup=cleanup
            )

        elif args.benchmark_type == "all":
            logger.info("Running ALL benchmarks")
            results = run_all_multi(
                ml_runs=args.ml_runs,
                storage_runs=args.storage_runs,
                pyspark_sample_runs=args.pyspark_sample_runs,
                pyspark_large_runs=args.pyspark_large_runs,
                cleanup=cleanup,
            )

        logger.info("\n✓ Multi-run benchmark completed successfully!")
        sys.exit(0)

    except Exception as e:
        logger.error(f"\n✗ Multi-run benchmark failed: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
