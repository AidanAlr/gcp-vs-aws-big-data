import argparse
import sys
import os

from gcp.dataproc_cluster import DataprocCluster
from gcp.dataproc_runner import DataprocRunner
from utils.metrics_collector import MetricsCollector
from utils.cost_calculator import CostCalculator
from utils.logger_config import setup_logger
from config import GCP_CONFIG, ML_CONFIG, EXPERIMENT_CONFIG

logger = setup_logger("gcp_experiment", log_file="../data/results/gcp_experiment.log")


def run_gcp_experiment(cleanup: bool = True, experiment_id: str = None) -> dict:
    """
    Run complete GCP Dataproc experiment.

    Args:
        cleanup: Whether to cleanup resources after experiment
        experiment_id: Optional custom experiment ID

    Returns:
        Dictionary with experiment results and metrics
    """
    # Initialize metrics collector
    metrics = MetricsCollector(provider="gcp", experiment_id=experiment_id)

    # Initialize cost calculator
    cost_calc = CostCalculator(provider="gcp")

    # Track experiment metadata
    metrics.add_metadata(
        "cluster_config",
        {
            "master_machine": GCP_CONFIG["master_machine_type"],
            "worker_machine": GCP_CONFIG["worker_machine_type"],
            "worker_count": GCP_CONFIG["worker_count"],
            "region": GCP_CONFIG["region"],
            "project_id": GCP_CONFIG["project_id"],
        },
    )
    metrics.add_metadata("ml_config", ML_CONFIG)

    cluster_name = None
    dataproc = None

    try:
        logger.info("=" * 60)
        logger.info("Starting GCP Dataproc Experiment")
        logger.info("=" * 60)

        # Start total time tracking
        metrics.start_timer("total_time")

        # =====================================================================
        # Step 1: Create Dataproc Cluster
        # =====================================================================
        logger.info("\n[1/6] Creating Dataproc cluster...")
        dataproc = DataprocCluster(
            project_id=GCP_CONFIG["project_id"], region=GCP_CONFIG["region"]
        )

        metrics.start_timer("cluster_provisioning")
        cluster_name = dataproc.create_cluster()
        metrics.add_metadata("cluster_name", cluster_name)

        # Note: create_cluster() waits for cluster to be ready
        provisioning_time = metrics.stop_timer("cluster_provisioning")

        # =====================================================================
        # Step 2: Upload ML Script to GCS
        # =====================================================================
        logger.info("\n[2/6] Uploading ML training script to GCS...")
        runner = DataprocRunner(
            cluster_name,
            project_id=GCP_CONFIG["project_id"],
            region=GCP_CONFIG["region"],
        )

        script_local_path = os.path.join(os.path.dirname(__file__), "ml_job.py")
        script_gcs_uri = runner.upload_script_to_gcs(script_local_path)

        metrics.add_metadata("script_gcs_uri", script_gcs_uri)

        # =====================================================================
        # Step 3: Submit Spark Job
        # =====================================================================
        logger.info("\n[3/6] Submitting Spark ML training job...")

        # Prepare job arguments
        data_path = f"{GCP_CONFIG['gcs_data_path']}{ML_CONFIG['dataset']}"
        output_path = f"{GCP_CONFIG['gcs_results_path']}{metrics.experiment_id}"

        job_args = [
            "--dataset",
            ML_CONFIG["dataset"],
            "--epochs",
            str(ML_CONFIG["epochs"]),
            "--batch-size",
            str(ML_CONFIG["batch_size"]),
            "--data-path",
            data_path,
            "--output-path",
            output_path,
        ]

        metrics.start_timer("ml_training")
        job_id = runner.submit_pyspark_job(
            main_python_file_uri=script_gcs_uri,
            job_name=f"ml-training-{metrics.experiment_id}",
            args=job_args,
        )

        metrics.add_metadata("job_id", job_id)

        # =====================================================================
        # Step 4: Monitor Job Completion
        # =====================================================================
        logger.info("\n[4/6] Monitoring job execution...")

        if not runner.wait_for_job_completion(
            job_id, timeout_minutes=EXPERIMENT_CONFIG["timeout_minutes"]
        ):
            raise RuntimeError("ML training job failed or timed out")

        training_time = metrics.stop_timer("ml_training")

        # =====================================================================
        # Step 5: Calculate Costs
        # =====================================================================
        logger.info("\n[5/6] Calculating costs...")

        # Total cluster runtime (from start to now)
        cluster_runtime = provisioning_time + training_time

        costs = cost_calc.calculate_total_experiment_cost(
            cluster_runtime_seconds=cluster_runtime,
            data_size_gb=0.5,  # Fashion-MNIST is ~500 MB
        )

        for cost_component, amount in costs.items():
            if cost_component != "total":
                metrics.record_cost(cost_component, amount)

        cost_calc.print_cost_estimate(cluster_runtime, data_size_gb=0.5)

        # =====================================================================
        # Step 6: Cleanup Resources
        # =====================================================================
        if cleanup:
            logger.info("\n[6/6] Cleaning up resources...")
            dataproc.delete_cluster(cluster_name)
            logger.info("Cluster deletion initiated (not waiting for completion)")
        else:
            logger.info("\n[6/6] Skipping cleanup (cluster will remain active)")
            logger.warning(f"Remember to manually delete cluster: {cluster_name}")

        # Stop total timer
        total_time = metrics.stop_timer("total_time")

        # =====================================================================
        # Results Summary
        # =====================================================================
        logger.info("\n" + "=" * 60)
        logger.info("GCP Experiment Completed Successfully!")
        logger.info("=" * 60)

        metrics.print_summary()

        # Save metrics to file
        ml_results_dir = os.path.join(EXPERIMENT_CONFIG["results_dir"], "ml")
        metrics_file = metrics.save_to_file(ml_results_dir)

        logger.info(f"Metrics saved to: {metrics_file}")
        logger.info("=" * 60)

        return metrics.get_metrics()

    except Exception as e:
        logger.error(f"\nExperiment failed: {str(e)}")
        metrics.record_error(str(e), error_type=type(e).__name__)

        # Cleanup on failure if requested
        if (
            cleanup
            and EXPERIMENT_CONFIG.get("cleanup_on_failure", True)
            and cluster_name
        ):
            logger.warning("Attempting cleanup after failure...")
            try:
                if dataproc:
                    dataproc.delete_cluster(cluster_name)
            except Exception as cleanup_error:
                logger.error(f"Cleanup failed: {str(cleanup_error)}")

        # Save metrics even on failure
        try:
            ml_results_dir = os.path.join(EXPERIMENT_CONFIG["results_dir"], "ml")
            metrics_file = metrics.save_to_file(ml_results_dir)
            logger.info(f"Metrics saved to: {metrics_file}")
        except Exception:
            pass

        raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run GCP Dataproc ML training experiment"
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Do not cleanup resources after experiment",
    )
    parser.add_argument("--experiment-id", type=str, help="Custom experiment ID")

    args = parser.parse_args()

    try:
        results = run_gcp_experiment(
            cleanup=not args.no_cleanup, experiment_id=args.experiment_id
        )
        logger.info("\nExperiment completed successfully!")
        sys.exit(0)

    except Exception as e:
        logger.error(f"\nExperiment failed: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
