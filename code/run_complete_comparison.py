import argparse
import sys
import os
import json
from datetime import datetime

from run_multi_benchmark_aws import run_all_multi as run_aws_multi
from run_multi_benchmark_gcp import run_all_multi as run_gcp_multi
from utils.logger_config import setup_logger
from config import EXPERIMENT_CONFIG

logger = setup_logger(
    "complete_comparison", log_file="../data/results/complete_comparison.log"
)


def run_complete_comparison(
    ml_runs: int = 10,
    storage_runs: int = 10,
    pyspark_sample_runs: int = 10,
    pyspark_large_runs: int = 5,
    cleanup: bool = True,
    run_aws: bool = True,
    run_gcp: bool = True,
) -> dict:
    """
    Run complete benchmark comparison between AWS and GCP.

    Args:
        ml_runs: Number of ML benchmark runs
        storage_runs: Number of storage benchmark runs
        pyspark_sample_runs: Number of PySpark sample runs
        pyspark_large_runs: Number of PySpark large runs
        cleanup: Whether to cleanup resources after each run
        run_aws: Whether to run AWS benchmarks
        run_gcp: Whether to run GCP benchmarks

    Returns:
        Dictionary with complete comparison results
    """
    start_time = datetime.now()

    logger.info("=" * 90)
    logger.info(" " * 20 + "AWS vs GCP Complete Benchmark Comparison")
    logger.info("=" * 90)
    logger.info(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("")
    logger.info("Benchmark Configuration:")
    logger.info(f"  ML Runs:             {ml_runs}")
    logger.info(f"  Storage Runs:        {storage_runs}")
    logger.info(f"  PySpark Sample Runs: {pyspark_sample_runs} (1M rows)")
    logger.info(f"  PySpark Large Runs:  {pyspark_large_runs} (131M rows ~10GB)")
    logger.info(f"  Cleanup Enabled:     {cleanup}")
    logger.info("")
    logger.info(f"Running:")
    logger.info(f"  AWS Benchmarks:      {'Yes' if run_aws else 'No'}")
    logger.info(f"  GCP Benchmarks:      {'Yes' if run_gcp else 'No'}")
    logger.info("=" * 90)

    results = {
        "comparison_start_time": start_time.isoformat(),
        "configuration": {
            "ml_runs": ml_runs,
            "storage_runs": storage_runs,
            "pyspark_sample_runs": pyspark_sample_runs,
            "pyspark_large_runs": pyspark_large_runs,
            "cleanup": cleanup,
        },
        "providers": {},
    }

    # Run AWS benchmarks
    if run_aws:
        logger.info("\n" + "=" * 90)
        logger.info(" " * 30 + "PHASE 1: AWS BENCHMARKS")
        logger.info("=" * 90)
        try:
            aws_results = run_aws_multi(
                ml_runs=ml_runs,
                storage_runs=storage_runs,
                pyspark_sample_runs=pyspark_sample_runs,
                pyspark_large_runs=pyspark_large_runs,
                cleanup=cleanup,
            )
            results["providers"]["aws"] = aws_results
            logger.info("\n✓ AWS benchmarks completed successfully!")
        except Exception as e:
            logger.error(f"\n✗ AWS benchmarks failed: {str(e)}")
            results["providers"]["aws"] = {"error": str(e)}
            import traceback

            traceback.print_exc()

    # Run GCP benchmarks
    if run_gcp:
        logger.info("\n" + "=" * 90)
        logger.info(" " * 30 + "PHASE 2: GCP BENCHMARKS")
        logger.info("=" * 90)
        try:
            gcp_results = run_gcp_multi(
                ml_runs=ml_runs,
                storage_runs=storage_runs,
                pyspark_sample_runs=pyspark_sample_runs,
                pyspark_large_runs=pyspark_large_runs,
                cleanup=cleanup,
            )
            results["providers"]["gcp"] = gcp_results
            logger.info("\n✓ GCP benchmarks completed successfully!")
        except Exception as e:
            logger.error(f"\n✗ GCP benchmarks failed: {str(e)}")
            results["providers"]["gcp"] = {"error": str(e)}
            import traceback

            traceback.print_exc()

    # Calculate total duration
    end_time = datetime.now()
    total_duration = (end_time - start_time).total_seconds()
    results["comparison_end_time"] = end_time.isoformat()
    results["total_duration_seconds"] = total_duration
    results["total_duration_hours"] = total_duration / 3600

    # Comparison results are available in the results dictionary
    logger.info("\n" + "=" * 90)
    logger.info(" " * 25 + "COMPARISON COMPLETE")
    logger.info("=" * 90)

    # Print final summary
    logger.info("\n" + "=" * 90)
    logger.info(" " * 20 + "COMPLETE COMPARISON FINISHED!")
    logger.info("=" * 90)
    logger.info(f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(
        f"Total Duration: {total_duration / 3600:.2f} hours ({total_duration / 60:.1f} minutes)"
    )
    logger.info("")

    if (
        run_aws
        and "aws" in results["providers"]
        and "error" not in results["providers"]["aws"]
    ):
        aws_duration = (
            results["providers"]["aws"].get("total_duration_seconds", 0) / 3600
        )
        logger.info(f"AWS Total Time: {aws_duration:.2f} hours")

    if (
        run_gcp
        and "gcp" in results["providers"]
        and "error" not in results["providers"]["gcp"]
    ):
        gcp_duration = (
            results["providers"]["gcp"].get("total_duration_seconds", 0) / 3600
        )
        logger.info(f"GCP Total Time: {gcp_duration:.2f} hours")

    logger.info("")
    logger.info("Results location:")
    logger.info(
        f"  Individual runs:     {EXPERIMENT_CONFIG['results_dir']}/{{ml,pyspark,storage}}/"
    )
    logger.info("=" * 90)

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run complete AWS vs GCP benchmark comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete comparison with default settings (10/10/10/5 runs):
  python run_complete_comparison.py

  # Run with fewer iterations for faster testing:
  python run_complete_comparison.py --ml-runs 3 --storage-runs 3 \\
      --pyspark-sample-runs 3 --pyspark-large-runs 2

  # Run only AWS benchmarks:
  python run_complete_comparison.py --aws-only

  # Run only GCP benchmarks:
  python run_complete_comparison.py --gcp-only
        """,
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
    parser.add_argument(
        "--aws-only", action="store_true", help="Run only AWS benchmarks"
    )
    parser.add_argument(
        "--gcp-only", action="store_true", help="Run only GCP benchmarks"
    )

    args = parser.parse_args()

    try:
        run_aws = not args.gcp_only
        run_gcp = not args.aws_only
        cleanup = not args.no_cleanup

        results = run_complete_comparison(
            ml_runs=args.ml_runs,
            storage_runs=args.storage_runs,
            pyspark_sample_runs=0,
            pyspark_large_runs=args.pyspark_large_runs,
            cleanup=cleanup,
            run_aws=run_aws,
            run_gcp=run_gcp,
        )

        logger.info("\n✓ Complete comparison finished successfully!")
        sys.exit(0)

    except KeyboardInterrupt:
        logger.warning("\n\nBenchmark interrupted by user!")
        sys.exit(130)

    except Exception as e:
        logger.error(f"\n✗ Complete comparison failed: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
