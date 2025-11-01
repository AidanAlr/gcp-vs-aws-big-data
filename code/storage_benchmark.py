import os
import sys
import time
import tempfile
import json
from datetime import datetime
from pathlib import Path

from config import AWS_CONFIG, GCP_CONFIG
from utils.logger_config import setup_logger
from utils.metrics_collector import MetricsCollector

import boto3

from google.cloud import storage as gcp_storage

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = setup_logger("storage_benchmark")


class StorageBenchmark:
    """Benchmark storage performance for cloud providers"""

    def __init__(self, provider="aws", test_file_path=None, max_file_size_mb=500):
        """
        Initialize storage benchmark

        Args:
            provider: "aws" or "gcp"
            test_file_path: Path to test file (default: transactions sample)
            max_file_size_mb: Maximum file size for upload tests (default: 500MB)
        """
        self.provider = provider.lower()
        self.max_file_size_mb = max_file_size_mb
        self.metrics = MetricsCollector(f"storage_benchmark_{provider}")

        # Set default test file
        if test_file_path is None:
            project_root = Path(__file__).parent.parent.parent
            test_file_path = (
                project_root / "data/transactions/transactions_sample_10k.csv"
            )

        self.test_file_path = Path(test_file_path)

        if not self.test_file_path.exists():
            raise FileNotFoundError(f"Test file not found: {self.test_file_path}")

        # Get file size
        self.file_size_bytes = self.test_file_path.stat().st_size
        self.file_size_mb = self.file_size_bytes / (1024 * 1024)

        logger.info(
            f"Test file: {self.test_file_path.name} ({self.file_size_mb:.2f} MB)"
        )

        # Check file size limit
        if self.file_size_mb > max_file_size_mb:
            logger.warning(
                f"File size ({self.file_size_mb:.2f} MB) exceeds max ({max_file_size_mb} MB)"
            )
            logger.warning("Upload tests will be skipped for safety")

        # Initialize cloud clients
        if self.provider == "aws":
            self.s3_client = boto3.client("s3")
            self.bucket_name = AWS_CONFIG["s3_bucket"]
        elif self.provider == "gcp":
            self.gcs_client = gcp_storage.Client(project=GCP_CONFIG["project_id"])
            self.bucket_name = GCP_CONFIG["gcs_bucket"]
        else:
            raise ValueError(f"Unknown provider: {provider}")

        logger.info(f"Initialized {provider.upper()} storage benchmark")
        logger.info(f"Bucket: {self.bucket_name}")

    def benchmark_upload(self, test_key=None):
        """Benchmark file upload to cloud storage"""
        if self.file_size_mb > self.max_file_size_mb:
            logger.warning(
                f"Skipping upload test - file too large ({self.file_size_mb:.2f} MB > {self.max_file_size_mb} MB)"
            )
            return None

        if test_key is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            test_key = f"benchmarks/storage_test_{timestamp}.csv"

        logger.info(f"Testing upload: {self.test_file_path.name} -> {test_key}")

        start_time = time.time()

        try:
            if self.provider == "aws":
                self.s3_client.upload_file(
                    str(self.test_file_path), self.bucket_name, test_key
                )
            elif self.provider == "gcp":
                bucket = self.gcs_client.bucket(self.bucket_name)
                blob = bucket.blob(test_key)
                # Increase timeout for large files (default is 60s per chunk)
                blob.upload_from_filename(str(self.test_file_path), timeout=600)

            upload_time = time.time() - start_time
            throughput_mbps = (
                (self.file_size_mb / upload_time) if upload_time > 0 else 0
            )

            result = {
                "operation": "upload",
                "file_size_mb": round(self.file_size_mb, 2),
                "duration_seconds": round(upload_time, 3),
                "throughput_mbps": round(throughput_mbps, 2),
                "key": test_key,
            }

            logger.info(
                f"Upload completed: {upload_time:.2f}s ({throughput_mbps:.2f} MB/s)"
            )
            self.metrics.record_metric("upload_time_seconds", upload_time)
            self.metrics.record_metric("upload_throughput_mbps", throughput_mbps)

            return result

        except Exception as e:
            logger.error(f"Upload failed: {str(e)}")
            return {"operation": "upload", "error": str(e)}

    def benchmark_download(self, test_key):
        """Benchmark file download from cloud storage"""
        logger.info(f"Testing download: {test_key}")

        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_path = tmp_file.name

        start_time = time.time()

        try:
            if self.provider == "aws":
                self.s3_client.download_file(self.bucket_name, test_key, tmp_path)
            elif self.provider == "gcp":
                bucket = self.gcs_client.bucket(self.bucket_name)
                blob = bucket.blob(test_key)
                blob.download_to_filename(tmp_path)

            download_time = time.time() - start_time

            # Get downloaded file size
            downloaded_size_mb = Path(tmp_path).stat().st_size / (1024 * 1024)
            throughput_mbps = (
                (downloaded_size_mb / download_time) if download_time > 0 else 0
            )

            result = {
                "operation": "download",
                "file_size_mb": round(downloaded_size_mb, 2),
                "duration_seconds": round(download_time, 3),
                "throughput_mbps": round(throughput_mbps, 2),
                "key": test_key,
            }

            logger.info(
                f"Download completed: {download_time:.2f}s ({throughput_mbps:.2f} MB/s)"
            )
            self.metrics.record_metric("download_time_seconds", download_time)
            self.metrics.record_metric("download_throughput_mbps", throughput_mbps)

            # Cleanup temp file
            os.unlink(tmp_path)

            return result

        except Exception as e:
            logger.error(f"Download failed: {str(e)}")
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            return {"operation": "download", "error": str(e)}

    def benchmark_read_performance(self, test_key, chunk_size_mb=1):
        """Benchmark sequential read performance"""
        logger.info(
            f"Testing read performance: {test_key} (chunk size: {chunk_size_mb} MB)"
        )

        chunk_size_bytes = int(chunk_size_mb * 1024 * 1024)
        total_bytes_read = 0

        start_time = time.time()

        try:
            if self.provider == "aws":
                response = self.s3_client.get_object(
                    Bucket=self.bucket_name, Key=test_key
                )
                stream = response["Body"]

                while True:
                    chunk = stream.read(chunk_size_bytes)
                    if not chunk:
                        break
                    total_bytes_read += len(chunk)

            elif self.provider == "gcp":
                bucket = self.gcs_client.bucket(self.bucket_name)
                blob = bucket.blob(test_key)

                # Download in chunks
                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    tmp_path = tmp_file.name

                blob.download_to_filename(tmp_path)

                with open(tmp_path, "rb") as f:
                    while True:
                        chunk = f.read(chunk_size_bytes)
                        if not chunk:
                            break
                        total_bytes_read += len(chunk)

                os.unlink(tmp_path)

            read_time = time.time() - start_time
            total_mb_read = total_bytes_read / (1024 * 1024)
            throughput_mbps = (total_mb_read / read_time) if read_time > 0 else 0

            result = {
                "operation": "sequential_read",
                "total_mb_read": round(total_mb_read, 2),
                "chunk_size_mb": chunk_size_mb,
                "duration_seconds": round(read_time, 3),
                "throughput_mbps": round(throughput_mbps, 2),
                "key": test_key,
            }

            logger.info(
                f"Read completed: {read_time:.2f}s ({throughput_mbps:.2f} MB/s)"
            )
            self.metrics.record_metric("read_time_seconds", read_time)
            self.metrics.record_metric("read_throughput_mbps", throughput_mbps)

            return result

        except Exception as e:
            logger.error(f"Read performance test failed: {str(e)}")
            return {"operation": "sequential_read", "error": str(e)}

    def benchmark_write_performance(self, test_key, data_size_mb=5):
        """Benchmark write performance with synthetic data"""
        logger.info(f"Testing write performance: {data_size_mb} MB")

        # Generate synthetic data
        data_size_bytes = int(data_size_mb * 1024 * 1024)
        synthetic_data = b"X" * data_size_bytes

        # Write to temp file first
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_path = tmp_file.name
            tmp_file.write(synthetic_data)

        start_time = time.time()

        try:
            if self.provider == "aws":
                self.s3_client.upload_file(tmp_path, self.bucket_name, test_key)
            elif self.provider == "gcp":
                bucket = self.gcs_client.bucket(self.bucket_name)
                blob = bucket.blob(test_key)
                blob.upload_from_filename(tmp_path, timeout=600)

            write_time = time.time() - start_time
            throughput_mbps = (data_size_mb / write_time) if write_time > 0 else 0

            result = {
                "operation": "write",
                "data_size_mb": data_size_mb,
                "duration_seconds": round(write_time, 3),
                "throughput_mbps": round(throughput_mbps, 2),
                "key": test_key,
            }

            logger.info(
                f"Write completed: {write_time:.2f}s ({throughput_mbps:.2f} MB/s)"
            )
            self.metrics.record_metric("write_time_seconds", write_time)
            self.metrics.record_metric("write_throughput_mbps", throughput_mbps)

            # Cleanup
            os.unlink(tmp_path)

            return result

        except Exception as e:
            logger.error(f"Write performance test failed: {str(e)}")
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            return {"operation": "write", "error": str(e)}

    def cleanup_test_files(self, test_keys):
        """Delete test files from cloud storage"""
        logger.info(f"Cleaning up {len(test_keys)} test files")

        for key in test_keys:
            try:
                if self.provider == "aws":
                    self.s3_client.delete_object(Bucket=self.bucket_name, Key=key)
                elif self.provider == "gcp":
                    bucket = self.gcs_client.bucket(self.bucket_name)
                    blob = bucket.blob(key)
                    blob.delete()
                logger.info(f"Deleted: {key}")
            except Exception as e:
                logger.warning(f"Failed to delete {key}: {str(e)}")

    def run_full_benchmark(self):
        """Run complete storage benchmark suite"""
        logger.info("=" * 60)
        logger.info(f"Starting {self.provider.upper()} Storage Benchmark")
        logger.info("=" * 60)

        results = {
            "provider": self.provider,
            "timestamp": datetime.now().isoformat(),
            "test_file": str(self.test_file_path.name),
            "file_size_mb": round(self.file_size_mb, 2),
            "bucket": self.bucket_name,
            "tests": [],
        }

        test_keys = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Test 1: Upload
        upload_key = f"benchmarks/upload_test_{timestamp}.csv"
        upload_result = self.benchmark_upload(upload_key)
        if upload_result and "error" not in upload_result:
            results["tests"].append(upload_result)
            test_keys.append(upload_key)

            # Test 2: Download (only if upload succeeded)
            download_result = self.benchmark_download(upload_key)
            if download_result:
                results["tests"].append(download_result)

            # Test 3: Read Performance (only if upload succeeded)
            read_result = self.benchmark_read_performance(upload_key, chunk_size_mb=1)
            if read_result:
                results["tests"].append(read_result)

        # Test 4: Write Performance
        write_key = f"benchmarks/write_test_{timestamp}.dat"
        write_result = self.benchmark_write_performance(write_key, data_size_mb=5)
        if write_result and "error" not in write_result:
            results["tests"].append(write_result)
            test_keys.append(write_key)

        # Calculate summary statistics
        results["summary"] = self._calculate_summary(results["tests"])

        # Cleanup test files
        if test_keys:
            self.cleanup_test_files(test_keys)

        # Save metrics
        self.metrics.add_metadata("test_summary", results["summary"])

        logger.info("=" * 60)
        logger.info("Storage Benchmark Complete")
        logger.info("=" * 60)
        self._print_summary(results["summary"])

        return results

    def _calculate_summary(self, tests):
        """Calculate summary statistics from test results"""
        summary = {}

        for test in tests:
            if "error" in test:
                continue

            op = test["operation"]
            if op not in summary:
                summary[op] = {
                    "avg_throughput_mbps": 0,
                    "avg_duration_seconds": 0,
                    "count": 0,
                }

            summary[op]["avg_throughput_mbps"] += test.get("throughput_mbps", 0)
            summary[op]["avg_duration_seconds"] += test.get("duration_seconds", 0)
            summary[op]["count"] += 1

        # Calculate averages
        for op in summary:
            if summary[op]["count"] > 0:
                summary[op]["avg_throughput_mbps"] /= summary[op]["count"]
                summary[op]["avg_duration_seconds"] /= summary[op]["count"]
                summary[op]["avg_throughput_mbps"] = round(
                    summary[op]["avg_throughput_mbps"], 2
                )
                summary[op]["avg_duration_seconds"] = round(
                    summary[op]["avg_duration_seconds"], 3
                )

        return summary

    def _print_summary(self, summary):
        """Print summary statistics"""
        print("\nSummary Statistics:")
        print("-" * 60)
        for operation, stats in summary.items():
            print(f"\n{operation.upper()}:")
            print(f"  Average Throughput: {stats['avg_throughput_mbps']:.2f} MB/s")
            print(f"  Average Duration:   {stats['avg_duration_seconds']:.3f} seconds")
            print(f"  Test Count:         {stats['count']}")
        print()


def main():
    """Main execution function"""
    import argparse

    parser = argparse.ArgumentParser(description="Storage Performance Benchmark")
    parser.add_argument(
        "--provider",
        choices=["aws", "gcp"],
        required=True,
        help="Cloud provider to benchmark",
    )
    parser.add_argument(
        "--test-file",
        type=str,
        default=None,
        help="Path to test file (default: transactions sample)",
    )
    parser.add_argument(
        "--max-size-mb",
        type=int,
        default=500,
        help="Maximum file size for upload tests in MB (default: 500)",
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Output file for results (JSON)"
    )

    args = parser.parse_args()

    try:
        # Run benchmark
        benchmark = StorageBenchmark(
            provider=args.provider,
            test_file_path=args.test_file,
            max_file_size_mb=args.max_size_mb,
        )

        results = benchmark.run_full_benchmark()

        # Save results
        if args.output:
            output_path = Path(args.output)
        else:
            project_root = Path(__file__).parent.parent.parent
            results_dir = project_root / "data/results"
            results_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = (
                results_dir / f"storage_benchmark_{args.provider}_{timestamp}.json"
            )

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to: {output_path}")

        # Save metrics
        project_root = Path(__file__).parent.parent.parent
        results_dir = project_root / "data/results"
        metrics_path = benchmark.metrics.save_to_file(str(results_dir))
        logger.info(f"Metrics saved to: {metrics_path}")

        print(f"\n✓ Storage benchmark completed successfully!")
        print(f"  Results: {output_path}")
        print(f"  Metrics: {metrics_path}")

    except Exception as e:
        logger.error(f"Benchmark failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
