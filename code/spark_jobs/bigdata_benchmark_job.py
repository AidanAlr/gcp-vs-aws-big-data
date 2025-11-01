import argparse
import json
import time
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    sum as spark_sum,
    avg,
    count,
    when,
    round as spark_round,
    desc,
    asc,
)


class BigDataBenchmark:
    """PySpark benchmark for big data operations"""

    def __init__(self, data_path, output_path, dataset_mode="sample"):
        """
        Initialize benchmark

        Args:
            data_path: Path to input data (S3 or GCS)
            output_path: Path for output results
            dataset_mode: "sample" or "full"
        """
        self.data_path = data_path
        self.output_path = output_path
        self.dataset_mode = dataset_mode

        # Initialize Spark session
        self.spark = SparkSession.builder.appName(
            f"BigDataBenchmark-{dataset_mode}"
        ).getOrCreate()

        self.spark.sparkContext.setLogLevel("WARN")

        self.results = {
            "benchmark_type": "big_data",
            "dataset_mode": dataset_mode,
            "data_path": data_path,
            "timestamp": datetime.now().isoformat(),
            "operations": [],
        }

        print(f"Initialized BigDataBenchmark ({dataset_mode} mode)")
        print(f"Data path: {data_path}")
        print(f"Output path: {output_path}")

    def load_data(self):
        """Load transactions data"""
        print("\n" + "=" * 60)
        print("Loading data...")
        print("=" * 60)

        start_time = time.time()

        # Load CSV data
        self.df = self.spark.read.csv(self.data_path, header=True, inferSchema=True)

        # For large dataset mode, sample 50% of rows to reduce processing time
        if self.dataset_mode == "large":
            print("Large dataset mode: sampling 50% of rows for faster processing...")
            self.df = self.df.sample(fraction=0.5, seed=42)

        # Cache the data for better performance
        self.df.cache()

        # Get initial count
        total_rows = self.df.count()
        load_time = time.time() - start_time

        print(f"Loaded {total_rows:,} rows in {load_time:.2f} seconds")
        if self.dataset_mode == "large":
            print("(50% sample of original large dataset)")

        self.results["total_rows"] = total_rows
        self.results["data_load_time_seconds"] = round(load_time, 3)
        self.results["sampled"] = self.dataset_mode == "large"
        self.results["sample_fraction"] = 0.5 if self.dataset_mode == "large" else 1.0

        # Show schema
        print("\nDataset Schema:")
        self.df.printSchema()

        return total_rows

    def run_operation(self, name, description, operation_func):
        """
        Run a single operation and collect metrics

        Args:
            name: Operation name
            description: Human-readable description
            operation_func: Function that performs the operation

        Returns:
            Result dictionary
        """
        print(f"\n{'-' * 60}")
        print(f"Running: {description}")
        print(f"{'-' * 60}")

        start_time = time.time()

        try:
            result_df = operation_func()

            # Force evaluation by counting results
            result_count = result_df.count()

            duration = time.time() - start_time

            result = {
                "name": name,
                "description": description,
                "duration_seconds": round(duration, 3),
                "result_rows": result_count,
                "success": True,
            }

            print(f"✓ Completed in {duration:.2f}s - {result_count:,} result rows")

            # Show sample results
            print("\nSample results:")
            result_df.show(5, truncate=False)

        except Exception as e:
            duration = time.time() - start_time
            result = {
                "name": name,
                "description": description,
                "duration_seconds": round(duration, 3),
                "error": str(e),
                "success": False,
            }
            print(f"✗ Failed after {duration:.2f}s: {str(e)}")

        self.results["operations"].append(result)
        return result

    def benchmark_filtering(self):
        """Test filtering operations (reduced to 2 tests)"""
        print("\n" + "=" * 60)
        print("BENCHMARK: Filtering & Transformations")
        print("=" * 60)

        # Test 1: Filter by expense type
        self.run_operation(
            "filter_expense_type",
            "Filter Housing and Entertainment expenses",
            lambda: self.df.filter(col("EXP_TYPE").isin(["Housing", "Entertainment"])),
        )

        # Test 2: Complex multi-condition filter
        self.run_operation(
            "filter_complex",
            "Complex filter: Housing > $300 in 2018-2019",
            lambda: self.df.filter(
                (col("EXP_TYPE") == "Housing")
                & (col("AMOUNT") > 300)
                & (col("YEAR").isin([2018, 2019]))
            ),
        )

    def benchmark_aggregations(self):
        """Test aggregation operations (reduced to 3 tests)"""
        print("\n" + "=" * 60)
        print("BENCHMARK: Aggregations")
        print("=" * 60)

        # Test 1: Total spending by customer
        self.run_operation(
            "agg_by_customer",
            "Total spending per customer (GROUP BY CUST_ID)",
            lambda: self.df.groupBy("CUST_ID").agg(
                spark_sum("AMOUNT").alias("total_spent"),
                count("*").alias("transaction_count"),
                avg("AMOUNT").alias("avg_transaction"),
            ),
        )

        # Test 2: Total spending by expense type
        self.run_operation(
            "agg_by_expense_type",
            "Total spending per expense type (GROUP BY EXP_TYPE)",
            lambda: self.df.groupBy("EXP_TYPE").agg(
                spark_sum("AMOUNT").alias("total_amount"),
                count("*").alias("count"),
                avg("AMOUNT").alias("avg_amount"),
            ),
        )

        # Test 3: Monthly spending trends
        self.run_operation(
            "agg_monthly_trends",
            "Monthly spending trends (GROUP BY YEAR, MONTH)",
            lambda: self.df.groupBy("YEAR", "MONTH")
            .agg(
                spark_sum("AMOUNT").alias("monthly_total"),
                count("*").alias("monthly_count"),
            )
            .orderBy("YEAR", "MONTH"),
        )

    def benchmark_sorting(self):
        """Test sorting operations (reduced to 1 test)"""
        print("\n" + "=" * 60)
        print("BENCHMARK: Sorting")
        print("=" * 60)

        # Test 1: Sort by amount DESC (top transactions)
        self.run_operation(
            "sort_by_amount_desc",
            "Sort by amount DESC - find top transactions",
            lambda: self.df.orderBy(desc("AMOUNT")),
        )

    def benchmark_joins(self):
        """Test join operations (reduced to 1 test)"""
        print("\n" + "=" * 60)
        print("BENCHMARK: Joins")
        print("=" * 60)

        # Prepare data for joins
        # Create customer summary
        customer_summary = self.df.groupBy("CUST_ID").agg(
            spark_sum("AMOUNT").alias("total_spent"),
            count("*").alias("total_transactions"),
            avg("AMOUNT").alias("avg_transaction_amount"),
        )

        # Test 1: Join transactions with customer summary
        self.run_operation(
            "join_customer_summary",
            "Join transactions with customer summary (enrichment)",
            lambda: self.df.join(customer_summary, on="CUST_ID", how="inner"),
        )

    def run_full_benchmark(self):
        """Run complete benchmark suite"""
        print("\n" + "=" * 60)
        print(f"Starting Big Data Benchmark ({self.dataset_mode} mode)")
        print("=" * 60)

        overall_start = time.time()

        # Load data
        total_rows = self.load_data()

        # Run all benchmark categories
        self.benchmark_filtering()
        self.benchmark_aggregations()
        self.benchmark_sorting()
        self.benchmark_joins()

        # Calculate overall metrics
        overall_duration = time.time() - overall_start
        self.results["total_duration_seconds"] = round(overall_duration, 3)

        # Calculate success rate
        total_ops = len(self.results["operations"])
        successful_ops = sum(1 for op in self.results["operations"] if op["success"])
        self.results["success_rate"] = (
            round(successful_ops / total_ops * 100, 2) if total_ops > 0 else 0
        )

        print("\n" + "=" * 60)
        print("Benchmark Complete!")
        print("=" * 60)
        print(f"Total operations: {total_ops}")
        print(f"Successful: {successful_ops}")
        print(f"Failed: {total_ops - successful_ops}")
        print(f"Success rate: {self.results['success_rate']}%")
        print(f"Total duration: {overall_duration:.2f} seconds")

        return self.results

    def save_results(self):
        """Save results to output path"""
        results_json = json.dumps(self.results, indent=2)

        # Save to cloud storage
        output_file = f"{self.output_path}/benchmark_results.json"

        # Write using Spark
        sc = self.spark.sparkContext
        rdd = sc.parallelize([results_json])
        rdd.saveAsTextFile(output_file)

        print(f"\n✓ Results saved to: {output_file}")

        return output_file

    def stop(self):
        """Stop Spark session"""
        self.spark.stop()


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Big Data Benchmark for PySpark")
    parser.add_argument(
        "--data-path", required=True, help="Path to input data (S3 or GCS)"
    )
    parser.add_argument("--output-path", required=True, help="Path for output results")
    parser.add_argument(
        "--dataset-mode",
        choices=["sample", "large", "full"],
        default="sample",
        help="Dataset mode: sample (1M rows), large (131M rows), or full (262M rows)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("PySpark Big Data Benchmark")
    print("=" * 60)
    print(f"Data path: {args.data_path}")
    print(f"Output path: {args.output_path}")
    print(f"Dataset mode: {args.dataset_mode}")
    print("=" * 60)

    try:
        # Run benchmark
        benchmark = BigDataBenchmark(
            data_path=args.data_path,
            output_path=args.output_path,
            dataset_mode=args.dataset_mode,
        )

        results = benchmark.run_full_benchmark()

        # Save results
        benchmark.save_results()

        # Print summary
        print("\n" + "=" * 60)
        print("Summary Statistics")
        print("=" * 60)
        print(f"Dataset: {args.dataset_mode}")
        print(f"Total rows processed: {results['total_rows']:,}")
        print(f"Total operations: {len(results['operations'])}")
        print(f"Success rate: {results['success_rate']}%")
        print(f"Total time: {results['total_duration_seconds']:.2f}s")

        # Print operation timings
        print("\nOperation Timings:")
        print("-" * 60)
        for op in results["operations"]:
            status = "✓" if op["success"] else "✗"
            print(f"{status} {op['name']:30} {op['duration_seconds']:6.2f}s")

        benchmark.stop()

        print("\n✓ Benchmark completed successfully!")

    except Exception as e:
        print(f"\n✗ Benchmark failed: {str(e)}")
        import traceback

        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
