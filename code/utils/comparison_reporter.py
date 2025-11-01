import json
import csv
import os
from datetime import datetime
from typing import Dict, Any, Optional


def calculate_percentage_difference(aws_value: float, gcp_value: float) -> float:
    """
    Calculate percentage difference between AWS and GCP values.

    Positive value means GCP is higher, negative means AWS is higher.

    Args:
        aws_value: AWS metric value
        gcp_value: GCP metric value

    Returns:
        Percentage difference
    """
    if aws_value == 0 and gcp_value == 0:
        return 0.0
    if aws_value == 0:
        return 100.0

    return ((gcp_value - aws_value) / aws_value) * 100


def extract_benchmark_metrics(
    provider_data: Dict[str, Any], benchmark_type: str
) -> Optional[Dict[str, Any]]:
    """
    Extract metrics for a specific benchmark type from provider data.

    Args:
        provider_data: Provider's complete benchmark data
        benchmark_type: Type of benchmark (ml, storage, pyspark_sample, pyspark_large)

    Returns:
        Benchmark metrics dictionary or None if not found
    """
    if "benchmarks" not in provider_data:
        return None
    return provider_data["benchmarks"].get(benchmark_type)


def generate_comparison_table(results: Dict[str, Any]) -> list:
    """
    Generate comparison table data for AWS vs GCP.

    Args:
        results: Complete comparison results

    Returns:
        List of comparison rows
    """
    table_rows = []

    if "providers" not in results:
        return table_rows

    aws_data = results["providers"].get("aws", {})
    gcp_data = results["providers"].get("gcp", {})

    # Benchmark types to compare
    benchmarks = [
        ("ml", "ML Training"),
        ("storage", "Storage Operations"),
        ("pyspark_sample", "PySpark Sample (1M rows)"),
        ("pyspark_large", "PySpark Large (131M rows)"),
    ]

    for bench_key, bench_name in benchmarks:
        aws_metrics = extract_benchmark_metrics(aws_data, bench_key)
        gcp_metrics = extract_benchmark_metrics(gcp_data, bench_key)

        if not aws_metrics or not gcp_metrics:
            continue

        # Extract timing metrics
        if "summary" in aws_metrics and "summary" in gcp_metrics:
            # Total time comparison
            if (
                "total_time" in aws_metrics["summary"]
                and "total_time" in gcp_metrics["summary"]
            ):
                aws_time = aws_metrics["summary"]["total_time"]["mean"]
                gcp_time = gcp_metrics["summary"]["total_time"]["mean"]
                diff_pct = calculate_percentage_difference(aws_time, gcp_time)

                table_rows.append(
                    {
                        "benchmark": bench_name,
                        "metric": "Total Time (seconds)",
                        "aws_mean": aws_time,
                        "aws_std": aws_metrics["summary"]["total_time"]["std"],
                        "gcp_mean": gcp_time,
                        "gcp_std": gcp_metrics["summary"]["total_time"]["std"],
                        "difference_pct": diff_pct,
                        "winner": "AWS"
                        if diff_pct > 0
                        else "GCP"
                        if diff_pct < 0
                        else "Tie",
                    }
                )

            # Total cost comparison
            if (
                "total_cost" in aws_metrics["summary"]
                and "total_cost" in gcp_metrics["summary"]
            ):
                aws_cost = aws_metrics["summary"]["total_cost"]["mean"]
                gcp_cost = gcp_metrics["summary"]["total_cost"]["mean"]
                diff_pct = calculate_percentage_difference(aws_cost, gcp_cost)

                table_rows.append(
                    {
                        "benchmark": bench_name,
                        "metric": "Total Cost (USD)",
                        "aws_mean": aws_cost,
                        "aws_std": aws_metrics["summary"]["total_cost"]["std"],
                        "gcp_mean": gcp_cost,
                        "gcp_std": gcp_metrics["summary"]["total_cost"]["std"],
                        "difference_pct": diff_pct,
                        "winner": "AWS"
                        if diff_pct > 0
                        else "GCP"
                        if diff_pct < 0
                        else "Tie",
                    }
                )

    return table_rows


def save_comparison_csv(table_rows: list, output_dir: str, timestamp: str) -> str:
    """
    Save comparison table as CSV.

    Args:
        table_rows: List of comparison row dictionaries
        output_dir: Output directory
        timestamp: Timestamp string for filename

    Returns:
        Path to saved CSV file
    """
    if not table_rows:
        return None

    filename = f"comparison_summary_{timestamp}.csv"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w", newline="") as f:
        fieldnames = [
            "benchmark",
            "metric",
            "aws_mean",
            "aws_std",
            "gcp_mean",
            "gcp_std",
            "difference_pct",
            "winner",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(table_rows)

    return filepath


def save_comparison_markdown(
    results: Dict[str, Any], table_rows: list, output_dir: str, timestamp: str
) -> str:
    """
    Save comparison as Markdown report.

    Args:
        results: Complete comparison results
        table_rows: List of comparison row dictionaries
        output_dir: Output directory
        timestamp: Timestamp string for filename

    Returns:
        Path to saved Markdown file
    """
    filename = f"comparison_report_{timestamp}.md"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w") as f:
        # Header
        f.write("# AWS vs GCP Cloud Platform Comparison\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Configuration
        if "configuration" in results:
            config = results["configuration"]
            f.write("## Benchmark Configuration\n\n")
            f.write(f"- ML Runs: {config.get('ml_runs', 'N/A')}\n")
            f.write(f"- Storage Runs: {config.get('storage_runs', 'N/A')}\n")
            f.write(
                f"- PySpark Sample Runs: {config.get('pyspark_sample_runs', 'N/A')} (1M rows)\n"
            )
            f.write(
                f"- PySpark Large Runs: {config.get('pyspark_large_runs', 'N/A')} (131M rows ~10GB)\n\n"
            )

        # Duration
        if "total_duration_hours" in results:
            f.write(
                f"**Total Benchmark Duration:** {results['total_duration_hours']:.2f} hours\n\n"
            )

        # Comparison Table
        f.write("## Performance Comparison\n\n")
        f.write(
            "| Benchmark | Metric | AWS (Mean ± Std) | GCP (Mean ± Std) | Difference | Winner |\n"
        )
        f.write(
            "|-----------|--------|------------------|------------------|------------|--------|\n"
        )

        for row in table_rows:
            benchmark = row["benchmark"]
            metric = row["metric"]

            if "Cost" in metric:
                aws_val = f"${row['aws_mean']:.4f} ± ${row['aws_std']:.4f}"
                gcp_val = f"${row['gcp_mean']:.4f} ± ${row['gcp_std']:.4f}"
            else:
                aws_val = f"{row['aws_mean']:.2f} ± {row['aws_std']:.2f}s"
                gcp_val = f"{row['gcp_mean']:.2f} ± {row['gcp_std']:.2f}s"

            diff = f"{row['difference_pct']:+.1f}%"
            winner = row["winner"]

            f.write(
                f"| {benchmark} | {metric} | {aws_val} | {gcp_val} | {diff} | **{winner}** |\n"
            )

        # Summary
        f.write("\n## Summary\n\n")

        # Count wins
        aws_wins = sum(1 for row in table_rows if row["winner"] == "AWS")
        gcp_wins = sum(1 for row in table_rows if row["winner"] == "GCP")
        ties = sum(1 for row in table_rows if row["winner"] == "Tie")

        f.write(f"- AWS Wins: {aws_wins}\n")
        f.write(f"- GCP Wins: {gcp_wins}\n")
        f.write(f"- Ties: {ties}\n\n")

        # Notes
        f.write("## Notes\n\n")
        f.write(
            "- **Difference %**: Positive values indicate GCP is higher/slower/more expensive, "
        )
        f.write("negative values indicate AWS is higher/slower/more expensive.\n")
        f.write(
            "- **Winner**: For time metrics, lower is better. For cost metrics, lower is better.\n"
        )
        f.write(
            "- All values represent mean ± standard deviation across multiple runs.\n\n"
        )

        f.write("---\n")
        f.write("*Generated using AWS vs GCP Benchmark Comparison Tool*\n")

    return filepath


def generate_comparison_report(results: Dict[str, Any], output_dir: str) -> str:
    """
    Generate comprehensive comparison report.

    Args:
        results: Complete comparison results dictionary
        output_dir: Directory to save reports

    Returns:
        Path to main report file
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Generate comparison table
    table_rows = generate_comparison_table(results)

    if not table_rows:
        print("Warning: No comparison data available to generate report")
        return None

    # Save CSV
    csv_file = save_comparison_csv(table_rows, output_dir, timestamp)
    if csv_file:
        print(f"Comparison CSV saved to: {csv_file}")

    # Save Markdown report
    md_file = save_comparison_markdown(results, table_rows, output_dir, timestamp)
    if md_file:
        print(f"Comparison report saved to: {md_file}")

    return md_file


def main():
    """
    Main entry point for standalone report generation.

    Loads a comparison JSON file and generates reports.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate comparison report from results JSON"
    )
    parser.add_argument("results_file", help="Path to complete comparison JSON file")
    parser.add_argument(
        "--output-dir", default=".", help="Output directory for reports"
    )

    args = parser.parse_args()

    with open(args.results_file, "r") as f:
        results = json.load(f)

    report_file = generate_comparison_report(results, args.output_dir)

    if report_file:
        print(f"\n✓ Comparison report generated successfully!")
        print(f"Report location: {report_file}")
    else:
        print("\n✗ Failed to generate comparison report")


if __name__ == "__main__":
    main()
