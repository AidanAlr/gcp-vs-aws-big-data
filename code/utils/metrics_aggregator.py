import json
import os
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional


class MetricsAggregator:
    """
    Aggregates metrics from multiple benchmark runs.

    Computes statistical summaries including mean, standard deviation,
    minimum, and maximum values across all timing and cost metrics.
    """

    def __init__(
        self, benchmark_type: str, provider: str, dataset_mode: Optional[str] = None
    ):
        """
        Initialize metrics aggregator.

        Args:
            benchmark_type: Type of benchmark (ml, storage, pyspark)
            provider: Cloud provider (aws or gcp)
            dataset_mode: Dataset mode for pyspark (sample, large, full)
        """
        self.benchmark_type = benchmark_type
        self.provider = provider.lower()
        self.dataset_mode = dataset_mode
        self.runs = []
        self.aggregated_metrics = None

    def add_run(self, metrics: Dict[str, Any]) -> None:
        """
        Add a single run's metrics to the aggregator.

        Args:
            metrics: Metrics dictionary from a single run
        """
        self.runs.append(metrics)

    def add_run_from_file(self, filepath: str) -> None:
        """
        Load and add metrics from a JSON file.

        Args:
            filepath: Path to metrics JSON file
        """
        with open(filepath, "r") as f:
            metrics = json.load(f)
        self.add_run(metrics)

    def _compute_stats(self, values: List[float]) -> Dict[str, float]:
        """
        Compute statistical summary for a list of values.

        Args:
            values: List of numeric values

        Returns:
            Dictionary with mean, std, min, max
        """
        if not values:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

        values_array = np.array(values)
        return {
            "mean": float(np.mean(values_array)),
            "std": float(np.std(values_array)),
            "min": float(np.min(values_array)),
            "max": float(np.max(values_array)),
            "count": len(values),
        }

    def aggregate(self) -> Dict[str, Any]:
        """
        Compute aggregated statistics across all runs.

        Returns:
            Dictionary containing aggregated metrics and statistics
        """
        if not self.runs:
            # Return empty aggregated metrics if no runs (e.g., when runs=0)
            self.aggregated_metrics = {
                "benchmark_type": self.benchmark_type,
                "provider": self.provider,
                "dataset_mode": self.dataset_mode,
                "num_runs": 0,
                "timestamp": datetime.utcnow().isoformat(),
                "timings": {},
                "costs": {},
                "summary": {"successful_runs": 0, "failed_runs": 0},
            }
            return self.aggregated_metrics

        # Initialize aggregation structure
        self.aggregated_metrics = {
            "benchmark_type": self.benchmark_type,
            "provider": self.provider,
            "dataset_mode": self.dataset_mode,
            "num_runs": len(self.runs),
            "timestamp": datetime.utcnow().isoformat(),
            "timings": {},
            "costs": {},
            "summary": {},
        }

        # Aggregate timing metrics
        timing_keys = set()
        for run in self.runs:
            if "timings" in run:
                timing_keys.update(run["timings"].keys())

        for key in timing_keys:
            values = [
                run["timings"].get(key, 0) for run in self.runs if "timings" in run
            ]
            self.aggregated_metrics["timings"][key] = self._compute_stats(values)

        # Aggregate cost metrics
        cost_keys = set()
        for run in self.runs:
            if "costs" in run:
                cost_keys.update(run["costs"].keys())

        for key in cost_keys:
            values = [run["costs"].get(key, 0) for run in self.runs if "costs" in run]
            self.aggregated_metrics["costs"][key] = self._compute_stats(values)

        # Compute total cost statistics if available
        total_costs = []
        for run in self.runs:
            if "costs" in run:
                total = sum(run["costs"].values())
                total_costs.append(total)

        if total_costs:
            self.aggregated_metrics["costs"]["total"] = self._compute_stats(total_costs)

        # Create summary with key metrics
        if "total_time" in self.aggregated_metrics["timings"]:
            self.aggregated_metrics["summary"]["total_time"] = self.aggregated_metrics[
                "timings"
            ]["total_time"]

        if "total" in self.aggregated_metrics["costs"]:
            self.aggregated_metrics["summary"]["total_cost"] = self.aggregated_metrics[
                "costs"
            ]["total"]

        # Count successful runs (no errors)
        successful_runs = sum(1 for run in self.runs if not run.get("errors", []))
        self.aggregated_metrics["summary"]["successful_runs"] = successful_runs
        self.aggregated_metrics["summary"]["failed_runs"] = (
            len(self.runs) - successful_runs
        )

        return self.aggregated_metrics

    def save_to_file(self, output_dir: str, filename: Optional[str] = None) -> str:
        """
        Save aggregated metrics to a JSON file.

        Args:
            output_dir: Directory to save the metrics file
            filename: Custom filename (optional)

        Returns:
            Path to the saved file
        """
        if self.aggregated_metrics is None:
            raise ValueError("Must call aggregate() before saving")

        os.makedirs(output_dir, exist_ok=True)

        if filename is None:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            mode_suffix = f"_{self.dataset_mode}" if self.dataset_mode else ""
            filename = f"{self.benchmark_type}_{self.provider}{mode_suffix}_aggregated_{timestamp}.json"

        filepath = os.path.join(output_dir, filename)

        with open(filepath, "w") as f:
            json.dump(self.aggregated_metrics, f, indent=2)

        print(f"[{self.provider.upper()}] Aggregated metrics saved to: {filepath}")
        return filepath

    def print_summary(self) -> None:
        """Print a summary of aggregated metrics to console."""
        if self.aggregated_metrics is None:
            raise ValueError("Must call aggregate() before printing summary")

        print(f"\n{'=' * 70}")
        print(f"Aggregated Metrics Summary - {self.provider.upper()}")
        print(f"Benchmark Type: {self.benchmark_type}")
        if self.dataset_mode:
            print(f"Dataset Mode: {self.dataset_mode}")
        print(f"Number of Runs: {self.aggregated_metrics['num_runs']}")
        print(f"{'=' * 70}")

        # Print timing statistics
        if self.aggregated_metrics["timings"]:
            print("\nTiming Metrics (seconds):")
            print(
                f"{'Metric':<25} {'Mean':>10} {'Std Dev':>10} {'Min':>10} {'Max':>10}"
            )
            print("-" * 70)
            for metric, stats in self.aggregated_metrics["timings"].items():
                print(
                    f"{metric:<25} {stats['mean']:>10.2f} {stats['std']:>10.2f} "
                    f"{stats['min']:>10.2f} {stats['max']:>10.2f}"
                )

        # Print cost statistics
        if self.aggregated_metrics["costs"]:
            print("\nCost Metrics (USD):")
            print(
                f"{'Component':<25} {'Mean':>10} {'Std Dev':>10} {'Min':>10} {'Max':>10}"
            )
            print("-" * 70)
            for component, stats in self.aggregated_metrics["costs"].items():
                print(
                    f"{component:<25} ${stats['mean']:>9.4f} ${stats['std']:>9.4f} "
                    f"${stats['min']:>9.4f} ${stats['max']:>9.4f}"
                )

        # Print summary
        print(f"\n{'=' * 70}")
        print(f"Summary:")
        print(
            f"  Successful Runs: {self.aggregated_metrics['summary']['successful_runs']}"
        )
        print(f"  Failed Runs: {self.aggregated_metrics['summary']['failed_runs']}")
        if "total_time" in self.aggregated_metrics["summary"]:
            tt = self.aggregated_metrics["summary"]["total_time"]
            print(f"  Avg Total Time: {tt['mean']:.2f}s ± {tt['std']:.2f}s")
        if "total_cost" in self.aggregated_metrics["summary"]:
            tc = self.aggregated_metrics["summary"]["total_cost"]
            print(f"  Avg Total Cost: ${tc['mean']:.4f} ± ${tc['std']:.4f}")
        print(f"{'=' * 70}\n")

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get the aggregated metrics dictionary.

        Returns:
            Dictionary containing all aggregated metrics
        """
        if self.aggregated_metrics is None:
            raise ValueError("Must call aggregate() before getting metrics")
        return self.aggregated_metrics.copy()
