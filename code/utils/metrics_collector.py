import time
import json
import os
from datetime import datetime
from typing import Dict, Optional, Any


class MetricsCollector:
    """
    Collects and stores performance metrics for cloud experiments.

    Tracks:
    - Cluster provisioning time
    - ML training execution time
    - Total end-to-end time
    - Cost estimates
    - Additional metadata
    """

    def __init__(self, provider: str, experiment_id: Optional[str] = None):
        """
        Initialize metrics collector.

        Args:
            provider: Cloud provider name ('aws' or 'gcp')
            experiment_id: Unique identifier for this experiment run
        """
        self.provider = provider.lower()
        self.experiment_id = experiment_id or self._generate_experiment_id()
        self.metrics = {
            "experiment_id": self.experiment_id,
            "provider": self.provider,
            "timestamp": datetime.utcnow().isoformat(),
            "timings": {},
            "costs": {},
            "metadata": {},
            "errors": [],
        }
        self._timers = {}

    def _generate_experiment_id(self) -> str:
        """Generate a unique experiment ID based on provider and timestamp."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        return f"{self.provider}_{timestamp}"

    def start_timer(self, metric_name: str) -> None:
        """
        Start a timer for a specific metric.

        Args:
            metric_name: Name of the metric to track (e.g., 'cluster_provisioning')
        """
        self._timers[metric_name] = time.time()
        print(f"[{self.provider.upper()}] Started timer: {metric_name}")

    def stop_timer(self, metric_name: str) -> float:
        """
        Stop a timer and record the elapsed time.

        Args:
            metric_name: Name of the metric being tracked

        Returns:
            Elapsed time in seconds

        Raises:
            ValueError: If timer was not started
        """
        if metric_name not in self._timers:
            raise ValueError(f"Timer '{metric_name}' was not started")

        elapsed = time.time() - self._timers[metric_name]
        self.metrics["timings"][metric_name] = elapsed

        # Convert to human-readable format
        if elapsed < 60:
            time_str = f"{elapsed:.2f} seconds"
        elif elapsed < 3600:
            time_str = f"{elapsed / 60:.2f} minutes"
        else:
            time_str = f"{elapsed / 3600:.2f} hours"

        print(f"[{self.provider.upper()}] Stopped timer: {metric_name} - {time_str}")

        del self._timers[metric_name]
        return elapsed

    def record_metric(
        self, metric_name: str, value: float, category: str = "timings"
    ) -> None:
        """
        Record a metric value directly without using timers.

        Args:
            metric_name: Name of the metric
            value: Metric value
            category: Category for the metric (default: 'timings')
        """
        if category not in self.metrics:
            self.metrics[category] = {}

        self.metrics[category][metric_name] = value
        print(f"[{self.provider.upper()}] Recorded {category}.{metric_name}: {value}")

    def record_cost(self, cost_component: str, amount: float) -> None:
        """
        Record a cost component.

        Args:
            cost_component: Name of the cost component (e.g., 'compute', 'storage')
            amount: Cost amount in USD
        """
        self.metrics["costs"][cost_component] = amount
        print(
            f"[{self.provider.upper()}] Recorded cost: {cost_component} = ${amount:.4f}"
        )

    def add_metadata(self, key: str, value: Any) -> None:
        """
        Add metadata about the experiment.

        Args:
            key: Metadata key
            value: Metadata value
        """
        self.metrics["metadata"][key] = value

    def record_error(
        self, error_message: str, error_type: Optional[str] = None
    ) -> None:
        """
        Record an error that occurred during the experiment.

        Args:
            error_message: Description of the error
            error_type: Type/category of error (optional)
        """
        error_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "message": error_message,
            "type": error_type,
        }
        self.metrics["errors"].append(error_entry)
        print(f"[{self.provider.upper()}] Error recorded: {error_message}")

    def get_total_time(self) -> Optional[float]:
        """
        Get the total end-to-end time if available.

        Returns:
            Total time in seconds, or None if not recorded
        """
        return self.metrics["timings"].get("total_time")

    def get_total_cost(self) -> float:
        """
        Get the total cost across all components.

        Returns:
            Total cost in USD
        """
        return sum(self.metrics["costs"].values())

    def save_to_file(self, output_dir: str, filename: Optional[str] = None) -> str:
        """
        Save metrics to a JSON file.

        Args:
            output_dir: Directory to save the metrics file
            filename: Custom filename (optional, defaults to experiment_id.json)

        Returns:
            Path to the saved file
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Generate filename
        if filename is None:
            filename = f"{self.experiment_id}.json"

        filepath = os.path.join(output_dir, filename)

        # Calculate summary metrics
        self.metrics["summary"] = {
            "total_time_seconds": self.get_total_time(),
            "total_cost_usd": self.get_total_cost(),
            "success": len(self.metrics["errors"]) == 0,
        }

        # Save to file
        with open(filepath, "w") as f:
            json.dump(self.metrics, f, indent=2)

        print(f"[{self.provider.upper()}] Metrics saved to: {filepath}")
        return filepath

    def print_summary(self) -> None:
        """Print a summary of collected metrics to console."""
        print(f"\n{'=' * 60}")
        print(f"Metrics Summary - {self.provider.upper()}")
        print(f"Experiment ID: {self.experiment_id}")
        print(f"{'=' * 60}")

        if self.metrics["timings"]:
            print("\nTimings:")
            for metric, value in self.metrics["timings"].items():
                if value < 60:
                    print(f"  {metric}: {value:.2f} seconds")
                elif value < 3600:
                    print(f"  {metric}: {value / 60:.2f} minutes ({value:.2f}s)")
                else:
                    print(f"  {metric}: {value / 3600:.2f} hours ({value:.2f}s)")

        if self.metrics["costs"]:
            print("\nCosts:")
            for component, cost in self.metrics["costs"].items():
                print(f"  {component}: ${cost:.4f}")
            print(f"  TOTAL: ${self.get_total_cost():.4f}")

        if self.metrics["metadata"]:
            print("\nMetadata:")
            for key, value in self.metrics["metadata"].items():
                print(f"  {key}: {value}")

        if self.metrics["errors"]:
            print("\nErrors:")
            for error in self.metrics["errors"]:
                print(f"  [{error['timestamp']}] {error['message']}")

        print(f"\n{'=' * 60}\n")

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get the complete metrics dictionary.

        Returns:
            Dictionary containing all collected metrics
        """
        return self.metrics.copy()
