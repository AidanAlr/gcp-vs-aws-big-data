from typing import Dict
import sys
import os

from config import AWS_CONFIG, AWS_PRICING, GCP_CONFIG, GCP_PRICING

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class CostCalculator:
    """
    Calculate estimated costs for AWS EMR and GCP Dataproc experiments.

    Costs include:
    - Compute instance costs
    - Platform fees (EMR/Dataproc)
    - Storage costs
    - Data transfer (minimal for these experiments)
    """

    def __init__(self, provider: str):
        """
        Initialize cost calculator for a specific provider.

        Args:
            provider: Cloud provider ('aws' or 'gcp')
        """
        self.provider = provider.lower()

        if self.provider == "aws":
            self.config = AWS_CONFIG
            self.pricing = AWS_PRICING
        elif self.provider == "gcp":
            self.config = GCP_CONFIG
            self.pricing = GCP_PRICING
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def calculate_cluster_cost(self, runtime_hours: float) -> Dict[str, float]:
        """
        Calculate the total cost for running a cluster for a given duration.

        Args:
            runtime_hours: Total cluster runtime in hours

        Returns:
            Dictionary with cost breakdown by component
        """
        if self.provider == "aws":
            return self._calculate_aws_cluster_cost(runtime_hours)
        else:
            return self._calculate_gcp_cluster_cost(runtime_hours)

    def _calculate_aws_cluster_cost(self, runtime_hours: float) -> Dict[str, float]:
        """Calculate AWS EMR cluster costs."""
        costs = {}

        # Master instance cost
        master_ec2_cost = (
            self.pricing[self.config["master_instance_type"]] * runtime_hours
        )
        master_emr_cost = self.pricing["emr_fee_per_instance"] * runtime_hours
        costs["master_ec2"] = master_ec2_cost
        costs["master_emr_fee"] = master_emr_cost

        # Core instances cost
        core_instance_type = self.config["core_instance_type"]
        core_count = self.config["core_instance_count"]

        core_ec2_cost = self.pricing[core_instance_type] * runtime_hours * core_count
        core_emr_cost = (
            self.pricing["emr_fee_per_instance"] * runtime_hours * core_count
        )

        costs["core_ec2"] = core_ec2_cost
        costs["core_emr_fee"] = core_emr_cost

        # Total compute
        costs["total_compute"] = (
            master_ec2_cost + master_emr_cost + core_ec2_cost + core_emr_cost
        )

        return costs

    def _calculate_gcp_cluster_cost(self, runtime_hours: float) -> Dict[str, float]:
        """Calculate GCP Dataproc cluster costs."""
        costs = {}
        # Master instance cost
        master_vm_cost = (
            self.pricing[self.config["master_machine_type"]] * runtime_hours
        )
        # Master is n1-standard-4 = 4 vCPUs
        master_dataproc_cost = self.pricing["dataproc_fee_per_vcpu"] * 4 * runtime_hours
        costs["master_vm"] = master_vm_cost
        costs["master_dataproc_fee"] = master_dataproc_cost

        # Worker instances cost
        worker_machine_type = self.config["worker_machine_type"]
        worker_count = self.config["worker_count"]

        worker_vm_cost = (
            self.pricing[worker_machine_type] * runtime_hours * worker_count
        )
        # n1-standard-4 = 4 vCPUs per worker
        worker_dataproc_cost = (
            self.pricing["dataproc_fee_per_vcpu"] * 4 * runtime_hours * worker_count
        )
        costs["worker_vm"] = worker_vm_cost
        costs["worker_dataproc_fee"] = worker_dataproc_cost

        # Total compute
        costs["total_compute"] = (
            master_vm_cost
            + master_dataproc_cost
            + worker_vm_cost
            + worker_dataproc_cost
        )

        return costs

    def calculate_storage_cost(
        self, data_size_gb: float, duration_hours: float
    ) -> float:
        """
        Calculate storage costs for the experiment.

        Args:
            data_size_gb: Size of data stored in GB
            duration_hours: How long data is stored in hours

        Returns:
            Storage cost in USD
        """
        # Convert hours to months (assuming 730 hours per month)
        duration_months = duration_hours / 730.0

        if self.provider == "aws":
            return self.pricing["s3_storage"] * data_size_gb * duration_months
        else:
            return self.pricing["gcs_storage"] * data_size_gb * duration_months

    def calculate_total_experiment_cost(
        self,
        cluster_runtime_seconds: float,
        data_size_gb: float = 0.5,
        storage_duration_hours: float = 24,
    ) -> Dict[str, float]:
        """
        Calculate the total cost for an entire experiment.

        Args:
            cluster_runtime_seconds: Cluster runtime in seconds
            data_size_gb: Data size in GB (default: 0.5 GB for Fashion-MNIST)
            storage_duration_hours: How long data is stored (default: 24 hours)

        Returns:
            Dictionary with complete cost breakdown
        """
        # Convert seconds to hours
        runtime_hours = cluster_runtime_seconds / 3600.0

        # Calculate cluster costs
        cluster_costs = self.calculate_cluster_cost(runtime_hours)

        # Calculate storage costs
        storage_cost = self.calculate_storage_cost(data_size_gb, storage_duration_hours)

        # Combine all costs
        total_costs = {
            **cluster_costs,
            "storage": storage_cost,
            "total": cluster_costs["total_compute"] + storage_cost,
        }

        return total_costs

    def get_hourly_rate(self) -> float:
        """
        Get the hourly rate for the configured cluster.

        Returns:
            Cost per hour in USD
        """
        costs = self.calculate_cluster_cost(runtime_hours=1.0)
        return costs["total_compute"]

    def print_cost_estimate(
        self, cluster_runtime_seconds: float, data_size_gb: float = 0.5
    ) -> None:
        """
        Print a formatted cost estimate for the experiment.

        Args:
            cluster_runtime_seconds: Cluster runtime in seconds
            data_size_gb: Data size in GB
        """
        costs = self.calculate_total_experiment_cost(
            cluster_runtime_seconds, data_size_gb
        )

        runtime_hours = cluster_runtime_seconds / 3600.0

        print(f"\n{'=' * 60}")
        print(f"Cost Estimate - {self.provider.upper()}")
        print(f"{'=' * 60}")
        print(
            f"Runtime: {runtime_hours:.4f} hours ({cluster_runtime_seconds:.2f} seconds)"
        )
        print(f"Data size: {data_size_gb:.2f} GB")
        print(f"\nCompute Costs:")

        if self.provider == "aws":
            print(f"  Master EC2:      ${costs['master_ec2']:.4f}")
            print(f"  Master EMR fee:  ${costs['master_emr_fee']:.4f}")
            print(f"  Core EC2:        ${costs['core_ec2']:.4f}")
            print(f"  Core EMR fee:    ${costs['core_emr_fee']:.4f}")
        else:
            print(f"  Master VM:       ${costs['master_vm']:.4f}")
            print(f"  Master DP fee:   ${costs['master_dataproc_fee']:.4f}")
            print(f"  Worker VM:       ${costs['worker_vm']:.4f}")
            print(f"  Worker DP fee:   ${costs['worker_dataproc_fee']:.4f}")

        print(f"\nStorage Costs:")
        print(f"  Data storage:    ${costs['storage']:.4f}")

        print(f"\n{'=' * 60}")
        print(f"TOTAL COST:        ${costs['total']:.4f}")
        print(f"Hourly rate:       ${self.get_hourly_rate():.4f}/hour")
        print(f"{'=' * 60}\n")
