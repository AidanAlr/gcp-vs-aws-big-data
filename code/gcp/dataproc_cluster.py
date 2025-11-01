from google.cloud import dataproc_v1
import time
from typing import Optional, Dict
import sys
import os

from config import GCP_CONFIG
from utils.logger_config import setup_logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logger = setup_logger("gcp_dataproc")


class DataprocCluster:
    """
    Manages GCP Dataproc cluster lifecycle.

    Provides methods to:
    - Create Dataproc clusters
    - Monitor cluster status
    - Wait for cluster to be ready
    - Delete clusters
    """

    def __init__(self, project_id: Optional[str] = None, region: Optional[str] = None):
        """
        Initialize Dataproc cluster manager.

        Args:
            project_id: GCP project ID (defaults to config value)
            region: GCP region (defaults to config value)
        """
        self.project_id = project_id or GCP_CONFIG["project_id"]
        self.region = region or GCP_CONFIG["region"]
        self.cluster_name = GCP_CONFIG["cluster_name"]

        # Initialize clients
        self.cluster_client = dataproc_v1.ClusterControllerClient(
            client_options={
                "api_endpoint": f"{self.region}-dataproc.googleapis.com:443"
            }
        )

    def cluster_exists(self, cluster_name: Optional[str] = None) -> bool:
        """
        Check if a cluster exists.

        Args:
            cluster_name: Cluster name (uses self.cluster_name if not provided)

        Returns:
            True if cluster exists, False otherwise
        """
        cname = cluster_name or self.cluster_name
        try:
            self.cluster_client.get_cluster(
                request={
                    "project_id": self.project_id,
                    "region": self.region,
                    "cluster_name": cname,
                }
            )
            return True
        except Exception:
            return False

    def create_cluster(self, force_new: bool = False) -> str:
        """
        Create a Dataproc cluster with the configured specifications.

        If cluster already exists and is running, it will be reused unless force_new=True.
        If cluster exists but is being deleted, waits for deletion to complete.

        Args:
            force_new: If True, delete existing cluster and create new one

        Returns:
            Cluster name

        Raises:
            Exception: If cluster creation fails
        """
        logger.info(f"Creating Dataproc cluster: {self.cluster_name}")

        # Check if cluster already exists
        if self.cluster_exists():
            status_info = self.get_cluster_status()
            state = status_info["state"]

            logger.info(f"Cluster already exists with state: {state}")

            if state == "RUNNING":
                if force_new:
                    logger.info("Force new cluster requested, deleting existing cluster...")
                    self.delete_cluster()
                    self.wait_for_cluster_deleted(timeout_minutes=15)
                else:
                    logger.info(f"Reusing existing cluster: {self.cluster_name}")
                    return self.cluster_name

            elif state == "DELETING":
                logger.info("Cluster is being deleted, waiting for deletion to complete...")
                self.wait_for_cluster_deleted(timeout_minutes=15)

            elif state == "ERROR":
                logger.warning(f"Existing cluster is in ERROR state, deleting it...")
                try:
                    self.delete_cluster()
                    self.wait_for_cluster_deleted(timeout_minutes=15)
                except Exception as e:
                    logger.error(f"Failed to delete ERROR cluster: {e}")
                    raise

            elif state == "CREATING":
                logger.info("Cluster is already being created, waiting for it to be ready...")
                if self.wait_for_cluster_ready(timeout_minutes=30):
                    return self.cluster_name
                else:
                    raise Exception("Cluster creation timed out")

        # Define cluster configuration
        cluster_config = {
            "project_id": self.project_id,
            "cluster_name": self.cluster_name,
            "config": {
                "master_config": {
                    "num_instances": 1,
                    "machine_type_uri": GCP_CONFIG["master_machine_type"],
                    "disk_config": {
                        "boot_disk_size_gb": 100,
                    },
                },
                "worker_config": {
                    "num_instances": GCP_CONFIG["worker_count"],
                    "machine_type_uri": GCP_CONFIG["worker_machine_type"],
                    "disk_config": {
                        "boot_disk_size_gb": 100,
                    },
                },
                "software_config": {
                    "image_version": GCP_CONFIG["image_version"],
                },
                "lifecycle_config": {
                    "idle_delete_ttl": {
                        "seconds": 3600
                    },  # Auto-delete after 1 hour idle
                },
                "gce_cluster_config": {
                    "metadata": {
                        "purpose": "ML-Comparison",
                        "project": "AWS-vs-GCP",
                    },
                },
            },
        }

        try:
            # Create the cluster
            operation = self.cluster_client.create_cluster(
                request={
                    "project_id": self.project_id,
                    "region": self.region,
                    "cluster": cluster_config,
                }
            )

            logger.info("Cluster creation initiated, waiting for completion...")
            # Wait for the operation to complete
            result = operation.result()

            logger.info(f"Cluster created successfully: {self.cluster_name}")
            return self.cluster_name

        except Exception as e:
            logger.error(f"Failed to create Dataproc cluster: {str(e)}")
            raise

    def get_cluster_status(self, cluster_name: Optional[str] = None) -> Dict:
        """
        Get the current status of a cluster.

        Args:
            cluster_name: Cluster name (uses self.cluster_name if not provided)

        Returns:
            Dictionary with cluster status information

        Raises:
            ValueError: If no cluster name is available
        """
        cname = cluster_name or self.cluster_name
        if not cname:
            raise ValueError("No cluster name provided or set")

        try:
            cluster = self.cluster_client.get_cluster(
                request={
                    "project_id": self.project_id,
                    "region": self.region,
                    "cluster_name": cname,
                }
            )

            status_info = {
                "cluster_name": cname,
                "state": cluster.status.state.name,
                "state_detail": cluster.status.detail,
                "state_start_time": cluster.status.state_start_time,
            }

            return status_info

        except Exception as e:
            logger.error(f"Failed to get cluster status: {str(e)}")
            raise

    def wait_for_cluster_ready(
        self,
        cluster_name: Optional[str] = None,
        timeout_minutes: int = 30,
        check_interval_seconds: int = 30,
    ) -> bool:
        """
        Wait for cluster to reach RUNNING state.

        Args:
            cluster_name: Cluster name (uses self.cluster_name if not provided)
            timeout_minutes: Maximum time to wait
            check_interval_seconds: How often to check status

        Returns:
            True if cluster is ready, False if timeout or failed

        Raises:
            ValueError: If no cluster name is available
        """
        cname = cluster_name or self.cluster_name
        if not cname:
            raise ValueError("No cluster name provided or set")

        logger.info(f"Waiting for cluster {cname} to be ready...")

        start_time = time.time()
        timeout_seconds = timeout_minutes * 60

        while True:
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > timeout_seconds:
                logger.error(
                    f"Timeout waiting for cluster after {timeout_minutes} minutes"
                )
                return False

            # Get cluster status
            status_info = self.get_cluster_status(cname)
            state = status_info["state"]

            logger.info(f"Cluster state: {state} (elapsed: {elapsed:.0f}s)")

            # Check if ready
            if state == "RUNNING":
                logger.info("Cluster is ready!")
                return True

            # Check for failure states
            if state in ["ERROR", "DELETING"]:
                detail = status_info.get("state_detail", "No details")
                logger.error(f"Cluster failed: {state}, detail: {detail}")
                return False

            # Wait before next check
            time.sleep(check_interval_seconds)

    def delete_cluster(self, cluster_name: Optional[str] = None) -> None:
        """
        Delete a Dataproc cluster.

        Args:
            cluster_name: Cluster name (uses self.cluster_name if not provided)

        Raises:
            ValueError: If no cluster name is available
        """
        cname = cluster_name or self.cluster_name
        if not cname:
            raise ValueError("No cluster name provided or set")

        logger.info(f"Deleting cluster: {cname}")

        try:
            operation = self.cluster_client.delete_cluster(
                request={
                    "project_id": self.project_id,
                    "region": self.region,
                    "cluster_name": cname,
                }
            )

            logger.info("Cluster deletion initiated")
            # Don't wait for deletion to complete in this method
            # Use wait_for_cluster_deleted if you need to wait

        except Exception as e:
            logger.error(f"Failed to delete cluster: {str(e)}")
            raise

    def wait_for_cluster_deleted(
        self,
        cluster_name: Optional[str] = None,
        timeout_minutes: int = 15,
        check_interval_seconds: int = 30,
    ) -> bool:
        """
        Wait for cluster to be fully deleted.

        Args:
            cluster_name: Cluster name (uses self.cluster_name if not provided)
            timeout_minutes: Maximum time to wait
            check_interval_seconds: How often to check status

        Returns:
            True if cluster is deleted, False if timeout
        """
        cname = cluster_name or self.cluster_name
        if not cname:
            raise ValueError("No cluster name provided or set")

        logger.info(f"Waiting for cluster {cname} to be deleted...")

        start_time = time.time()
        timeout_seconds = timeout_minutes * 60

        while True:
            elapsed = time.time() - start_time
            if elapsed > timeout_seconds:
                logger.warning(
                    f"Timeout waiting for deletion after {timeout_minutes} minutes"
                )
                return False

            try:
                status_info = self.get_cluster_status(cname)
                state = status_info["state"]
                logger.info(f"Cluster state: {state} (elapsed: {elapsed:.0f}s)")
            except Exception:
                # If we can't get the cluster, it's been deleted
                logger.info("Cluster deleted (no longer found)")
                return True

            time.sleep(check_interval_seconds)

    def list_clusters(self) -> list:
        """
        List all clusters in the project and region.

        Returns:
            List of cluster information dictionaries
        """
        try:
            request = {
                "project_id": self.project_id,
                "region": self.region,
            }

            clusters = []
            page_result = self.cluster_client.list_clusters(request=request)

            for cluster in page_result:
                clusters.append(
                    {
                        "name": cluster.cluster_name,
                        "state": cluster.status.state.name,
                        "state_start_time": cluster.status.state_start_time,
                    }
                )

            return clusters

        except Exception as e:
            logger.error(f"Failed to list clusters: {str(e)}")
            raise
