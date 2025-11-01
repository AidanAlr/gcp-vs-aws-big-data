import boto3
import time
from typing import Optional, Dict
import sys
import os

from config import AWS_CONFIG
from utils.logger_config import setup_logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = setup_logger("aws_emr")


class EMRCluster:
    """
    Manages AWS EMR cluster.

    Provides methods to:
    - Create EMR clusters
    - Monitor cluster status
    - Wait for cluster to be ready
    - Terminate clusters
    """

    def __init__(self, region: Optional[str] = None):
        """
        Initialize EMR cluster manager.

        Args:
            region: AWS region (defaults to config value)
        """
        self.region = region or AWS_CONFIG["region"]
        self.emr_client = boto3.client("emr", region_name=self.region)
        self.cluster_id = None
        self.cluster_name = AWS_CONFIG["cluster_name"]

    def terminate_all_active_clusters(self, wait_for_termination: bool = True) -> int:
        """
        Terminate all active clusters in the region.

        Args:
            wait_for_termination: Whether to wait for clusters to finish terminating

        Returns:
            Number of clusters terminated
        """
        logger.info("Checking for active clusters to terminate...")

        active_clusters = self.list_active_clusters()

        if not active_clusters:
            logger.info("No active clusters found")
            return 0

        logger.info(f"Found {len(active_clusters)} active cluster(s), terminating...")

        for cluster in active_clusters:
            cluster_id = cluster["id"]
            cluster_name = cluster["name"]
            cluster_state = cluster["state"]
            logger.info(f"Terminating cluster: {cluster_id} ({cluster_name}) - State: {cluster_state}")

            try:
                self.emr_client.terminate_job_flows(JobFlowIds=[cluster_id])
                logger.info(f"Termination initiated for cluster: {cluster_id}")
            except Exception as e:
                logger.error(f"Failed to terminate cluster {cluster_id}: {str(e)}")

        # Wait for all clusters to terminate if requested
        if wait_for_termination:
            logger.info("Waiting for all clusters to terminate...")
            for cluster in active_clusters:
                cluster_id = cluster["id"]
                try:
                    self.wait_for_cluster_terminated(
                        cluster_id=cluster_id,
                        timeout_minutes=15,
                        check_interval_seconds=20
                    )
                except Exception as e:
                    logger.warning(f"Error waiting for cluster {cluster_id} to terminate: {str(e)}")

        return len(active_clusters)

    def create_cluster(self, cleanup_existing: bool = True) -> str:
        """
        Create an EMR cluster with the configured specifications.

        Args:
            cleanup_existing: If True, terminate any existing clusters before creating new one

        Returns:
            Cluster ID

        Raises:
            Exception: If cluster creation fails
        """
        logger.info(f"Creating EMR cluster: {self.cluster_name}")

        # Cleanup existing clusters if requested
        if cleanup_existing:
            num_terminated = self.terminate_all_active_clusters(wait_for_termination=True)
            if num_terminated > 0:
                logger.info(f"Cleaned up {num_terminated} existing cluster(s)")
                # Brief pause to ensure AWS is ready for new cluster
                logger.info("Waiting 30 seconds before creating new cluster...")
                time.sleep(30)

        try:
            # Build instances config
            instances_config = {
                "MasterInstanceType": AWS_CONFIG["master_instance_type"],
                "SlaveInstanceType": AWS_CONFIG["core_instance_type"],
                "InstanceCount": 1
                + AWS_CONFIG["core_instance_count"],  # master + core nodes
                "KeepJobFlowAliveWhenNoSteps": True,
                "TerminationProtected": False,
            }

            # Only add Ec2KeyName if it's configured
            if AWS_CONFIG.get("ec2_key_name"):
                instances_config["Ec2KeyName"] = AWS_CONFIG["ec2_key_name"]

            response = self.emr_client.run_job_flow(
                Name=self.cluster_name,
                ReleaseLabel=AWS_CONFIG["release_label"],
                LogUri=AWS_CONFIG["log_uri"],
                Instances=instances_config,
                Applications=[
                    {"Name": "Spark"},
                    {"Name": "Hadoop"},
                ],
                VisibleToAllUsers=True,
                JobFlowRole="EMR_EC2_DefaultRole",
                ServiceRole="EMR_DefaultRole",
                Tags=[
                    {"Key": "Purpose", "Value": "ML-Comparison"},
                    {"Key": "Project", "Value": "AWS-vs-GCP"},
                ],
            )

            self.cluster_id = response["JobFlowId"]
            logger.info(f"Cluster created successfully: {self.cluster_id}")
            return self.cluster_id

        except Exception as e:
            logger.error(f"Failed to create EMR cluster: {str(e)}")
            raise

    def get_cluster_status(self, cluster_id: Optional[str] = None) -> Dict:
        """
        Get the current status of a cluster.

        Args:
            cluster_id: Cluster ID (uses self.cluster_id if not provided)

        Returns:
            Dictionary with cluster status information

        Raises:
            ValueError: If no cluster ID is available
        """
        cid = cluster_id or self.cluster_id
        if not cid:
            raise ValueError("No cluster ID provided or set")

        try:
            response = self.emr_client.describe_cluster(ClusterId=cid)
            cluster = response["Cluster"]

            status_info = {
                "cluster_id": cid,
                "name": cluster["Name"],
                "state": cluster["Status"]["State"],
                "state_change_reason": cluster["Status"].get("StateChangeReason", {}),
                "creation_time": cluster["Status"]["Timeline"].get("CreationDateTime"),
                "ready_time": cluster["Status"]["Timeline"].get("ReadyDateTime"),
                "end_time": cluster["Status"]["Timeline"].get("EndDateTime"),
            }

            return status_info

        except Exception as e:
            logger.error(f"Failed to get cluster status: {str(e)}")
            raise

    def wait_for_cluster_ready(
        self,
        cluster_id: Optional[str] = None,
        timeout_minutes: int = 30,
        check_interval_seconds: int = 30,
    ) -> bool:
        """
        Wait for cluster to reach WAITING state (ready for jobs).

        Args:
            cluster_id: Cluster ID (uses self.cluster_id if not provided)
            timeout_minutes: Maximum time to wait
            check_interval_seconds: How often to check status

        Returns:
            True if cluster is ready, False if timeout or failed

        Raises:
            ValueError: If no cluster ID is available
        """
        cid = cluster_id or self.cluster_id
        if not cid:
            raise ValueError("No cluster ID provided or set")

        logger.info(f"Waiting for cluster {cid} to be ready...")

        start_time = time.time()
        timeout_seconds = timeout_minutes * 60

        while True:
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > timeout_seconds:
                logger.error(
                    f"Timeout waiting for cluster to be ready after {timeout_minutes} minutes"
                )
                return False

            # Get cluster status
            status_info = self.get_cluster_status(cid)
            state = status_info["state"]

            logger.info(f"Cluster state: {state} (elapsed: {elapsed:.0f}s)")

            # Check if ready
            if state == "WAITING":
                logger.info("Cluster is ready!")
                return True

            # Check for failure states
            if state in ["TERMINATING", "TERMINATED", "TERMINATED_WITH_ERRORS"]:
                reason = status_info.get("state_change_reason", {})
                logger.error(f"Cluster failed: {state}, reason: {reason}")
                return False

            # Wait before next check
            time.sleep(check_interval_seconds)

    def terminate_cluster(self, cluster_id: Optional[str] = None) -> None:
        """
        Terminate an EMR cluster.

        Args:
            cluster_id: Cluster ID (uses self.cluster_id if not provided)

        Raises:
            ValueError: If no cluster ID is available
        """
        cid = cluster_id or self.cluster_id
        if not cid:
            raise ValueError("No cluster ID provided or set")

        logger.info(f"Terminating cluster: {cid}")

        try:
            self.emr_client.terminate_job_flows(JobFlowIds=[cid])
            logger.info(f"Cluster termination initiated: {cid}")

        except Exception as e:
            logger.error(f"Failed to terminate cluster: {str(e)}")
            raise

    def wait_for_cluster_terminated(
        self,
        cluster_id: Optional[str] = None,
        timeout_minutes: int = 15,
        check_interval_seconds: int = 30,
    ) -> bool:
        """
        Wait for cluster to be fully terminated.

        Args:
            cluster_id: Cluster ID (uses self.cluster_id if not provided)
            timeout_minutes: Maximum time to wait
            check_interval_seconds: How often to check status

        Returns:
            True if cluster is terminated, False if timeout
        """
        cid = cluster_id or self.cluster_id
        if not cid:
            raise ValueError("No cluster ID provided or set")

        logger.info(f"Waiting for cluster {cid} to terminate...")

        start_time = time.time()
        timeout_seconds = timeout_minutes * 60

        while True:
            elapsed = time.time() - start_time
            if elapsed > timeout_seconds:
                logger.warning(
                    f"Timeout waiting for termination after {timeout_minutes} minutes"
                )
                return False

            status_info = self.get_cluster_status(cid)
            state = status_info["state"]

            logger.info(f"Cluster state: {state} (elapsed: {elapsed:.0f}s)")

            if state in ["TERMINATED", "TERMINATED_WITH_ERRORS"]:
                logger.info("Cluster terminated")
                return True

            time.sleep(check_interval_seconds)

    def list_active_clusters(self) -> list:
        """
        List all active clusters in the region.

        Returns:
            List of cluster information dictionaries
        """
        try:
            response = self.emr_client.list_clusters(
                ClusterStates=["STARTING", "BOOTSTRAPPING", "RUNNING", "WAITING"]
            )

            clusters = []
            for cluster in response.get("Clusters", []):
                clusters.append(
                    {
                        "id": cluster["Id"],
                        "name": cluster["Name"],
                        "state": cluster["Status"]["State"],
                        "creation_time": cluster["Status"]["Timeline"][
                            "CreationDateTime"
                        ],
                    }
                )

            return clusters

        except Exception as e:
            logger.error(f"Failed to list clusters: {str(e)}")
            raise
