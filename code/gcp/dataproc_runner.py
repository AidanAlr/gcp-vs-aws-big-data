from google.cloud import dataproc_v1
from google.cloud import storage
import time
from typing import Optional, Dict, List
import sys
import os

from config import GCP_CONFIG
from utils.logger_config import setup_logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logger = setup_logger("gcp_dataproc_runner")


class DataprocRunner:
    """
    Manages Spark job execution on GCP Dataproc clusters.

    Provides methods to:
    - Submit Spark jobs
    - Monitor job progress
    - Get job results
    """

    def __init__(
        self,
        cluster_name: str,
        project_id: Optional[str] = None,
        region: Optional[str] = None,
    ):
        """
        Initialize Dataproc runner.

        Args:
            cluster_name: Dataproc cluster name to run jobs on
            project_id: GCP project ID (defaults to config value)
            region: GCP region (defaults to config value)
        """
        self.cluster_name = cluster_name
        self.project_id = project_id or GCP_CONFIG["project_id"]
        self.region = region or GCP_CONFIG["region"]

        # Initialize clients
        self.job_client = dataproc_v1.JobControllerClient(
            client_options={
                "api_endpoint": f"{self.region}-dataproc.googleapis.com:443"
            }
        )
        self.storage_client = storage.Client(project=self.project_id)

    def upload_script_to_gcs(
        self, local_script_path: str, gcs_path: Optional[str] = None
    ) -> str:
        """
        Upload a local script file to Google Cloud Storage.

        Args:
            local_script_path: Path to local script file
            gcs_path: GCS path (defaults to scripts/filename in config bucket)

        Returns:
            Full GCS URI of uploaded script

        Raises:
            Exception: If upload fails
        """
        bucket_name = GCP_CONFIG["gcs_bucket"]

        if gcs_path is None:
            script_name = os.path.basename(local_script_path)
            gcs_path = f"scripts/{script_name}"

        logger.info(f"Uploading {local_script_path} to gs://{bucket_name}/{gcs_path}")

        try:
            bucket = self.storage_client.bucket(bucket_name)
            blob = bucket.blob(gcs_path)
            blob.upload_from_filename(local_script_path)

            gcs_uri = f"gs://{bucket_name}/{gcs_path}"
            logger.info(f"Script uploaded successfully to {gcs_uri}")
            return gcs_uri

        except Exception as e:
            logger.error(f"Failed to upload script to GCS: {str(e)}")
            raise

    def submit_pyspark_job(
        self,
        main_python_file_uri: str,
        job_name: Optional[str] = None,
        args: Optional[List[str]] = None,
        python_file_uris: Optional[List[str]] = None,
        jar_file_uris: Optional[List[str]] = None,
        properties: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Submit a PySpark job to the Dataproc cluster.

        Args:
            main_python_file_uri: GCS URI of the main PySpark script
            job_name: Name for the job (auto-generated if not provided)
            args: Arguments to pass to the script
            python_file_uris: Additional Python files to include
            jar_file_uris: JAR files to include
            properties: Spark properties

        Returns:
            Job ID of the submitted job

        Raises:
            Exception: If job submission fails
        """
        if job_name is None:
            import uuid

            job_name = f"ml-training-{uuid.uuid4().hex[:8]}"

        logger.info(f"Submitting PySpark job: {job_name}")
        logger.info(f"Main script: {main_python_file_uri}")

        # Build PySpark job configuration
        job = {
            "placement": {"cluster_name": self.cluster_name},
            "pyspark_job": {
                "main_python_file_uri": main_python_file_uri,
            },
        }

        if args:
            job["pyspark_job"]["args"] = args
            logger.info(f"Arguments: {args}")

        if python_file_uris:
            job["pyspark_job"]["python_file_uris"] = python_file_uris

        if jar_file_uris:
            job["pyspark_job"]["jar_file_uris"] = jar_file_uris

        if properties:
            job["pyspark_job"]["properties"] = properties

        try:
            operation = self.job_client.submit_job_as_operation(
                request={
                    "project_id": self.project_id,
                    "region": self.region,
                    "job": job,
                }
            )

            # Extract job ID from operation metadata
            job_id = operation.metadata.job_id

            logger.info(f"Job submitted successfully. Job ID: {job_id}")
            return job_id

        except Exception as e:
            logger.error(f"Failed to submit job: {str(e)}")
            raise

    def get_job_status(self, job_id: str) -> Dict:
        """
        Get the status of a job.

        Args:
            job_id: Job ID to check

        Returns:
            Dictionary with job status information

        Raises:
            Exception: If status check fails
        """
        try:
            job = self.job_client.get_job(
                request={
                    "project_id": self.project_id,
                    "region": self.region,
                    "job_id": job_id,
                }
            )

            status_info = {
                "job_id": job_id,
                "state": job.status.state.name,
                "state_start_time": job.status.state_start_time,
            }

            return status_info

        except Exception as e:
            logger.error(f"Failed to get job status: {str(e)}")
            raise

    def wait_for_job_completion(
        self, job_id: str, timeout_minutes: int = 60, check_interval_seconds: int = 30
    ) -> bool:
        """
        Wait for a job to complete.

        Args:
            job_id: Job ID to monitor
            timeout_minutes: Maximum time to wait
            check_interval_seconds: How often to check status

        Returns:
            True if job completed successfully, False otherwise
        """
        logger.info(f"Waiting for job {job_id} to complete...")

        start_time = time.time()
        timeout_seconds = timeout_minutes * 60

        while True:
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > timeout_seconds:
                logger.error(f"Timeout waiting for job after {timeout_minutes} minutes")
                return False

            # Get job status
            status_info = self.get_job_status(job_id)
            state = status_info["state"]

            logger.info(f"Job state: {state} (elapsed: {elapsed:.0f}s)")

            # Check if completed
            if state == "DONE":
                logger.info("Job completed successfully!")
                return True

            # Check for failure states
            if state in ["ERROR", "CANCELLED"]:
                detail = status_info.get("state_detail", "No details")
                logger.error(f"Job failed: {state}, detail: {detail}")
                return False

            # Wait before next check
            time.sleep(check_interval_seconds)

    def list_jobs(self, max_results: int = 10) -> List[Dict]:
        """
        List recent jobs on the cluster.

        Args:
            max_results: Maximum number of jobs to return

        Returns:
            List of job information dictionaries
        """
        try:
            request = {
                "project_id": self.project_id,
                "region": self.region,
                "cluster_name": self.cluster_name,
            }

            jobs = []
            page_result = self.job_client.list_jobs(request=request)

            for i, job in enumerate(page_result):
                if i >= max_results:
                    break

                jobs.append(
                    {
                        "job_id": job.reference.job_id,
                        "state": job.status.state.name,
                        "state_start_time": job.status.state_start_time,
                    }
                )

            return jobs

        except Exception as e:
            logger.error(f"Failed to list jobs: {str(e)}")
            raise

    def download_results_from_gcs(
        self, gcs_path: str, local_path: str, bucket_name: Optional[str] = None
    ) -> None:
        """
        Download results file from GCS to local filesystem.

        Args:
            gcs_path: GCS path of the file to download
            local_path: Local path to save the file
            bucket_name: GCS bucket (defaults to config bucket)

        Raises:
            Exception: If download fails
        """
        bucket_name = bucket_name or GCP_CONFIG["gcs_bucket"]
        logger.info(f"Downloading gs://{bucket_name}/{gcs_path} to {local_path}")

        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(local_path), exist_ok=True)

            bucket = self.storage_client.bucket(bucket_name)
            blob = bucket.blob(gcs_path)
            blob.download_to_filename(local_path)

            logger.info(f"File downloaded successfully to {local_path}")

        except Exception as e:
            logger.error(f"Failed to download from GCS: {str(e)}")
            raise
