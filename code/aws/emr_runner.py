import boto3
import time
from typing import Optional, Dict, List
import sys
import os

from config import AWS_CONFIG
from utils.logger_config import setup_logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logger = setup_logger("aws_emr_runner")


class EMRRunner:
    """
    Manages Spark job execution on AWS EMR clusters.

    Provides methods to:
    - Submit Spark jobs
    - Monitor job progress
    - Get job results
    """

    def __init__(self, cluster_id: str, region: Optional[str] = None):
        """
        Initialize EMR runner.

        Args:
            cluster_id: EMR cluster ID to run jobs on
            region: AWS region (defaults to config value)
        """
        self.cluster_id = cluster_id
        self.region = region or AWS_CONFIG["region"]
        self.emr_client = boto3.client("emr", region_name=self.region)
        self.s3_client = boto3.client("s3", region_name=self.region)

    def upload_script_to_s3(
        self, local_script_path: str, s3_key: Optional[str] = None
    ) -> str:
        """
        Upload a local script file to S3.

        Args:
            local_script_path: Path to local script file
            s3_key: S3 key (defaults to script filename in scripts path)

        Returns:
            Full S3 URI of uploaded script

        Raises:
            Exception: If upload fails
        """
        bucket = AWS_CONFIG["s3_bucket"]

        if s3_key is None:
            script_name = os.path.basename(local_script_path)
            s3_key = f"scripts/{script_name}"

        logger.info(f"Uploading {local_script_path} to s3://{bucket}/{s3_key}")

        try:
            self.s3_client.upload_file(local_script_path, bucket, s3_key)
            s3_uri = f"s3://{bucket}/{s3_key}"
            logger.info(f"Script uploaded successfully to {s3_uri}")
            return s3_uri

        except Exception as e:
            logger.error(f"Failed to upload script to S3: {str(e)}")
            raise

    def submit_spark_job(
        self,
        script_s3_uri: str,
        job_name: str = "ML Training Job",
        args: Optional[List[str]] = None,
        spark_submit_args: Optional[List[str]] = None,
    ) -> str:
        """
        Submit a Spark job to the EMR cluster.

        Args:
            script_s3_uri: S3 URI of the PySpark script
            job_name: Name for the job step
            args: Arguments to pass to the script
            spark_submit_args: Additional spark-submit arguments

        Returns:
            Step ID of the submitted job

        Raises:
            Exception: If job submission fails
        """
        logger.info(f"Submitting Spark job: {job_name}")
        logger.info(f"Script: {script_s3_uri}")

        # Build spark-submit command arguments
        command_args = ["spark-submit"]

        # Add spark-submit specific args if provided
        if spark_submit_args:
            command_args.extend(spark_submit_args)

        # Add the script
        command_args.append(script_s3_uri)

        # Add script arguments if provided
        if args:
            command_args.extend(args)

        logger.info(f"Command: {' '.join(command_args)}")

        try:
            response = self.emr_client.add_job_flow_steps(
                JobFlowId=self.cluster_id,
                Steps=[
                    {
                        "Name": job_name,
                        "ActionOnFailure": "CONTINUE",
                        "HadoopJarStep": {
                            "Jar": "command-runner.jar",
                            "Args": command_args,
                        },
                    }
                ],
            )

            step_id = response["StepIds"][0]
            logger.info(f"Job submitted successfully. Step ID: {step_id}")
            return step_id

        except Exception as e:
            logger.error(f"Failed to submit job: {str(e)}")
            raise

    def get_step_status(self, step_id: str) -> Dict:
        """
        Get the status of a job step.

        Args:
            step_id: Step ID to check

        Returns:
            Dictionary with step status information

        Raises:
            Exception: If status check fails
        """
        try:
            response = self.emr_client.describe_step(
                ClusterId=self.cluster_id, StepId=step_id
            )

            step = response["Step"]
            status_info = {
                "step_id": step_id,
                "name": step["Name"],
                "state": step["Status"]["State"],
                "state_change_reason": step["Status"].get("StateChangeReason", {}),
                "creation_time": step["Status"]["Timeline"].get("CreationDateTime"),
                "start_time": step["Status"]["Timeline"].get("StartDateTime"),
                "end_time": step["Status"]["Timeline"].get("EndDateTime"),
            }

            return status_info

        except Exception as e:
            logger.error(f"Failed to get step status: {str(e)}")
            raise

    def wait_for_step_completion(
        self, step_id: str, timeout_minutes: int = 60, check_interval_seconds: int = 30
    ) -> bool:
        """
        Wait for a job step to complete.

        Args:
            step_id: Step ID to monitor
            timeout_minutes: Maximum time to wait
            check_interval_seconds: How often to check status

        Returns:
            True if step completed successfully, False otherwise
        """
        logger.info(f"Waiting for step {step_id} to complete...")

        start_time = time.time()
        timeout_seconds = timeout_minutes * 60

        while True:
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > timeout_seconds:
                logger.error(
                    f"Timeout waiting for step after {timeout_minutes} minutes"
                )
                return False

            # Get step status
            status_info = self.get_step_status(step_id)
            state = status_info["state"]

            logger.info(f"Step state: {state} (elapsed: {elapsed:.0f}s)")

            # Check if completed
            if state == "COMPLETED":
                logger.info("Step completed successfully!")
                return True

            # Check for failure states
            if state in ["FAILED", "CANCELLED", "INTERRUPTED"]:
                reason = status_info.get("state_change_reason", {})
                logger.error(f"Step failed: {state}, reason: {reason}")
                return False

            # Wait before next check
            time.sleep(check_interval_seconds)

    def list_steps(self, max_results: int = 10) -> List[Dict]:
        """
        List recent steps on the cluster.

        Args:
            max_results: Maximum number of steps to return

        Returns:
            List of step information dictionaries
        """
        try:
            response = self.emr_client.list_steps(
                ClusterId=self.cluster_id,
                StepStates=["PENDING", "RUNNING", "COMPLETED", "FAILED"],
            )

            steps = []
            for step in response.get("Steps", [])[:max_results]:
                steps.append(
                    {
                        "id": step["Id"],
                        "name": step["Name"],
                        "state": step["Status"]["State"],
                        "creation_time": step["Status"]["Timeline"].get(
                            "CreationDateTime"
                        ),
                    }
                )

            return steps

        except Exception as e:
            logger.error(f"Failed to list steps: {str(e)}")
            raise

    def download_results_from_s3(
        self, s3_key: str, local_path: str, bucket: Optional[str] = None
    ) -> None:
        """
        Download results file from S3 to local filesystem.

        Args:
            s3_key: S3 key of the file to download
            local_path: Local path to save the file
            bucket: S3 bucket (defaults to config bucket)

        Raises:
            Exception: If download fails
        """
        bucket = bucket or AWS_CONFIG["s3_bucket"]
        logger.info(f"Downloading s3://{bucket}/{s3_key} to {local_path}")

        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(local_path), exist_ok=True)

            self.s3_client.download_file(bucket, s3_key, local_path)
            logger.info(f"File downloaded successfully to {local_path}")

        except Exception as e:
            logger.error(f"Failed to download from S3: {str(e)}")
            raise
