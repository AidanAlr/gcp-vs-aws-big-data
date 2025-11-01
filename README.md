# AWS EMR vs GCP Dataproc Performance Comparison

## **METCS777 Big Data Analytics - Term Paper Project**

## Table of Contents

- [Sample Datasets](#sample-datasets)
- [Environment Setup](#environment-setup)
- [How to Run](#how-to-run)
- [Results](#results)

---

## Sample Datasets

This project uses two main datasets for benchmarking:

### 1. Transactions Dataset (PySpark Benchmarks)

**Source**: [Kaggle - Simulated Transactions](https://www.kaggle.com/datasets/conorsully1/simulated-transactions)

We test with two dataset sizes:

**Sample Dataset**: `data/transactions/transactions_sample_1m.csv`

- **Size**: 81 MB
- **Rows**: 1,000,000 transactions
- **Columns**: 10 fields
- **Use Case**: Development and quick testing

**Large Dataset**: `data/transactions/transactions_large.csv`

- **Size**: 5.5 GB
- **Rows**: 70,000,000 transactions
- **Columns**: 10 fields
- **Use Case**: Production-scale performance validation

**Schema** (same for both datasets):

```
CUST_ID         - Customer ID
START_DATE      - Customer start date
END_DATE        - Customer end date (if applicable)
TRANS_ID        - Transaction ID
DATE            - Transaction date
YEAR            - Transaction year
MONTH           - Transaction month
DAY             - Transaction day
EXP_TYPE        - Expense type (Groceries, Entertainment, Motor/Travel, Housing, Savings)
AMOUNT          - Transaction amount (USD)
```

**Sample Data**:

```
CUST_ID,START_DATE,END_DATE,TRANS_ID,DATE,YEAR,MONTH,DAY,EXP_TYPE,AMOUNT
CI6XLYUMQK,2015-05-01,,T8I9ZB5A6X90UG8,2015-09-11,2015,9,11,Motor/Travel,20.27
CI6XLYUMQK,2015-05-01,,TZ4JSLS7SC7FO9H,2017-02-08,2017,2,8,Motor/Travel,12.85
CI6XLYUMQK,2015-05-01,,TTUKRDDJ6B6F42H,2015-08-01,2015,8,1,Housing,383.8
```

### 2. Fashion-MNIST Dataset (ML Benchmarks)

**Source**: Fashion-MNIST image classification dataset

**Location**: `data/fashion-mnist/`

- `train.parquet` - Training data
- `test.parquet` - Test data

**Details**:

- 60,000 training images
- 10,000 test images
- 10 clothing categories
- 28x28 grayscale images

---

## Environment Setup

### Prerequisites

- AWS Account with EMR permissions
- GCP Account with Dataproc permissions
- Python 3.8-3.11
- AWS CLI configured
- gcloud SDK configured

### 1. Install Dependencies

```bash
# Clone or navigate to project
cd term-paper

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r code/requirements.txt
```

**Dependencies include**:

- `boto3` - AWS SDK
- `google-cloud-dataproc`, `google-cloud-storage` - GCP SDKs
- `pyspark` - Distributed computing
- `pandas`, `numpy`, `pyarrow` - Data processing
- `python-dotenv` - Configuration management

### 2. Configure Cloud Credentials

#### AWS Setup

```bash
brew install awscli  # macOS

# Configure credentials
aws configure

# Create S3 bucket
aws s3 mb s3://your-bucket-name --region us-east-1

# Create EMR default roles
aws emr create-default-roles
```

#### GCP Setup

```bash
# Install gcloud SDK (if not installed)
brew install --cask google-cloud-sdk  # macOS

# Authenticate and set project
gcloud auth login
gcloud config set project your-project-id

# Enable required APIs
gcloud services enable dataproc.googleapis.com compute.googleapis.com storage.googleapis.com

# Create GCS bucket
gsutil mb -l us-east1 gs://your-bucket-name

# Create service account
gcloud iam service-accounts create dataproc-sa --display-name="Dataproc Service Account"

# Grant permissions
gcloud projects add-iam-policy-binding your-project-id \
  --member="serviceAccount:dataproc-sa@your-project-id.iam.gserviceaccount.com" \
  --role="roles/dataproc.editor"

gcloud projects add-iam-policy-binding your-project-id \
  --member="serviceAccount:dataproc-sa@your-project-id.iam.gserviceaccount.com" \
  --role="roles/storage.admin"

# Generate service account key
gcloud iam service-accounts keys create gcp-key.json \
    --iam-account=dataproc-sa@your-project-id.iam.gserviceaccount.com
```

### 3. Create Configuration File

Create `.env` file in the project root:

```bash
# AWS Configuration
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=us-east-1
AWS_S3_BUCKET=your-bucket-name

# GCP Configuration
GCP_PROJECT_ID=your-project-id
GCP_REGION=us-east1
GCP_GCS_BUCKET=your-bucket-name
GOOGLE_APPLICATION_CREDENTIALS=/path/to/gcp-key.json

# Cluster Configuration
AWS_MASTER_INSTANCE_TYPE=m5.xlarge
AWS_CORE_INSTANCE_TYPE=m5.xlarge
AWS_CORE_INSTANCE_COUNT=2
GCP_MASTER_MACHINE_TYPE=n1-standard-4
GCP_WORKER_MACHINE_TYPE=n1-standard-4
GCP_WORKER_COUNT=2
```

### 4. Upload Sample Data to Cloud Storage

```bash
# Upload transactions sample to AWS S3
aws s3 cp data/transactions/transactions_sample_1m.csv \
  s3://your-bucket-name/data/transactions/transactions_sample_1m.csv

# Upload transactions sample to GCP GCS
gsutil cp data/transactions/transactions_sample_1m.csv \
  gs://your-bucket-name/data/transactions/transactions_sample_1m.csv

# Upload transactions Large dataset
aws cp data/transactions/transactions_large.csv \
  s3://your-bucket-name/data/transactions/transactions_large.csv

aws cp data/transactions/transactions_large.csv \
  s3://your-bucket-name/data/transactions/transactions_large.csv

# Prepare and upload Fashion-MNIST data
cd code
python prepare_data.py --dataset fashion-mnist --upload-aws --upload-gcp
```

---

## How to Run

### Running Benchmarks

Navigate to the code directory and run benchmarks for each cloud provider:

```bash
cd code

# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 1. PySpark Benchmarks (Sample Dataset)

Run multiple iterations to get statistical averages:

```bash
# Run 10 iterations on AWS EMR
python run_multi_benchmark_aws.py --experiment-type pyspark --dataset-mode sample --num-runs 10

# Run 10 iterations on GCP Dataproc
python run_multi_benchmark_gcp.py --experiment-type pyspark --dataset-mode sample --num-runs 10
```

**What this does**:

- Creates EMR/Dataproc cluster
- Uploads PySpark job to cluster
- Processes 1M transactions with 15 operations
- Collects timing and cost metrics
- Deletes cluster automatically
- Saves results to `data/results/`

### 2. Machine Learning Benchmarks

```bash
# Run 10 ML training iterations on AWS
python run_multi_benchmark_aws.py --experiment-type ml --num-runs 10

# Run 10 ML training iterations on GCP
python run_multi_benchmark_gcp.py --experiment-type ml --num-runs 10
```

**What this does**:

- Creates cluster with ML libraries
- Trains Fashion-MNIST classifier
- Measures training time and accuracy
- Records cluster provisioning time
- Saves cost breakdown

### 3. Storage Benchmarks

```bash
# Run 10 storage performance tests on GCP
python run_multi_benchmark_gcp.py --experiment-type storage --num-runs 10

# Or use single-run benchmark
python run_aws_benchmark.py --experiment-type storage
python run_gcp_benchmark.py --experiment-type storage
```

### Command Options

```
--experiment-type {pyspark,ml,storage}  - Benchmark type to run
--dataset-mode {sample,large}           - Dataset size (for pyspark only)
--num-runs N                           - Number of iterations (default: 1)
--no-cleanup                           - Keep cluster running
```

### Expected Output

Each run creates result files in `data/results/`:

```
data/results/
├── pyspark/
│   ├── pyspark_sample_run01_TIMESTAMP.json
│   ├── pyspark_sample_run02_TIMESTAMP.json
│   └── ...
├── ml/
│   ├── ml_run01_TIMESTAMP.json
│   └── ...
├── storage/
│   └── storage_gcp_storage_run01_TIMESTAMP.json
└── summary/
    ├── pyspark_aws_sample_aggregated_TIMESTAMP.json
    ├── pyspark_gcp_sample_aggregated_TIMESTAMP.json
    ├── ml_gcp_aggregated_TIMESTAMP.json
    └── storage_gcp_aggregated_TIMESTAMP.json
```

---

## Configuration & Pricing

### Instance Pricing Configuration

All instance costs and pricing are defined in `code/config.py`. The cost calculations use hourly rates for compute instances and platform fees.

**AWS Pricing**

```python
AWS_PRICING = {
    "m4.large": 0.10,              # USD per hour (EC2 instance)
    "emr_fee_per_instance": 0.03,  # USD per hour (EMR platform fee)
    "s3_storage": 0.023,           # USD per GB-month
    "s3_requests_put": 0.005,      # USD per 1000 requests
}
```

**GCP Pricing**

```python
GCP_PRICING = {
    "n1-standard-2": 0.095,           # USD per hour (VM instance)
    "dataproc_fee_per_vcpu": 0.010,   # USD per vCPU-hour (Dataproc fee)
    "gcs_storage": 0.020,             # USD per GB-month
    "gcs_operations_class_a": 0.005,  # USD per 10K operations
}
```

### Cost Calculation Method

Costs are calculated in `code/utils/cost_calculator.py`:

**AWS EMR Total Cost** =

- Master EC2: `m4.large rate × hours`
- Master EMR Fee: `emr_fee × hours`
- Core EC2: `m4.large rate × hours × core_count`
- Core EMR Fee: `emr_fee × hours × core_count`
- Storage: `s3_storage × GB × (hours/730)`

**GCP Dataproc Total Cost** =

- Master VM: `n1-standard-2 rate × hours`
- Master Dataproc Fee: `dataproc_fee × vCPUs × hours` (2 vCPUs for n1-standard-2)
- Worker VM: `n1-standard-2 rate × hours × worker_count`
- Worker Dataproc Fee: `dataproc_fee × vCPUs × hours × worker_count`
- Storage: `gcs_storage × GB × (hours/730)`

### Instance Configuration

Cluster configurations used in benchmarks (defined in `code/config.py`):

**AWS EMR**:

- Master: `m4.large` (2 vCPUs, 8 GB RAM)
- Workers: 2× `m4.large`
- Region: `us-east-1`

**GCP Dataproc**:

- Master: `n1-standard-2` (2 vCPUs, 7.5 GB RAM)
- Workers: 2× `n1-standard-2`
- Region: `us-east1`

These values can be customized via environment variables in the `.env` file:

```bash
# AWS Configuration
AWS_MASTER_INSTANCE_TYPE=m4.large
AWS_CORE_INSTANCE_TYPE=m4.large
AWS_CORE_INSTANCE_COUNT=2

# GCP Configuration
GCP_MASTER_MACHINE_TYPE=n1-standard-2
GCP_WORKER_MACHINE_TYPE=n1-standard-2
GCP_WORKER_COUNT=2
```

---

## Results

Results from running multiple iterations of each benchmark on both AWS EMR and GCP Dataproc.

### Result Files Location

All benchmark results are stored as JSON files in `data/results/`:

```
data/results/
├── storage/
│   ├── storage_aws_storage_run01_20251031_071549.json
│   ├── storage_gcp_storage_run01_20251031_205346.json
│   └── ...
├── pyspark/
│   ├── pyspark_aws_sample_run01_20251031_153720.json
│   ├── pyspark_gcp_sample_run01_20251031_153157.json
│   ├── pyspark_aws_large_run01_20251031_160620.json
│   ├── pyspark_gcp_large_run01_20251031_155254.json
│   └── ...
└── ml/
    ├── ml_aws_run01_20251031_172431.json
    ├── ml_gcp_run01_20251031_114412.json
    └── ...
```

**Example JSON structure** (PySpark benchmark):

```json
{
  "experiment_id": "pyspark_aws_sample_run01_20251031_153720",
  "provider": "aws",
  "timestamp": "2025-10-31T19:37:20.707286",
  "timings": {
    "benchmark_execution": 151.93,
    "total_time": 152.38
  },
  "costs": {
    "master_ec2": 0.00422,
    "master_emr_fee": 0.00127,
    "core_ec2": 0.00844,
    "core_emr_fee": 0.00253,
    "total_compute": 0.01646,
    "storage": 0.00006
  },
  "metadata": {
    "benchmark_type": "bigdata",
    "dataset_mode": "sample",
    "cluster_config": {
      "master_instance": "m4.large",
      "core_instance": "m4.large",
      "core_instance_count": 2,
      "region": "us-east-1"
    }
  },
  "summary": {
    "total_time_seconds": 152.38,
    "total_cost_usd": 0.032979,
    "success": true
  }
}
```

### Statistical Comparison

We can use `data/results/compare_stats.py` to get this nice comparison between the services

```
================================================================================
STORAGE
================================================================================

UPLOAD:
  AWS (n=10): mean=25.37s  std=1.30s  min=23.34s  max=27.27s
  GCP (n=10): mean=82.71s  std=57.66s  min=31.10s  max=240.40s
  Comparison: GCP is +226.0% vs AWS

DOWNLOAD:
  AWS (n=10): mean=17.02s  std=0.93s  min=15.61s  max=18.44s
  GCP (n=10): mean=26.60s  std=7.32s  min=11.26s  max=35.56s
  Comparison: GCP is +56.3% vs AWS

SEQUENTIAL_READ:
  AWS (n=10): mean=16.36s  std=1.23s  min=14.27s  max=18.17s
  GCP (n=10): mean=23.77s  std=7.87s  min=10.96s  max=36.86s
  Comparison: GCP is +45.3% vs AWS

WRITE:
  AWS (n=10): mean=3.97s  std=0.96s  min=2.38s  max=6.03s
  GCP (n=10): mean=9.65s  std=18.29s  min=1.43s  max=64.20s
  Comparison: GCP is +143.2% vs AWS

================================================================================
PYSPARK SAMPLE
================================================================================

AWS (n=10):
  Time:  mean=125.20s  std=9.07s  min=121.75s  max=152.38s
  Cost:  mean=$0.0271  std=$0.0020  min=$0.0264  max=$0.0330

GCP (n=10):
  Time:  mean=97.19s  std=10.37s  min=91.64s  max=122.83s
  Cost:  mean=$0.0213  std=$0.0020  min=$0.0205  max=$0.0274

Comparison:
  Time: GCP is -22.4% vs AWS
  Cost: GCP is -21.4% vs AWS

================================================================================
PYSPARK LARGE
================================================================================

AWS (n=5):
  Time:  mean=938.06s  std=22.74s  min=911.93s  max=972.88s
  Cost:  mean=$0.2069  std=$0.0049  min=$0.2013  max=$0.2145

GCP (n=5):
  Time:  mean=912.56s  std=22.53s  min=876.39s  max=937.06s
  Cost:  mean=$0.2084  std=$0.0050  min=$0.2003  max=$0.2139

Comparison:
  Time: GCP is -2.7% vs AWS
  Cost: GCP is +0.7% vs AWS

================================================================================
ML
================================================================================

AWS (n=12):
  Time:  mean=1687.31s  std=56.32s  min=1586.07s  max=1780.11s
  Cost:  mean=$0.3652  std=$0.0116  min=$0.3439  max=$0.3842

GCP (n=10):
  Time:  mean=716.37s  std=59.81s  min=630.10s  max=812.86s
  Cost:  mean=$0.1611  std=$0.0134  min=$0.1417  max=$0.1828

Comparison:
  Time: GCP is -57.5% vs AWS
  Cost: GCP is -55.9% vs AWS
```

---

### Key Findings

**Storage Performance:**

- AWS significantly faster across all operations: uploads (226% faster), writes (143% faster), downloads (56% faster), and sequential reads (45% faster)
- GCP shows high variability in performance (high std deviation), especially for upload and write operations
- AWS provides more consistent performance with lower standard deviations

**PySpark Sample Dataset (1M rows, 81 MB):**

- GCP is 22.4% faster and 21.4% cheaper
- Both platforms process in ~2 minutes
- GCP Dataproc has clear advantage for small datasets

**PySpark Large Dataset (70M rows, 5.5 GB):**

- Performance converges: GCP only 2.7% faster
- Costs nearly identical (~$0.21 per run)
- Both platforms scale efficiently

**Machine Learning (Fashion-MNIST):**

- GCP dramatically faster: 57.5% improvement
- GCP much cheaper: 55.9% cost reduction
- AWS avg: 28 min, $0.37 | GCP avg: 12 min, $0.16

### Recommendations

**For Storage Operations:**

- Use AWS S3 for write-heavy workloads
- Performance is more consistent on AWS

**For Small to Medium PySpark Jobs:**

- Use GCP Dataproc for 20%+ cost and time savings

**For Large PySpark Jobs:**

- Choose based on existing infrastructure
- Performance and cost are nearly identical

**For Machine Learning:**

- Strong advantage for GCP: 2x faster, 2x cheaper
- Significant cost savings for ML workloads
