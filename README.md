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

### 4. Upload Data to Cloud Storage

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
- Processes transactions with pyspark
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
  AWS (n=20): mean=41.21s  std=16.97s  min=23.34s  max=73.23s
  GCP (n=20): mean=191.00s  std=144.45s  min=31.10s  max=533.93s
  Comparison: GCP is +363.4% vs AWS

DOWNLOAD:
  AWS (n=20): mean=23.54s  std=9.71s  min=15.61s  max=49.70s
  GCP (n=20): mean=35.86s  std=19.25s  min=11.26s  max=92.57s
  Comparison: GCP is +52.3% vs AWS

SEQUENTIAL_READ:
  AWS (n=20): mean=23.00s  std=9.12s  min=14.27s  max=38.88s
  GCP (n=20): mean=34.05s  std=19.23s  min=10.96s  max=91.64s
  Comparison: GCP is +48.0% vs AWS

WRITE:
  AWS (n=20): mean=5.26s  std=2.32s  min=2.38s  max=12.41s
  GCP (n=20): mean=12.44s  std=14.37s  min=1.49s  max=64.20s
  Comparison: GCP is +136.6% vs AWS

================================================================================
PYSPARK SAMPLE
================================================================================

AWS (n=20):
  Time:  mean=130.80s  std=12.09s  min=121.75s  max=160.53s
  Cost:  mean=$0.0282  std=$0.0025  min=$0.0264  max=$0.0345

GCP (n=20):
  Time:  mean=79.63s  std=14.40s  min=64.72s  max=96.88s
  Cost:  mean=$0.0176  std=$0.0033  min=$0.0142  max=$0.0211

Comparison:
  Time: GCP is -39.1% vs AWS
  Cost: GCP is -37.7% vs AWS

================================================================================
PYSPARK LARGE
================================================================================

AWS (n=15):
  Time:  mean=938.31s  std=20.90s  min=910.72s  max=978.53s
  Cost:  mean=$0.2069  std=$0.0045  min=$0.2010  max=$0.2155

GCP (n=15):
  Time:  mean=810.67s  std=201.49s  min=399.63s  max=953.16s
  Cost:  mean=$0.1853  std=$0.0454  min=$0.0927  max=$0.2171

Comparison:
  Time: GCP is -13.6% vs AWS
  Cost: GCP is -10.5% vs AWS

================================================================================
ML
================================================================================

AWS (n=20):
  Time:  mean=1177.50s  std=45.07s  min=1122.00s  max=1261.74s
  Cost:  mean=$0.2553  std=$0.0097  min=$0.2434  max=$0.2734

GCP (n=20):
  Time:  mean=313.21s  std=46.62s  min=252.46s  max=401.96s
  Cost:  mean=$0.0702  std=$0.0105  min=$0.0567  max=$0.0903

Comparison:
  Time: GCP is -73.4% vs AWS
  Cost: GCP is -72.5% vs AWS

ML METRICS:
  AWS (n=20):
    Accuracy: mean=0.8063  std=0.0000  min=0.8063  max=0.8063
    F1 Score: mean=0.8039  std=0.0000  min=0.8039  max=0.8039
  GCP (n=30):
    Accuracy: mean=0.8063  std=0.0000  min=0.8063  max=0.8063
    F1 Score: mean=0.8039  std=0.0000  min=0.8039  max=0.8039
  ML Metrics Comparison:
    Accuracy: GCP is +0.00% vs AWS
    F1 Score: GCP is -0.00% vs AWS
```

---

### Key Findings

**Storage Performance:**

- AWS significantly faster across all operations: uploads (363.4% faster), writes (136.6% faster), downloads (52.3% faster), and sequential reads (48.0% faster)
- GCP shows very high variability in performance (high std deviation), especially for upload and write operations
- AWS provides more consistent and reliable performance with lower standard deviations

**PySpark Sample Dataset (1M rows, 81 MB):**

- GCP is 39.1% faster and 37.7% cheaper
- AWS avg: 2.2 min, $0.028 | GCP avg: 1.3 min, $0.018
- GCP Dataproc has clear advantage for small datasets

**PySpark Large Dataset (70M rows, 5.5 GB):**

- GCP maintains advantage: 13.6% faster and 10.5% cheaper
- AWS avg: 15.6 min, $0.207 | GCP avg: 13.5 min, $0.185
- Both platforms scale efficiently, but GCP shows better performance at scale
- Note: GCP shows higher variability (std=201.49s) compared to AWS (std=20.90s)

**Machine Learning (Fashion-MNIST):**

- GCP dramatically faster: 73.4% improvement
- GCP much cheaper: 72.5% cost reduction
- AWS avg: 19.6 min, $0.255 | GCP avg: 5.2 min, $0.070
- Both achieve identical model accuracy (0.8063) and F1 score (0.8039)
- Massive performance advantage for ML workloads on GCP

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
