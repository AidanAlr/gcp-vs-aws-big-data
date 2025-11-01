#!/bin/bash
# GCP Service Account Setup Script
# This script automates the creation and configuration of a GCP service account
# with the necessary permissions for Dataproc and Cloud Storage operations.

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}GCP Service Account Setup${NC}"
echo -e "${GREEN}========================================${NC}"

# Load environment variables from .env
if [ -f .env ]; then
    source .env
else
    echo -e "${RED}Error: .env file not found!${NC}"
    echo "Please create a .env file based on .env.example"
    exit 1
fi

# Validate required environment variables
if [ -z "$GCP_PROJECT_ID" ]; then
    echo -e "${RED}Error: GCP_PROJECT_ID not set in .env file${NC}"
    exit 1
fi

SERVICE_ACCOUNT_NAME="dataproc-ml-sa"
SERVICE_ACCOUNT_EMAIL="${SERVICE_ACCOUNT_NAME}@${GCP_PROJECT_ID}.iam.gserviceaccount.com"
KEY_FILE="gcp-service-account-key.json"

echo -e "${YELLOW}Project ID: ${GCP_PROJECT_ID}${NC}"
echo -e "${YELLOW}Service Account: ${SERVICE_ACCOUNT_EMAIL}${NC}"
echo ""

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}Error: gcloud CLI is not installed${NC}"
    echo "Please install it from: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Check if user is authenticated
echo -e "${GREEN}[1/5] Checking gcloud authentication...${NC}"
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    echo -e "${YELLOW}Not authenticated. Running 'gcloud auth login'...${NC}"
    gcloud auth login
fi

# Set the active project
echo -e "${GREEN}[2/5] Setting active project...${NC}"
gcloud config set project "$GCP_PROJECT_ID"

# Check if service account already exists
echo -e "${GREEN}[3/5] Checking if service account exists...${NC}"
if gcloud iam service-accounts describe "$SERVICE_ACCOUNT_EMAIL" --project="$GCP_PROJECT_ID" &> /dev/null; then
    echo -e "${YELLOW}Service account already exists. Skipping creation.${NC}"
else
    echo -e "${GREEN}Creating service account...${NC}"
    gcloud iam service-accounts create "$SERVICE_ACCOUNT_NAME" \
        --display-name="Dataproc ML Service Account" \
        --project="$GCP_PROJECT_ID"
    echo -e "${GREEN}Service account created successfully!${NC}"
fi

# Grant IAM roles
echo -e "${GREEN}[4/5] Granting IAM roles...${NC}"

echo "  - Granting Dataproc Editor role..."
gcloud projects add-iam-policy-binding "$GCP_PROJECT_ID" \
    --member="serviceAccount:${SERVICE_ACCOUNT_EMAIL}" \
    --role="roles/dataproc.editor" \
    --quiet

echo "  - Granting Storage Admin role..."
gcloud projects add-iam-policy-binding "$GCP_PROJECT_ID" \
    --member="serviceAccount:${SERVICE_ACCOUNT_EMAIL}" \
    --role="roles/storage.admin" \
    --quiet

echo "  - Granting Service Account User role..."
gcloud projects add-iam-policy-binding "$GCP_PROJECT_ID" \
    --member="serviceAccount:${SERVICE_ACCOUNT_EMAIL}" \
    --role="roles/iam.serviceAccountUser" \
    --quiet

echo -e "${GREEN}IAM roles granted successfully!${NC}"

# Generate service account key
echo -e "${GREEN}[5/5] Generating service account key...${NC}"

# Remove old key file if it exists
if [ -f "$KEY_FILE" ]; then
    echo -e "${YELLOW}Removing old key file...${NC}"
    rm "$KEY_FILE"
fi

gcloud iam service-accounts keys create "$KEY_FILE" \
    --iam-account="$SERVICE_ACCOUNT_EMAIL" \
    --project="$GCP_PROJECT_ID"

echo -e "${GREEN}Service account key generated: ${KEY_FILE}${NC}"

# Update .env file
echo -e "${GREEN}Updating .env file...${NC}"
KEY_FILE_PATH="$(pwd)/${KEY_FILE}"

# Check if GOOGLE_APPLICATION_CREDENTIALS already exists in .env
if grep -q "GOOGLE_APPLICATION_CREDENTIALS=" .env; then
    # Update existing line
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        sed -i '' "s|GOOGLE_APPLICATION_CREDENTIALS=.*|GOOGLE_APPLICATION_CREDENTIALS=${KEY_FILE_PATH}|" .env
    else
        # Linux
        sed -i "s|GOOGLE_APPLICATION_CREDENTIALS=.*|GOOGLE_APPLICATION_CREDENTIALS=${KEY_FILE_PATH}|" .env
    fi
    echo -e "${GREEN}Updated GOOGLE_APPLICATION_CREDENTIALS in .env${NC}"
else
    # Append new line
    echo "" >> .env
    echo "GOOGLE_APPLICATION_CREDENTIALS=${KEY_FILE_PATH}" >> .env
    echo -e "${GREEN}Added GOOGLE_APPLICATION_CREDENTIALS to .env${NC}"
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Setup completed successfully!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${YELLOW}Service Account Details:${NC}"
echo -e "  Email: ${SERVICE_ACCOUNT_EMAIL}"
echo -e "  Key File: ${KEY_FILE_PATH}"
echo ""
echo -e "${YELLOW}Granted Roles:${NC}"
echo -e "  - roles/dataproc.editor (Create/manage Dataproc clusters)"
echo -e "  - roles/storage.admin (Full GCS access)"
echo -e "  - roles/iam.serviceAccountUser (Use service account)"
echo ""
echo -e "${GREEN}You can now run your GCP experiments!${NC}"
