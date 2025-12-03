#!/bin/bash

# ==============================================================================
# PyFinAgent Deployment Script
#
# This script automates the deployment of the quant-agent and ingestion-agent
# to Google Cloud Functions. It ensures both functions are deployed with the
# correct settings and environment variables.
#
# Instructions:
# 1. Fill in the configuration variables in the section below.
# 2. Make the script executable by running: chmod +x deploy_agents.sh
# 3. Run the script from this directory: ./deploy_agents.sh
# 4. After deployment, copy the new URLs and update your Streamlit secrets.
# ==============================================================================

# --- Configuration ---
# REQUIRED: Set your GCP Project ID, Region, GCS Bucket, and User-Agent Email.

GCP_PROJECT_ID="sunny-might-477607-p8"      # <-- REPLACE with your GCP Project ID
GCP_REGION="us-central1"                  # <-- REPLACE with your preferred GCP region
BUCKET_NAME="10k-filling-data"      # <-- REPLACE with your GCS bucket name
USER_AGENT_EMAIL="peder.bkoppang@hotmail.no" # <-- REPLACE with your email for the SEC User-Agent
GCP_USER_ACCOUNT="peder.bkoppang@hotmail.no" # <-- The user account that will run this script

# --- Script Logic (No changes needed below this line) ---

echo "Starting PyFinAgent deployment..."
echo "---------------------------------"
echo "Project ID:     $GCP_PROJECT_ID"
echo "Region:         $GCP_REGION"
echo "GCS Bucket:     $BUCKET_NAME"
echo "---------------------------------"
echo

# Validate that placeholder values have been changed
if [[ "$GCP_PROJECT_ID" == "your-gcp-project-id" || "$BUCKET_NAME" == "your-gcs-bucket-name" ]]; then
  echo "ERROR: Please replace the placeholder values in the 'Configuration' section of the script before running."
  exit 1
fi

# --- Pre-flight Checks & Setup ---
echo "Running pre-flight checks and setup..."

# 1. Set the project for the gcloud session
gcloud config set project $GCP_PROJECT_ID

# 2. Set the active account for this command
# Note: You must be logged in to gcloud first. If this fails, run 'gcloud auth login' manually.
echo "Setting active gcloud account to $GCP_USER_ACCOUNT..."
gcloud config set account $GCP_USER_ACCOUNT

# 3. Grant required IAM roles to the user. These commands are idempotent.
#    This ensures the user has permission to deploy and manage Cloud Functions.
echo "Ensuring user has 'Cloud Functions Admin' role..."
gcloud projects add-iam-policy-binding $GCP_PROJECT_ID \
  --member="user:$GCP_USER_ACCOUNT" \
  --role="roles/cloudfunctions.admin" \
  --quiet

echo "Ensuring user has 'Service Account User' role..."
gcloud projects add-iam-policy-binding $GCP_PROJECT_ID \
  --member="user:$GCP_USER_ACCOUNT" \
  --role="roles/iam.serviceAccountUser" \
  --quiet

echo "✅ Pre-flight checks complete."
echo

# --- Pre-deployment: Fetch Dynamic Configuration ---
echo "--- Fetching Redis IP for agent configuration ---"
REDIS_HOST=$(gcloud redis instances describe pyfinagent-redis --region="$GCP_REGION" --project="$GCP_PROJECT_ID" --format="value(host)")

if [[ -z "$REDIS_HOST" ]]; then
  echo "ERROR: Failed to retrieve Redis host IP. Please check that the Redis instance 'pyfinagent-redis' exists in region '$GCP_REGION'."
  exit 1
fi
echo "Found Redis Host IP: $REDIS_HOST"
echo

# --- Pre-deployment: Grant IAM Roles to Service Account ---
echo "--- Granting required IAM roles to the pyfinagent-runner service account ---"
echo "Granting BigQuery Data Editor role..."
gcloud projects add-iam-policy-binding $GCP_PROJECT_ID \
  --member="serviceAccount:pyfinagent-runner@${GCP_PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/bigquery.dataEditor" \
  --condition=None \
  --quiet

echo "✅ IAM policies updated."
echo

# --- 1. Deploy quant-agent ---
echo "--- Checking quant-agent for changes ---"
cd quant-agent
if git diff --quiet HEAD -- .; then
    echo "No changes detected in quant-agent. Skipping deployment."
else
    echo "Changes detected. Deploying quant-agent..."
    gcloud functions deploy quant-agent \
      --gen2 \
      --region="$GCP_REGION" \
      --runtime=python311 \
      --source=. \
      --entry-point=quant_agent \
      --trigger-http \
      --service-account="pyfinagent-runner@${GCP_PROJECT_ID}.iam.gserviceaccount.com" \
      --vpc-connector=pyfinagent-connector \
      --allow-unauthenticated \
      --set-env-vars="GCP_PROJECT_ID=$GCP_PROJECT_ID,REDIS_HOST=$REDIS_HOST,REDIS_PORT=6379,BUCKET_NAME=$BUCKET_NAME,USER_AGENT_EMAIL=$USER_AGENT_EMAIL" \
      || { echo "Deployment of quant-agent failed."; cd ..; exit 1; }
    echo "✅ quant-agent deployment command issued."
fi
cd .. # Return to the root directory
echo

# --- 2. Deploy ingestion-agent ---
echo "--- Checking ingestion-agent for changes ---"
cd ingestion_agent
if git diff --quiet HEAD -- .; then
    echo "No changes detected in ingestion-agent. Skipping deployment."
else
    echo "Changes detected. Deploying ingestion-agent..."
    gcloud functions deploy ingestion-agent \
      --gen2 \
      --region="$GCP_REGION" \
      --runtime=python311 \
      --source=. \
      --entry-point=ingestion_agent_http \
      --trigger-http \
      --service-account="pyfinagent-runner@${GCP_PROJECT_ID}.iam.gserviceaccount.com" \
      --allow-unauthenticated \
      --set-env-vars="GCP_PROJECT_ID=$GCP_PROJECT_ID,BUCKET_NAME=$BUCKET_NAME,USER_AGENT_EMAIL=$USER_AGENT_EMAIL" \
      --timeout=900s \
      || { echo "Deployment of ingestion-agent failed."; cd ..; exit 1; }
    echo "✅ ingestion-agent deployment command issued."
fi
cd .. # Return to the root directory
echo

# --- 3. Deploy risk-management-agent ---
echo "--- Checking risk-management-agent for changes ---"
# The path is different for this agent, it's inside pyfinagent-app
cd pyfinagent-app/risk-management-agent
if git diff --quiet HEAD -- .; then
    echo "No changes detected in risk-management-agent. Skipping deployment."
else
    echo "Changes detected. Deploying risk-management-agent..."
    gcloud functions deploy risk-management-agent \
      --gen2 \
      --region="$GCP_REGION" \
      --runtime=python311 \
      --source=. \
      --entry-point=risk_gatekeeper_http \
      --trigger-http \
      --service-account="pyfinagent-runner@${GCP_PROJECT_ID}.iam.gserviceaccount.com" \
      --vpc-connector=pyfinagent-connector \
      --allow-unauthenticated \
      --set-env-vars="GCP_PROJECT_ID=$GCP_PROJECT_ID,REDIS_HOST=$REDIS_HOST,REDIS_PORT=6379,RMA_RISK_TARGET_FRACTION=0.01,RMA_MAX_DRAWDOWN=0.15,RMA_VIX_HALT=35.0,RMA_VIX_WARNING=25.0,RMA_SKEW_FEAR_THRESHOLD=1.5,RMA_OFI_TOXICITY_THRESHOLD=5.0,RMA_MAX_L1_PARTICIPATION=0.10,RMA_MAX_CONCENTRATION=0.20,RMA_MAX_GROSS_LEVERAGE=1.5" \
      || { echo "Deployment of risk-management-agent failed."; cd ../..; exit 1; }
    echo "✅ risk-management-agent deployment command issued."
fi
cd ../..

echo "🚀 All deployment commands have been issued successfully!"
echo "Please check the Google Cloud Console for deployment status and update your Streamlit secrets with the new URLs."