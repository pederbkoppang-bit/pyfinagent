#!/bin/bash

# This script automates the process of rebuilding and restarting the Docker container.
# It ensures that the latest code changes are deployed and running.

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
# Define the names for your container and image to make them easy to change.
CONTAINER_NAME="pyfinagent-container"
IMAGE_NAME="pyfinagent-app"
HOST_PORT="8501"
CONTAINER_PORT="8501" # This should match the EXPOSE port in your Dockerfile

# Generate a version number based on the current timestamp (e.g., 20231027-143000)
APP_VERSION=$(date +%Y%m%d-%H%M%S)
echo "📦 Version: $APP_VERSION"

# --- Script Start ---
# Read the project ID from the secrets file to ensure we're using the correct one.
# This requires `toml-cli` to be installed: `pip install toml-cli`
PROJECT_ID=$(toml get --toml-path .streamlit/secrets.toml gcp.project_id | tr -d '"')

# --- Service Account Authentication ---
# Authenticate using the service account key file for non-interactive deployment.
# This avoids the need for browser-based login.
echo "🔐 Using service account for authentication."
gcloud auth activate-service-account --key-file=service-account-key.json
gcloud config set project "$PROJECT_ID"


echo "🚀 Starting deployment script..."

# 1. Stop and remove the old container if it exists.
# The '-q' flag gets the quiet ID. If it returns anything, the container exists.
if [ "$(docker ps -a -q -f name=$CONTAINER_NAME)" ]; then
    echo "✅ Stopping and removing existing container: $CONTAINER_NAME"
    docker rm -f $CONTAINER_NAME
else
    echo "ℹ️ No existing container named '$CONTAINER_NAME' found. Skipping removal."
fi

# 2. Prune old Docker resources to ensure there is enough space for the new build.
# This is crucial in environments with limited disk space like Cloud Shell.
echo "🧹 Pruning old Docker resources (containers, images, networks)..."
docker system prune -f

# 2. Build the new Docker image from the Dockerfile in the current directory.
echo "🛠️  Building new Docker image: $IMAGE_NAME"
docker build --progress=plain --build-arg APP_VERSION=$APP_VERSION -t $IMAGE_NAME .

# 4. Run the new container in detached mode.
# We mount the secrets directory and provide the service account key directly to the container
# via a volume mount and the GOOGLE_APPLICATION_CREDENTIALS environment variable.
echo "▶️  Starting new container '$CONTAINER_NAME' from image '$IMAGE_NAME'..."
docker run -d -p $HOST_PORT:$CONTAINER_PORT --name $CONTAINER_NAME \
    -v "$(pwd)/.streamlit:/app/.streamlit" \
    -v "$(pwd)/service-account-key.json:/app/service-account-key.json" \
    -e GOOGLE_APPLICATION_CREDENTIALS="/app/service-account-key.json" \
    $IMAGE_NAME

echo "🎉 Deployment complete! Your application is running."
echo "➡️  Access it at: http://localhost:$HOST_PORT"