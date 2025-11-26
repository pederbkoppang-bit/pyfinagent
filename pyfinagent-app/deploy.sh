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
# This uses yq (https://github.com/mikefarah/yq), a lightweight and portable command-line YAML/XML/TOML processor.
PROJECT_ID=$(yq '.gcp_service_account.project_id' .streamlit/secrets.toml)

# --- Service Account Authentication ---
# For non-interactive deployment, we will reconstruct the service
# account JSON from secrets.toml to authenticate gcloud.
echo "🔐 Authenticating service account from secrets.toml..."

# The private_key from TOML is malformed with literal '\\n' characters.
# We will use a Python script to load the TOML data, fix the private key string,
# and pipe the corrected JSON directly to the gcloud command's standard input.
SERVICE_ACCOUNT_EMAIL=$(yq '.gcp_service_account.client_email' .streamlit/secrets.toml)

python3 -c 'import toml, json, sys; secrets = toml.load(".streamlit/secrets.toml")["gcp_service_account"]; secrets["private_key"] = secrets["private_key"].replace("\\n", "\n"); print(json.dumps(secrets))' | gcloud auth activate-service-account "$SERVICE_ACCOUNT_EMAIL" --key-file=-

# Set the active project.
gcloud config set project "$PROJECT_ID"
echo "✅ gcloud authenticated for project $PROJECT_ID."

# --- Deploy Cloud Function Agents (if changed) ---
echo "🔄 Checking for updates to backend agents..."
# The agent deployment script is in the parent directory.
# It has its own logic to check for git changes and will skip deployment if there are none.
if [ -f ../deploy_agents.sh ]; then
    (cd .. && ./deploy_agents.sh) || { echo "Agent deployment failed. Aborting app deployment."; exit 1; }
else
    echo "⚠️ Warning: ../deploy_agents.sh not found. Skipping agent deployment."
fi

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
echo "▶️  Starting new container '$CONTAINER_NAME' from image '$IMAGE_NAME'..."
docker run -d -p $HOST_PORT:$CONTAINER_PORT --name $CONTAINER_NAME $IMAGE_NAME

echo "🎉 Deployment complete! Your application is running."
echo "➡️  Access it at: http://localhost:$HOST_PORT"