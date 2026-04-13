#!/bin/sh
# entrypoint.sh

# Wait for the Streamlit app to be healthy
echo "Waiting for Streamlit app to become healthy..."
while ! curl -f http://app:8080/_stcore/health; do
    echo "Streamlit app is not ready yet. Retrying in 2 seconds..."
    sleep 2
done

echo "Streamlit app is healthy. Starting Nginx."

# Start Nginx in the foreground
exec nginx -g 'daemon off;'