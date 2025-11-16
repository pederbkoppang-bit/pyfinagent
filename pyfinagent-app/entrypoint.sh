#!/bin/sh
# entrypoint.sh

echo "Waiting for Streamlit app to start..."
sleep 5

exec nginx -g 'daemon off;'