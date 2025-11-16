#!/bin/bash

# This script is used to run the Streamlit application in debug mode.
# It starts the application, tails the debug log for real-time error
# checking, and provides a cleanup mechanism to stop the app.

APP_DIR="$(dirname "$(readlink -f "$0")")"
LOG_FILE="$APP_DIR/debug.log"

# Function to clean up background processes on exit
cleanup() {
    echo "Stopping Streamlit app..."
    if [ -n "$PID" ]; then
        kill "$PID"
    fi
    exit
}

trap cleanup SIGINT SIGTERM

# Ensure the log file exists before tailing it
rm -f "$LOG_FILE"
touch "$LOG_FILE"

echo "Starting Streamlit app in background..."
# We redirect both stdout and stderr (using &>) to the log file.
# This captures all output from the streamlit process, including early crashes
# that might happen before the Python logging configuration is active.
streamlit run "$APP_DIR/app.py" --server.address=0.0.0.0 --server.port=8081 --server.headless=true --logger.level=debug &> "$LOG_FILE" &
PID=$!

# Wait a moment for the app to start and create the log file
sleep 2

echo "Tailing $LOG_FILE... (Press Ctrl+C to stop)"
tail -f "$LOG_FILE"