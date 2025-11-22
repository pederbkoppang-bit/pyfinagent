# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Set environment variables for Streamlit
ENV STREAMLIT_SERVER_PORT=8080 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false

# Create a non-privileged user to run the app
RUN addgroup --system --gid 1001 streamlit && \
    adduser --system --uid 1001 --ingroup streamlit streamlit

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# We install as root, which is standard and safe inside a container.
# The --no-warn-script-location flag suppresses warnings about PATH, as we know the executables
# will be in the system's standard /usr/local/bin.
RUN pip install --no-cache-dir --upgrade pip && \ 
    pip install --no-cache-dir --no-warn-script-location -r requirements.txt

# Copy the rest of the application's code into the container
COPY . .

# Switch to the non-privileged user
USER streamlit

# Expose the port the app runs on
EXPOSE 8080

# Run app.py when the container launches
# The flags below are crucial for running Streamlit behind a reverse proxy like Nginx.
CMD ["streamlit", "run", "app.py", \
     "--browser.serverAddress=0.0.0.0", \
     "--server.enableCORS=false", \
     "--browser.gatherUsageStats=false"]