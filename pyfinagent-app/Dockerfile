# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Set environment variables for Streamlit
ENV STREAMLIT_SERVER_PORT=8080 \
    STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false \
    STREAMLIT_SERVER_ENABLE_CORS=false

# Copy all application code first
COPY requirements.txt .
COPY synthesis_prompt.txt . 
COPY Dashboard.py .
COPY pages/ ./pages/

# Copy the secrets file for Streamlit
COPY .streamlit/ .streamlit/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \ 
    pip install --no-cache-dir --no-warn-script-location -r requirements.txt
# RUN addgroup --system --gid 1001 streamlit && \
#     adduser --system --uid 1001 --ingroup streamlit streamlit && \
#     chown -R streamlit:streamlit /app
# 
# # Switch to the non-privileged user
# USER streamlit

# Expose the port the app runs on
EXPOSE 8080

# Run app.py when the container launches
# The flags below are crucial for running Streamlit behind a reverse proxy, especially for authentication.
CMD ["streamlit", "run", "Dashboard.py", "--server.address=0.0.0.0"]