# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Accept the version number as a build argument
ARG APP_VERSION

# Set environment variables for Streamlit
ENV STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false \
    STREAMLIT_SERVER_ENABLE_CORS=false \
    APP_VERSION=$APP_VERSION \
    PYTHONPATH=/app

# Copy all application code first
COPY requirements.txt .
COPY synthesis_prompt.txt . 
COPY Home.py .
COPY agent_prompts.py .
COPY pages/ ./pages/
COPY components/ ./components/
COPY tools/ ./tools/

# Copy the .streamlit directory which contains secrets.toml and config.toml
COPY .streamlit/ ./.streamlit/

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
EXPOSE 8501

# Run app.py when the container launches
# The flags below are crucial for running Streamlit behind a reverse proxy, especially for authentication.
CMD ["streamlit", "run", "Home.py", "--server.address=0.0.0.0"]