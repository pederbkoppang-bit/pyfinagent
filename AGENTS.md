# AGENTS.md for PyFinAgent

This document provides essential context and instructions for AI coding agents to effectively understand, operate, and contribute to the PyFinAgent project.

## Project Overview

*   **Project Name**: PyFinAgent
*   **Purpose**: An agentic AI financial analyst that performs deep-dive company analysis by orchestrating multiple specialized agents to produce a comprehensive investment report.
*   **Core Function**: The system coordinates a series of expert AI agents, each with a specific role, to gather, analyze, and synthesize financial data into a final, structured output.

## System Roles

The system is composed of a frontend coordinator, backend data-processing agents, and a suite of in-app analytical agents.

#### Frontend: Streamlit Coordinator (`pyfinagent-app`)

*   Orchestrates the entire analysis pipeline from user input to final report generation.
*   Implements the 'Glass Box' philosophy by providing a transparent dashboard to observe the inputs and outputs of each agent in the workflow.

#### Backend Agents (GCP Cloud Functions)

*   `quant-agent`: Fetches quantitative data from SEC filings (EDGAR) and market data from yfinance.
*   `ingestion-agent`: Loads fetched data into Google BigQuery for structured storage and analysis.
*   `risk-management-agent`: Acts as a guardrail, validating data and ensuring the analysis process adheres to predefined rules.
*   `earnings-ingestion-agent`: Ingests and processes earnings call transcripts for qualitative analysis.

#### In-App Agents (LLM-based)

*   `RAG Agent`: Analyzes 10-K and 10-Q documents, focusing on Economic Moat, Governance, and key business Risks.
*   `Market Agent`: Analyzes the macroeconomic environment (PESTEL) and overall market sentiment.
*   `Competitor Scout`: Identifies and analyzes key competitors to assess the competitive landscape.
*   `Macro Strategist`: Provides high-level macroeconomic context and identifies relevant industry trends.
*   `Deep Dive Agent`: Answers critical, specific questions by performing targeted searches within 10-K documents.
*   `Synthesis Agent`: Synthesizes the outputs from all other agents into a final, structured JSON investment analysis report.
*   `Critic Agent`: Reviews the final report for hallucinations, logical inconsistencies, and factual errors before presenting it to the user.

## Environment & Setup

*   **Docker**: The local development environment is managed via Docker and Docker Compose. To start all required services (like Redis), run:
    ```bash
    docker-compose up
    ```
*   **Streamlit Secrets**: For local development, API keys and configuration are managed in `pyfinagent-app/.streamlit/secrets.toml`.
    *   This file is explicitly ignored by git.
    *   Refer to `secrets.toml.example` for the required structure and variables.
    *   Do **NOT** commit the `secrets.toml` file.

## Key Conventions

*   **Agent Communication**: All communication and data handoffs between agents **MUST** be in a structured JSON format. This ensures predictable and parsable inputs/outputs. Refer to the agent-specific prompt files (e.g., `synthesis_prompt.txt`) for the exact JSON schemas.
*   **Glass Box Philosophy**: The frontend is intentionally designed as a "Glass Box." When modifying the Streamlit app, prioritize transparency. Ensure that the inputs, outputs, and reasoning process of each agent are clearly visible to the user.

## Deployment & Testing

*   **Backend Agents**: The GCP Cloud Functions are deployed using a shell script. To deploy all backend agents, run:
    ```bash
    ./deploy_agents.sh
    ```
*   **Frontend Application**: The Streamlit application is deployed using its own dedicated script. To deploy the frontend, run:
    ```bash
    ./pyfinagent-app/deploy.sh
    ```

## Security

*   **SEC EDGAR User-Agent**: When making requests to the SEC EDGAR database, you **MUST** declare a custom User-Agent string in the format `FirstName LastName email@domain.com`. Failure to do so will result in the IP address being blocked by the SEC. This is managed via secrets.
*   **Secret Management**:
    *   **Local**: Use the `pyfinagent-app/.streamlit/secrets.toml` file for local development.
    *   **Production**: All production secrets (API keys, service account keys) are managed using **Google Cloud Secret Manager**. Do not hardcode secrets in the source code.