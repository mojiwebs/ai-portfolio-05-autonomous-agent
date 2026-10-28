# Use official lightweight Python image
FROM python:3.11-slim

# Install uv package manager directly into container
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency definition file
COPY pyproject.toml ./

# Install project dependencies globally in container using uv pip
RUN uv pip install --system fastapi uvicorn pydantic httpx openai pytest

# Copy the rest of the application code
COPY . .

# Default command runs the evaluation harness to test the agent stack
CMD ["python", "src/evaluation/evaluator.py"]