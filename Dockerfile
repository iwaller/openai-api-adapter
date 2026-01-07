FROM python:3.12-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy project files
COPY pyproject.toml uv.lock ./
COPY openai_api_adapter ./openai_api_adapter

# Install dependencies
RUN uv sync --frozen --no-dev

# Create logs directory
RUN mkdir -p /app/logs

# Expose port
EXPOSE 6600

# Environment variables for logging (can be overridden)
ENV LOG_LEVEL=INFO
ENV LOG_DIR=/app/logs

# Run the application
CMD ["uv", "run", "uvicorn", "openai_api_adapter.main:app", "--host", "0.0.0.0", "--port", "6600"]
