# MindMirror Production Multi-Component Container
# Base Python 3.12 Slim Environment
FROM python:3.12-slim AS base

# Install system dependencies for audio, build tools, and libzmq
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libportaudio2 \
    libportaudiocpp0 \
    portaudio19-dev \
    libasound2 \
    libasound2-plugins \
    libsndfile1 \
    libzmq3-dev \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source, runtime agent data, and scripts
COPY src/ /app/src/
COPY data/AGENTS.md /app/data/AGENTS.md
COPY data/skills/ /app/data/skills/
COPY scripts/ /app/scripts/
COPY tests/ /app/tests/
COPY pytest.ini /app/
COPY README.md /app/

ENV PYTHONPATH=/app/src:/app
ENV PYTHONUNBUFFERED=1

# Default health check via LocalBus self-diagnostic
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "from mindmirror.pipeline.transport.local import LocalBus; import asyncio; bus = LocalBus(); asyncio.run(bus.close())" || exit 1

# Default entrypoint runs the turnkey regression suite in fast mode
CMD ["python", "scripts/run_regression_suite.py", "--mode=fast"]
