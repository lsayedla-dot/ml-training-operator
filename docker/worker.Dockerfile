FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src/ src/

RUN pip install --no-cache-dir ".[worker]" \
    --extra-index-url https://download.pytorch.org/whl/cpu

ENTRYPOINT ["python", "-m", "src.worker.train"]
