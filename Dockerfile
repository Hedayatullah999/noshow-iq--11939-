# Stage 1: Builder
FROM python:3.11-slim AS builder
WORKDIR /build

RUN apt-get update && apt-get install -y gcc python3-dev && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2: Final lightweight image
FROM python:3.11-slim

# Create a non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /home/appuser/app

# Copy installed packages from builder
COPY --from=builder /root/.local /home/appuser/.local

# Copy project files with correct ownership
COPY --chown=appuser:appuser . .

USER appuser

ENV PATH=/home/appuser/.local/bin:$PATH

EXPOSE 8000

CMD ["uvicorn", "noshow_iq.api:app", "--host", "0.0.0.0", "--port", "8000"]