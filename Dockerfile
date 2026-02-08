# HiveFrame - Bee-Inspired Distributed Data Processing Framework
# Multi-stage build for optimized production image

# Build arguments
ARG PYTHON_VERSION=3.11
ARG HIVEFRAME_VERSION=0.3.0-dev

# Stage 1: Builder
FROM python:${PYTHON_VERSION}-slim AS builder

# Set working directory
WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -e ".[production]"

# Stage 2: Runtime
FROM python:${PYTHON_VERSION}-slim

# Re-declare build args for this stage
ARG HIVEFRAME_VERSION=0.3.0-dev

# Set labels
LABEL maintainer="HiveFrame Contributors"
LABEL description="A bee-inspired distributed data processing framework"
LABEL version="${HIVEFRAME_VERSION}"

# Set working directory
WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY src/ ./src/
COPY examples/ ./examples/

# Set Python path
ENV PYTHONPATH=/app/src:$PYTHONPATH

# Environment variables for HiveFrame configuration
ENV HIVE_EMPLOYED_RATIO=0.5
ENV HIVE_ONLOOKER_RATIO=0.4
ENV HIVE_SCOUT_RATIO=0.1
ENV HIVE_ABANDONMENT_LIMIT=10
ENV HIVE_MAX_CYCLES=100
ENV HIVE_NUM_WORKERS=8

# Expose ports
# 8080: Dashboard HTTP server
# 9090: Prometheus metrics
EXPOSE 8080 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.path.insert(0, '/app/src'); from hiveframe import create_hive; print('healthy')" || exit 1

# Default command: Start Python REPL with hiveframe available
CMD ["python3"]
