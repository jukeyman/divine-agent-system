# Divine Agent System - Supreme Agentic Orchestrator (SAO)
# Multi-stage Docker build for production deployment

# Stage 1: Base Python environment
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libpq-dev \
    libssl-dev \
    libffi-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r divine && useradd -r -g divine divine

# Set working directory
WORKDIR /app

# Stage 2: Dependencies installation
FROM base as dependencies

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Stage 3: Development environment
FROM dependencies as development

# Install development dependencies
RUN pip install pytest pytest-asyncio pytest-cov black flake8 mypy isort

# Copy source code
COPY . .

# Change ownership to non-root user
RUN chown -R divine:divine /app

# Switch to non-root user
USER divine

# Expose ports
EXPOSE 8000 8001 8080 9090

# Development command
CMD ["python", "-m", "agents", "--dev"]

# Stage 4: Production environment
FROM dependencies as production

# Copy only necessary files
COPY agents/ ./agents/
COPY config.yaml .
COPY setup.py .
COPY README.md .

# Install the package
RUN pip install -e .

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/backups

# Change ownership to non-root user
RUN chown -R divine:divine /app

# Switch to non-root user
USER divine

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose ports
EXPOSE 8000 8001 8080 9090

# Production command
CMD ["python", "-m", "agents", "--production"]

# Stage 5: Testing environment
FROM development as testing

# Copy test files
COPY tests/ ./tests/

# Run tests
RUN python -m pytest tests/ -v --cov=agents --cov-report=html

# Stage 6: Documentation builder
FROM dependencies as docs

# Install documentation dependencies
RUN pip install sphinx sphinx-rtd-theme

# Copy documentation source
COPY docs/ ./docs/
COPY README.md .

# Build documentation
RUN cd docs && make html

# Stage 7: Quantum-enhanced version (experimental)
FROM production as quantum

# Install quantum computing dependencies
RUN pip install qiskit qiskit-aer cirq

# Enable quantum features
ENV DIVINE_AGENT_QUANTUM_ENABLED=true

# Stage 8: Multi-cloud version
FROM production as multicloud

# Install cloud provider SDKs
RUN pip install boto3 azure-storage-blob google-cloud-storage

# Enable multi-cloud features
ENV DIVINE_AGENT_MULTICLOUD_ENABLED=true

# Stage 9: Monitoring-enhanced version
FROM production as monitoring

# Install monitoring dependencies
RUN pip install prometheus-client grafana-api influxdb-client

# Enable advanced monitoring
ENV DIVINE_AGENT_MONITORING_ENHANCED=true

# Expose additional monitoring ports
EXPOSE 3000 9090 8086

# Stage 10: Security-hardened version
FROM production as security

# Install security dependencies
RUN pip install cryptography PyJWT bcrypt

# Enable security features
ENV DIVINE_AGENT_SECURITY_HARDENED=true

# Remove unnecessary packages
RUN apt-get update && apt-get remove -y \
    build-essential \
    git \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Set strict file permissions
RUN chmod -R 750 /app

# Default target is production
FROM production as final

# Labels for metadata
LABEL maintainer="Divine Agent System Team <contact@divineagentsystem.ai>" \
      version="1.0.0" \
      description="Supreme Agentic Orchestrator - Multi-Agent Cloud Mastery System" \
      org.opencontainers.image.title="Divine Agent System" \
      org.opencontainers.image.description="Supreme Agentic Orchestrator" \
      org.opencontainers.image.version="1.0.0" \
      org.opencontainers.image.vendor="Divine Agent System" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.source="https://github.com/divineagentsystem/sao" \
      org.opencontainers.image.documentation="https://divineagentsystem.readthedocs.io/"

# Final configuration
VOLUME ["/app/data", "/app/logs", "/app/backups"]

# Entry point script
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["python", "-m", "agents"]