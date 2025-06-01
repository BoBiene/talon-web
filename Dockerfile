# =============================================================================
# BUILD STAGE - Compilation and Dependency Installation
# =============================================================================
FROM python:3.13-alpine AS builder

# Build Environment Variables
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install system dependencies for build
RUN apk update && \
    apk add --no-cache \
        build-base \
        gfortran \
        git \
        libffi-dev \
        libxml2-dev \
        libxslt-dev \
        linux-headers \
        musl-dev

# Install Python dependencies first (for better Docker layer caching)
COPY requirements.txt setup.py MANIFEST.in README.md ./
RUN pip3 install --upgrade pip && \
    pip3 install --prefix=/opt/venv -r requirements.txt && \
    pip3 install --prefix=/opt/venv .

# Copy source code and precompile
COPY . .
RUN python3 -m compileall -b -x "venv|__pycache__" /app && \
    python3 -c "import sys; sys.path.insert(0, '/opt/venv/lib/python3.13/site-packages'); import talon; import talon.web.bootstrap; print('✓ All modules compiled successfully')"

# =============================================================================
# RUNTIME STAGE - Minimal Production Image
# =============================================================================
FROM python:3.13-alpine AS runtime

# Production Environment Variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=0 \
    PYTHONOPTIMIZE=1 \
    PYTHONPATH=/app:/opt/venv/lib/python3.13/site-packages \
    PRODUCTION=true

# Install only runtime dependencies (much smaller!)
RUN apk update && \
    apk add --no-cache \
        curl \
        libffi \
        libgomp \
        libstdc++ \
        libxml2 \
        libxslt \
        openblas && \
    rm -rf /var/cache/apk/*

# Non-root user for security
RUN addgroup -g 1000 talon && \
    adduser -D -u 1000 -G talon talon

WORKDIR /app

# Copy precompiled dependencies from build stage
COPY --from=builder /opt/venv /opt/venv

# Copy precompiled application code from build stage
COPY --from=builder /app /app

# Assign application code to talon user only
RUN chown -R talon:talon /app

USER talon

EXPOSE 5505

# Docker healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:5505/health || exit 1

ENTRYPOINT ["/opt/venv/bin/gunicorn"]
CMD ["-w", "4", "-b", "0.0.0.0:5505", "talon.web.bootstrap:app"]
