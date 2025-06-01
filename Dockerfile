# =============================================================================
# BUILD STAGE - Kompilierung und Dependency Installation
# =============================================================================
FROM python:3.13-alpine AS builder

# Build Environment Variables
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# System-Dependencies für Build installieren
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

# Python Dependencies zuerst (für besseres Docker Layer Caching)
COPY requirements.txt setup.py MANIFEST.in README.md ./
RUN pip3 install --upgrade pip && \
    pip3 install --prefix=/opt/venv -r requirements.txt && \
    pip3 install --prefix=/opt/venv .

# Source Code kopieren und vorkompilieren
COPY . .
RUN python3 -m compileall -b /app/talon && \
    python3 -m compileall -b /app/tests && \
    find /app -name "*.py" -not -path "*/venv/*" -not -path "*/__pycache__/*" -exec python3 -m py_compile {} \; && \
    python3 -c "import sys; sys.path.insert(0, '/opt/venv/lib/python3.13/site-packages'); import talon; import talon.web.bootstrap; print('✓ All modules compiled successfully')"

# =============================================================================
# RUNTIME STAGE - Minimales Production Image
# =============================================================================
FROM python:3.13-alpine AS runtime

# Production Environment Variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=0 \
    PYTHONOPTIMIZE=1 \
    PYTHONPATH=/app:/opt/venv/lib/python3.13/site-packages

# Nur Runtime-Dependencies installieren (viel kleiner!)
RUN apk update && \
    apk add --no-cache \
        libffi \
        libgomp \
        libstdc++ \
        libxml2 \
        libxslt \
        openblas && \
    rm -rf /var/cache/apk/*

# Non-root User für Sicherheit
RUN addgroup -g 1000 talon && \
    adduser -D -u 1000 -G talon talon

WORKDIR /app

# Kopiere vorkompilierte Dependencies aus Build Stage
COPY --from=builder /opt/venv /opt/venv

# Kopiere vorkompilierten Application Code aus Build Stage
COPY --from=builder /app /app

# Nur Anwendungscode zu talon User zuweisen
RUN chown -R talon:talon /app

USER talon

EXPOSE 5505

# Optimierter Healthcheck (nutzt das interne Python)
#HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
#    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:5505/health', timeout=10)" || exit 1

ENTRYPOINT ["python3"]
CMD ["/app/talon/web/bootstrap.py"]
