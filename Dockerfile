FROM python:3.12-slim

# Sicherheit: Non-root User
RUN groupadd -r talon && useradd -r -g talon talon

# System-Abhängigkeiten für lxml, numpy, etc.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libatlas3-base \
    libxml2 \
    libxml2-dev \
    libxslt1-dev \
    libffi8 \
    libffi-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /app

# Python-Path setzen, damit das Paket immer gefunden wird
ENV PYTHONPATH=/app

# Nur relevante Dateien für den Build kopieren
COPY requirements.txt setup.py MANIFEST.in README.md ./

# Python-Abhängigkeiten installieren
RUN pip3 install --upgrade pip --no-cache-dir && \
    pip3 install --no-cache-dir -r requirements.txt && \
    pip3 install --no-cache-dir . && \
    apt-get remove -y build-essential libxml2-dev libxslt1-dev libffi-dev && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Restlichen Code kopieren
COPY . .

# Ownership an non-root user übergeben
RUN chown -R talon:talon /app

USER talon

EXPOSE 5505

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python3 -c "import requests; requests.get('http://localhost:5505/health', timeout=10)" || exit 1

ENTRYPOINT ["python3"]
CMD ["/app/talon/web/bootstrap.py"]
