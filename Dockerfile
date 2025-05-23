FROM python:3.12-slim

# Create non-root user for security
RUN groupadd -r talon && useradd -r -g talon talon

# Install system dependencies
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

# Set working directory
WORKDIR /app

# Copy requirements and setup files first for better Docker layer caching
COPY requirements.txt .
COPY setup.py .
COPY MANIFEST.in .
COPY README.md .

# Install Python dependencies
RUN pip3 install --upgrade pip --no-cache-dir && \
    pip3 install --no-cache-dir -r requirements.txt && \
    pip3 install --no-cache-dir . && \
    apt-get remove -y build-essential libxml2-dev libxslt1-dev libffi-dev && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Copy application code
COPY . .

# Change ownership to non-root user
RUN chown -R talon:talon /app

# Switch to non-root user
USER talon

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python3 -c "import requests; requests.get('http://localhost:5000/health', timeout=10)" || exit 1

# Run application
ENTRYPOINT ["python3"]
CMD ["/app/talon/web/bootstrap.py"]
