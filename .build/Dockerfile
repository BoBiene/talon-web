FROM python:3.13-slim-buster AS deps

RUN apt-get update && \
    apt-get install -y \
        build-essential \
        curl \
        git \
        libatlas-base-dev \
        libatlas3-base \
        libffi-dev \
        libffi6 \
        liblapack-dev \
        libxml2 \
        libxml2-dev \
        libxslt-dev \
        musl-dev \
        python3-dev \
    && rm -rf /var/lib/apt/lists/*

FROM deps AS testable
ARG REPORT_PATH

VOLUME ["/var/mailgun", "/etc/mailgun/ssl", ${REPORT_PATH}]

COPY . /app
WORKDIR /app
COPY wheel/* /wheel/

RUN mkdir -p ${REPORT_PATH} && \
    python ./setup.py build bdist_wheel -d /wheel && \
    pip install --no-deps /wheel/*

ENTRYPOINT ["/bin/sh", "/app/run_tests.sh"]
