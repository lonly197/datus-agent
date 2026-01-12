FROM python:3.12-slim

# Build args / defaults
ARG PYTHONUNBUFFERED=1
ENV PYTHONUNBUFFERED=${PYTHONUNBUFFERED}
ARG APP_HOME=/opt/datus
ENV APP_HOME=${APP_HOME}

WORKDIR ${APP_HOME}

# Install system dependencies needed for Python packages
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential gcc g++ \
       libpq-dev \
       libssl-dev \
       libffi-dev \
       libxml2-dev \
       libxslt-dev \
       zlib1g-dev \
       ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install the package and its runtime dependencies
COPY pyproject.toml ./
COPY datus/ ./datus/
COPY datus/conf/ ./datus/conf/

RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir . \
    && pip install --no-cache-dir datus-mysql datus-starrocks

# Expose service port
EXPOSE 8080

# Default config path (can be overridden by mounting a file or passing env)
ENV DATUS_CONFIG=/root/.datus/conf/agent.yml

ENTRYPOINT ["python","-m","datus.api.server"]
CMD ["--config","/root/.datus/conf/agent.yml","--namespace","test","--workflow","chat_agentic","--port","8080","--host","0.0.0.0"]