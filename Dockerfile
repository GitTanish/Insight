FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Create unprivileged user and pre-create runtime artifacts directory
RUN useradd --create-home --shell /bin/bash insight \
    && mkdir -p /app/.artifacts \
    && chown -R insight:insight /app

# Install dependencies (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files with ownership to avoid duplicate layer overhead
COPY --chown=insight:insight insight/ ./insight/
COPY --chown=insight:insight webapp/ ./webapp/
COPY --chown=insight:insight tests/ ./tests/
COPY --chown=insight:insight pytest.ini ./
COPY --chown=insight:insight assets/ ./assets/
COPY --chown=insight:insight main.py ui_components.py utils.py style.css README.md HOW_TO_RUN.md ./

USER insight

EXPOSE 8000

# NOTE: single-worker by design. WebSession state (uploaded CSV bytes, profile,
# turns) lives in process memory — run exactly ONE uvicorn worker. Horizontal
# scaling requires externalizing session state (e.g. Redis) first.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz')" || exit 1

CMD ["uvicorn", "webapp.app:app", "--host", "0.0.0.0", "--port", "8000"]
