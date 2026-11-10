# Ring Health v3 API (research/demo). Build from repo root:
#   docker build -f deployment/ring_health_v3/Dockerfile -t ring-health-v3 .
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
WORKDIR /app
COPY requirements-ring-health-v3-serving.txt .
RUN pip install --no-cache-dir -r requirements-ring-health-v3-serving.txt
COPY src/ring_health_v3 src/ring_health_v3
COPY models/exported/ring_health/v3 models/exported/ring_health/v3
COPY models/registry/ring_health_v3.json models/registry/ring_health_v3.json
RUN touch src/__init__.py && useradd -r -u 10001 app && chown -R app /app
USER app
EXPOSE 8790
CMD ["uvicorn", "src.ring_health_v3.api:app", "--host", "0.0.0.0", "--port", "8790", "--no-access-log", "--workers", "2"]
