# Build from the module root:  docker build -f deployment/Dockerfile -t ring-health-v3 .
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src src
COPY models models
RUN useradd -r -u 10001 app && chown -R app /app
USER app
EXPOSE 8790
CMD ["uvicorn", "src.ring_health_v3.api:app", "--host", "0.0.0.0", "--port", "8790", "--no-access-log", "--workers", "2"]
