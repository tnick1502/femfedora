FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=1 RING_HEALTH_DEPLOYMENT=production
WORKDIR /app
COPY requirements-ring-health-serving.txt /app/
RUN pip install --no-cache-dir -r requirements-ring-health-serving.txt
COPY src/__init__.py /app/src/__init__.py
COPY src/ring_health /app/src/ring_health
COPY models/exported/ring_health/v2 /app/models/exported/ring_health/v2
COPY models/registry/ring_health_v2.json /app/models/registry/ring_health_v2.json
RUN useradd --uid 10001 --create-home appuser
USER 10001
EXPOSE 8788
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8788/health/ready',timeout=3)"
CMD ["python","-m","uvicorn","src.ring_health.api_v2:app","--host","0.0.0.0","--port","8788","--no-access-log"]
