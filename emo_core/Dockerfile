FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 POETRY_VIRTUALENVS_CREATE=false
WORKDIR /app
RUN pip install --no-cache-dir poetry==2.1.3
COPY pyproject.toml poetry.lock ./
RUN poetry install --only main --no-root --no-interaction
COPY . .
RUN poetry install --only-root --no-interaction && useradd --uid 10001 --create-home emo && chown -R emo:emo /app
USER emo
EXPOSE 8000
CMD ["uvicorn", "emo_core.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
