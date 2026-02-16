FROM python:3.14-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir --upgrade pip uv

COPY pyproject.toml uv.lock ./
RUN uv sync --no-install-project

COPY . .
RUN uv sync

ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

CMD ["uvicorn", "paper_analysis_deepagents.api:app", "--host", "0.0.0.0", "--port", "8000"]
