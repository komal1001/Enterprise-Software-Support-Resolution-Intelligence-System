FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN useradd -m -u 1000 user
WORKDIR /app

COPY --chown=user requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY --chown=user src ./src
COPY --chown=user Documents ./Documents
COPY --chown=user nodes_cache.json .

USER user
# Render injects PORT at runtime; 8000 is the fallback for local `docker run`
ENV HOME=/home/user PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT}"]
