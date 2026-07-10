FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN addgroup --system wavelock \
    && adduser --system --ingroup wavelock wavelock

COPY pyproject.toml README.md ./
COPY wavelock ./wavelock

RUN python -m pip install --upgrade pip \
    && python -m pip install . \
    && mkdir -p /app/ledger /app/commitments \
    && chown -R wavelock:wavelock /app

USER wavelock

EXPOSE 9001
VOLUME ["/app/ledger", "/app/commitments"]

CMD ["wavelockd"]
