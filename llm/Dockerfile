FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ .

ENV LECTURES_DIR=/data/lectures
ENV QDRANT_URL=http://qdrant:6333
ENV EMBED_URL=http://embed:8080

CMD ["python", "main.py"]
