FROM python:3.11-slim
WORKDIR /app

# CPU-only PyTorch keeps the image small (no GPU on the free tier)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

# Download the embedding model at build time so the app starts faster
ENV HF_HOME=/app/.cache
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-small')" \
    && chmod -R 777 /app/.cache

COPY scripts/ scripts/
COPY services/ services/
COPY app/ app/
COPY data/index/kb.parquet data/index/kb_text_emb.npy data/index/novelty.json data/index/
COPY start.sh .
RUN chmod +x start.sh

ENV LOG_DIR=/tmp/logs API_URL=http://localhost:8000 DAILY_LIMIT=200
EXPOSE 7860
CMD ["./start.sh"]
