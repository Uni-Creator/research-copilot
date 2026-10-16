# One image, two entrypoints:  APP=ui (Streamlit, default)  |  APP=api (FastAPI + SSE)
#   docker build -t research-copilot .                      # with real embeddings (bge-small, CPU)
#   docker build --build-arg WITH_BGE=0 -t research-copilot .   # slim: set COPILOT_EMBEDDER=hash
FROM python:3.11-slim

ARG WITH_BGE=1
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    HF_HOME=/app/.hf \
    COPILOT_CACHE_DIR=/data/cache \
    COPILOT_TRACES_DIR=/data/runs \
    COPILOT_EMBEDDER=bge

WORKDIR /app
COPY requirements.txt requirements-app.txt ./
RUN pip install -r requirements.txt -r requirements-app.txt \
 && if [ "$WITH_BGE" = "1" ]; then \
      pip install torch --index-url https://download.pytorch.org/whl/cpu \
   && pip install sentence-transformers \
   && python -c "from sentence_transformers import SentenceTransformer as S; S('BAAI/bge-small-en-v1.5')"; \
    else echo "slim build: set COPILOT_EMBEDDER=hash"; fi

COPY copilot ./copilot
COPY ui ./ui
COPY eval ./eval
COPY scripts ./scripts

RUN useradd --create-home app && mkdir -p /data && chown -R app /app /data
USER app
VOLUME /data
EXPOSE 7860 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
  CMD python -c "import os,urllib.request as u; p=os.environ.get('PORT') or ('8000' if os.environ.get('APP')=='api' else '7860'); u.urlopen('http://127.0.0.1:%s/%s' % (p, 'health' if os.environ.get('APP')=='api' else '_stcore/health'))" || exit 1
CMD ["sh", "scripts/start.sh"]
