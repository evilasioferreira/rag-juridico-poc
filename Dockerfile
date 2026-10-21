# Imagem para Hugging Face Spaces (SDK Docker). Também roda em qualquer host com Docker.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1

# Spaces executa o contêiner com o usuário 1000.
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

COPY --chown=user requirements.txt pyproject.toml ./
COPY --chown=user src ./src
RUN pip install -r requirements.txt streamlit && pip install -e .

COPY --chown=user . .
USER user

# Fontes, embeddings e índice são gerados no build: o app sobe pronto, sem esperar download.
RUN python -m rag_juridico.cli download && python -m rag_juridico.cli ingest

EXPOSE 7860
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]
