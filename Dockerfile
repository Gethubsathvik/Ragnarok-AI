# syntax=docker/dockerfile:1
# Change the tag to match your driver if needed (hub.docker.com/r/pytorch/pytorch/tags).
ARG PYTORCH_IMAGE=pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# Stage 1: build the React UI
FROM node:22-alpine AS web
WORKDIR /web
COPY web/frontend/package.json ./
RUN npm install --no-audit --no-fund
COPY web/frontend/ ./
RUN npm run build

# Stage 2: Python + CUDA runtime (PyTorch image already contains the CUDA libs)
FROM ${PYTORCH_IMAGE}
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 MINIGPT_HOST=0.0.0.0 TOKENIZERS_PARALLELISM=false
COPY requirements.txt requirements-web.txt ./
RUN pip install -r requirements.txt -r requirements-web.txt
COPY scripts ./scripts
COPY configs ./configs
COPY web/backend ./web/backend
COPY web/static ./web/static
COPY --from=web /web/dist ./web/frontend/dist
RUN useradd -m app && mkdir -p models data logs && chown -R app /app
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health').status==200 else 1)"
CMD ["python", "web/backend/app.py"]
