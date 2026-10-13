# Build the web app, then run the API with ONNX Runtime on CPU.
FROM node:22-slim AS web
WORKDIR /web
COPY package.json package-lock.json build.mjs ./
COPY src ./src
COPY public ./public
RUN npm ci --ignore-scripts && node build.mjs

FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
RUN useradd -m app
WORKDIR /srv
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY backend/app ./app
COPY backend/scripts ./scripts
RUN sh scripts/download_weights.sh
COPY --from=web /web/dist ./static
USER app
ENV ENHANCER_WEIGHTS=/srv/weights ENHANCER_STATIC=/srv/static
EXPOSE 8000
# one worker: each worker holds its own model copy and inference is CPU-bound
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--proxy-headers"]
