.PHONY: dev test lint build up down logs models

# Run locally without a GPU using the tone-generator engine.
dev:
	TTS_ENGINE=fake DATA_DIR=data uvicorn app.main:create_app --factory --reload --port 8000

# Run locally with Chatterbox (needs the engine deps: pip install -r requirements-engine.txt).
dev-gpu:
	DATA_DIR=data uvicorn app.main:create_app --factory --port 8000

test:
	pytest

lint:
	ruff check app tests scripts

build:
	docker compose build

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f app

# Pre-download model weights into ./data/hf-cache.
models:
	docker compose run --rm app python -m scripts.download_models
