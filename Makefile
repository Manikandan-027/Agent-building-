.PHONY: install dev test evals lint run worker docker-build docker-up clean

install:
	pip install -e ".[dev]"

test:
	python -m pytest tests/ -q

evals:
	python scripts/run_evals.py

lint:
	ruff check ara colpali_service evals tests scripts

run:
	uvicorn ara.service.app:create_app --factory --host 0.0.0.0 --port 8000 --reload

worker:
	python -m ara.service.worker

colpali-mock:
	COLPALI_MODE=mock uvicorn colpali_service.main:app --host 0.0.0.0 --port 8100

docker-build:
	docker compose build

docker-up:
	docker compose up -d

clean:
	rm -rf .pytest_cache .ruff_cache data/pages *.db
