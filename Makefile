.PHONY: help install lint format type test test-cov serve infer docker-build docker-up clean

PYTHON ?= python3
UV     ?= uv
APP    ?= src.anpr.api.app:create_app

help:
	@echo "anpr — common dev tasks"
	@echo ""
	@echo "  make install        Install with dev deps via uv"
	@echo "  make lint           Run ruff lint"
	@echo "  make format         Run ruff format"
	@echo "  make type           Run pyright"
	@echo "  make test           Run pytest"
	@echo "  make test-cov       Run pytest with coverage"
	@echo "  make serve          Start FastAPI dev server (uvicorn --reload)"
	@echo "  make infer IMG=p    Run anpr infer on image at IMG"
	@echo "  make docker-build   Build Docker image"
	@echo "  make docker-up      Start docker compose stack"
	@echo "  make clean          Remove caches and build artifacts"

install:
	$(UV) sync --all-extras

lint:
	$(UV) run ruff check src tests scripts

format:
	$(UV) run ruff format src tests scripts
	$(UV) run ruff check --fix src tests scripts

type:
	$(UV) run pyright src tests

test:
	$(UV) run pytest

test-cov:
	$(UV) run pytest --cov=anpr --cov-report=term-missing --cov-report=html

serve:
	$(UV) run uvicorn $(APP) --factory --host 0.0.0.0 --port 8000 --reload

infer:
	$(UV) run anpr infer $(IMG)

docker-build:
	docker build -t anpr:latest .

docker-up:
	docker compose up --build

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache .pyright .coverage htmlcov
	rm -rf build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
