.PHONY: install install-dev install-worker test lint format type-check docker-build-api docker-build-worker deploy teardown clean

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

install-worker:
	pip install -e ".[worker]"

install-all:
	pip install -e ".[dev,worker]"

test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --tb=short --cov=src --cov-report=term-missing

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

type-check:
	mypy src/

docker-build-api:
	docker build -f docker/api.Dockerfile -t ml-training-api:latest .

docker-build-worker:
	docker build -f docker/worker.Dockerfile -t ml-training-worker:latest .

docker-build: docker-build-api docker-build-worker

deploy:
	kubectl apply -k k8s/overlays/dev/

teardown:
	kubectl delete -k k8s/overlays/dev/ --ignore-not-found

clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache dist build *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
