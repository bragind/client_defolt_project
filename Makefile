.PHONY: test build push deploy

test:
	python -m pytest tests/ -v

build:
	docker build -t credit-default-model .

push:
	docker push credit-default-model:latest

deploy:
	ssh user@server "cd /app && ./scripts/deploy.sh"

ci-test:
	python -m pytest tests/ --cov=src --cov-report=xml

format:
	black src/ tests/

lint:
	flake8 src/ tests/