# Makefile for RLHF Phi-3 Pipeline

.PHONY: help install install-dev test test-unit test-property test-integration test-fast test-slow lint format format-check security coverage clean validate setup ci

# Default target
help:
	@echo "Available targets:"
	@echo "  setup          - Validate project setup and install pre-commit hooks"
	@echo "  install        - Install package and dependencies"
	@echo "  install-dev    - Install package with development dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  test           - Run all tests with coverage"
	@echo "  test-unit      - Run unit tests only"
	@echo "  test-property  - Run property-based tests only"
	@echo "  test-integration - Run integration tests only"
	@echo "  test-fast      - Run fast tests (excluding slow, gpu, network)"
	@echo "  test-slow      - Run slow integration tests"
	@echo "  coverage       - Generate detailed coverage report"
	@echo ""
	@echo "Code Quality:"
	@echo "  lint           - Run linting checks (black, isort, flake8, mypy)"
	@echo "  format         - Format code with black and isort"
	@echo "  format-check   - Check code formatting without making changes"
	@echo "  security       - Run security checks (safety, bandit)"
	@echo ""
	@echo "CI/CD:"
	@echo "  ci             - Run complete CI pipeline"
	@echo "  clean          - Clean build artifacts and cache"
	@echo "  validate       - Validate setup"

# Setup and installation
setup: validate
	@echo "Setting up development environment..."
	pre-commit install
	@echo "Development environment ready!"

validate:
	python3 validate_setup.py

install:
	pip install -r requirements.txt
	pip install -e .

install-dev:
	pip install -r requirements.txt
	pip install -e ".[dev]"
	pip install pre-commit
	pre-commit install

# Testing targets using the test runner
test:
	python run_tests.py all --verbose

test-unit:
	python run_tests.py unit --verbose

test-property:
	python run_tests.py property --verbose

test-integration:
	python run_tests.py integration --verbose

test-fast:
	python run_tests.py fast --verbose

test-slow:
	python run_tests.py integration --include-slow --verbose

coverage:
	python run_tests.py coverage

# Alternative pytest targets (for direct pytest usage)
pytest-unit:
	pytest tests/unit/ -v -m "unit" --cov=rlhf_phi3 --cov-report=term-missing

pytest-property:
	pytest tests/property/ -v -m "property" --tb=short

pytest-integration:
	pytest tests/integration/ -v -m "integration and not slow" --tb=short

pytest-all:
	pytest tests/ -v --cov=rlhf_phi3 --cov-report=html --cov-report=term-missing --cov-fail-under=90

# Code quality targets
lint:
	python run_tests.py lint

format:
	python run_tests.py format

format-check:
	black --check --diff rlhf_phi3/ tests/
	isort --check-only --diff rlhf_phi3/ tests/

security:
	python run_tests.py security

# Individual linting tools
black-check:
	black --check --diff rlhf_phi3/ tests/

black-fix:
	black rlhf_phi3/ tests/

isort-check:
	isort --check-only --diff rlhf_phi3/ tests/

isort-fix:
	isort rlhf_phi3/ tests/

flake8:
	flake8 rlhf_phi3/ tests/

mypy:
	mypy rlhf_phi3/

bandit:
	bandit -r rlhf_phi3/ -f json || true

safety:
	safety check --json || true

# CI/CD targets
ci:
	python run_tests.py ci

# Tox targets (if tox is available)
tox-test:
	tox -e py39

tox-lint:
	tox -e lint

tox-coverage:
	tox -e coverage

tox-all:
	tox

# Cleanup targets
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .tox/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf coverage.xml
	rm -rf .mypy_cache/
	rm -rf .hypothesis/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

clean-all: clean
	rm -rf .venv/
	rm -rf venv/
	rm -rf env/

# Development workflow targets
dev-setup: install-dev setup
	@echo "Development environment ready!"

dev-test: format lint test-fast
	@echo "Development tests completed!"

pre-commit: format lint test-fast
	@echo "Pre-commit checks completed!"

# Documentation targets
docs-check:
	python -c "
	import re
	with open('README.md') as f:
	    content = f.read()
	required_sections = ['Installation', 'Usage', 'Testing']
	for section in required_sections:
	    if section not in content:
	        print(f'Warning: README missing {section} section')
	"

# Package building
build:
	python -m build

build-check: build
	twine check dist/*

# Release targets
release-test: ci build-check
	@echo "Release checks completed!"

# Environment targets
env-create:
	python -m venv .venv
	.venv/bin/pip install --upgrade pip

env-activate:
	@echo "Run: source .venv/bin/activate"

# Jupyter notebook targets (if notebooks exist)
notebook-clean:
	find notebooks/ -name "*.ipynb" -exec jupyter nbconvert --clear-output --inplace {} \; 2>/dev/null || true

notebook-test:
	find notebooks/ -name "*.ipynb" -exec jupyter nbconvert --execute --to notebook --inplace {} \; 2>/dev/null || true

# Performance profiling (if needed)
profile-tests:
	pytest tests/unit/ --profile-svg

# Database/cache cleanup (if applicable)
cache-clear:
	rm -rf ~/.cache/huggingface/
	rm -rf ~/.cache/torch/
	rm -rf .pytest_cache/

# Help for specific targets
help-testing:
	@echo "Testing Commands:"
	@echo "  make test           - Run all tests with coverage"
	@echo "  make test-unit      - Run unit tests only"
	@echo "  make test-property  - Run property-based tests"
	@echo "  make test-fast      - Run fast tests (no slow/gpu/network)"
	@echo "  make coverage       - Generate HTML coverage report"

help-quality:
	@echo "Code Quality Commands:"
	@echo "  make lint           - Run all linting checks"
	@echo "  make format         - Auto-format code"
	@echo "  make format-check   - Check formatting without changes"
	@echo "  make security       - Run security scans"