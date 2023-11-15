.PHONY: tests docs

dependencies: 
	@echo "Initializing Git..."
	git init
	@echo "Installing dependencies..."
	poetry install
	poetry run pre-commit install

env: dependencies
	@echo "Activating virtual environment..."
	poetry shell

tests:
	pytest

docs:
	@echo serving documentation using mkdocs, mkdocs-materials, mkdocstrings
	mkdocs serve
