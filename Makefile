.PHONY: tests docs


tests:
	pytest

docs:
	@echo serving documentation using mkdocs, mkdocs-materials, mkdocstrings
	mkdocs serve
