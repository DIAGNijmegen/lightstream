.PHONY: tests docs


tests:
	pytest

docs:
	@echo serving documentation using mkdocs, mkdocs-materials, mkdocstrings
	pip install mkdocs mkdocstrings[python] mkdocs-material
	mkdocs serve
