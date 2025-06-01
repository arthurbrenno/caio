.PHONY: run

DEFAULT_GOAL := run

run:
	uv run streamlit run ui.py
