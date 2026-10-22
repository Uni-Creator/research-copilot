.PHONY: install test lint ui api eval docker map map-check
install:
	pip install -r requirements.txt -r requirements-app.txt
test:
	pytest -q
lint:
	ruff check .
ui:
	streamlit run ui/app.py
api:
	uvicorn copilot.service:create_app --factory --reload --port 8000
eval:
	uv run python -m eval.run_eval --topics eval/datasets/slp.json --out eval/results
docker:
	docker build -t research-copilot .
map:
	uv run python scripts/codebase_map.py --incremental
map-check:
	uv run python scripts/codebase_map.py --check

