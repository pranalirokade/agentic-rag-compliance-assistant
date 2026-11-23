.PHONY: install install-full ingest test eval serve
install:
	pip install -r requirements.txt
install-full:
	pip install -r requirements-full.txt
ingest:
	PYTHONPATH=src python -m rag.cli ingest --docs data/sample_docs --out index
test:
	PYTHONPATH=src pytest -q
eval:
	PYTHONPATH=src python eval/eval_retrieval.py
serve:
	PYTHONPATH=src uvicorn app.main:app --reload
