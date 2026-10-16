# Official UniAgent PyTerrier baseline image recipe, extended with a BM25F field index and description-keyword query expansion.
# baseline.py, test_baseline.py and german-stopwords.txt are unchanged copies of
# ../retrieval-baseline-pyterrier/ (imported, never edited).
FROM ghcr.io/uniagent-webis-de/uniagent-pyterrier-baseline:dev-0.0.1

WORKDIR /app

ADD german-stopwords.txt test_baseline.py baseline.py /app/
ADD bm25f_combined.py event_logging.py test_bm25f_combined.py /app/

RUN PYTHONPATH=. pytest .
