# Official UniAgent PyTerrier baseline image recipe, extended with query expansion.
# baseline.py, test_baseline.py and german-stopwords.txt are unchanged copies of
# ../retrieval-baseline-pyterrier/ (imported, never edited).
FROM ghcr.io/uniagent-webis-de/uniagent-pyterrier-baseline:dev-0.0.1

WORKDIR /app

ADD german-stopwords.txt test_baseline.py baseline.py /app/
ADD query_expansion.py event_logging.py test_query_expansion.py /app/

RUN PYTHONPATH=. pytest .
