# Compact version of ../retrieval-collection-routed/: the same three systems, routed per collection,
# in two modules (components.py, submission.py). baseline.py, test_baseline.py and german-stopwords.txt
# are unchanged copies of the official ../retrieval-baseline-pyterrier/. No extra packages.
FROM ghcr.io/uniagent-webis-de/uniagent-pyterrier-baseline:dev-0.0.1

WORKDIR /app

ADD german-stopwords.txt test_baseline.py baseline.py /app/
ADD domain_dictionary.json components.py submission.py test_submission.py /app/

RUN PYTHONPATH=. pytest .
