FROM ghcr.io/uniagent-webis-de/uniagent-pyterrier-baseline:dev-0.0.1
WORKDIR /app
COPY german-stopwords.txt german-stopwords.NOTICE test_baseline.py baseline.py /app/
COPY domain_dictionary.json components.py submission.py test_submission.py /app/
COPY neural_reranking.py test_neural_reranking.py requirements-neural.txt /app/
RUN python3 -m pip install --no-cache-dir -r requirements-neural.txt
# Download weights with download_model.py BEFORE building. Runtime loads local files only.
COPY download_model.py /app/download_model.py
RUN python3 /app/download_model.py --output /app/models/qwen3-reranker
RUN PYTHONPATH=. python3 -m pytest .
ENTRYPOINT ["python3", "/app/submission.py"]
