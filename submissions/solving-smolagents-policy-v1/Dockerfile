FROM python:3.11-slim

RUN apt-get update \
    && apt-get install --yes --no-install-recommends poppler-utils \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir --requirement /requirements.txt

COPY event_logging.py case_tools.py predict.py /
RUN chmod +x /predict.py

ENTRYPOINT ["/predict.py"]
