FROM eclipse-temurin:11-jre-jammy AS java

FROM python:3.11-slim-bookworm

COPY --from=java /opt/java/openjdk /opt/java/openjdk

ENV JAVA_HOME=/opt/java/openjdk \
    JVM_PATH=/opt/java/openjdk/lib/server/libjvm.so \
    PYTERRIER_HOME=/opt/pyterrier \
    TERRIER_VERSION=5.11 \
    TERRIER_HELPER_VERSION=0.0.8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4

ENV PATH="/opt/java/openjdk/bin:${PATH}"

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
       ca-certificates poppler-utils libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch==2.13.0+cpu"
RUN python -m pip install --no-cache-dir --requirement /app/requirements.txt \
    && python -m pip check

COPY . /app/

RUN chmod +x /app/run.sh \
    && python /app/build_assets.py

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

ENTRYPOINT ["/app/run.sh"]
