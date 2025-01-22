ARG PYTHON_VERSION
FROM python:${PYTHON_VERSION}-slim
WORKDIR /opt/ml/processing

COPY dist/*.whl .
RUN pip install --no-cache-dir *.whl && \
    rm -rf *.whl

RUN mkdir -p /opt/ml/processing/input \
             /opt/ml/processing/output \
             /opt/ml/processing/tmp