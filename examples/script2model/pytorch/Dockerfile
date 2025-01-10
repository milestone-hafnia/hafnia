FROM pytorch/pytorch:latest

RUN pip install torchmetrics tqdm tensorboard

## MDI section of the Dockerfile
RUN MDI_PYPI=98.83.173.60 && \
    pip install mdi-python-tools --index-url http://${MDI_PYPI}/simple/ --trusted-host ${MDI_PYPI}

# Copy recipe files in the docker image
ENV RECIPE_DIR=/opt/recipe
COPY src $RECIPE_DIR
