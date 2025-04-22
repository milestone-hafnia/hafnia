FROM pytorch/pytorch:latest

RUN pip install torchmetrics tqdm tensorboard

## Hafnia section of the Dockerfile
RUN pip install hafnia

# Copy recipe files in the docker image
ENV RECIPE_DIR=/opt/recipe
COPY src $RECIPE_DIR