FROM pytorch/pytorch:latest

RUN pip install torchmetrics tqdm tensorboard

# Copy recipe files in the docker image
ENV RECIPE_DIR=/opt/recipe
COPY src $RECIPE_DIR
