FROM pytorch/pytorch:latest

RUN pip install torchmetrics tqdm tensorboard

## MDI section of the Dockerfile
RUN pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/hafnia

# Copy recipe files in the docker image
ENV RECIPE_DIR=/opt/recipe
COPY src $RECIPE_DIR