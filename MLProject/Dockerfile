FROM continuumio/miniconda3

WORKDIR /app
COPY conda.yaml .
RUN conda env create -f conda.yaml

SHELL ["conda", "run", "-n", "depression_env", "/bin/bash", "-c"]

COPY . /app

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "depression_env", "python", "modelling.py"]
