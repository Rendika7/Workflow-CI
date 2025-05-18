FROM continuumio/miniconda3

WORKDIR /app

COPY MLProject/conda.yaml .
RUN conda env create -f conda.yaml

ENV PATH /opt/conda/envs/depression_env/bin:$PATH

COPY MLProject/ MLProject/

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "depression_env", "python", "modelling.py", "--data_path", "MLProject/student-depression-dataset_preprocessing.csv"]
