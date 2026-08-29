FROM python:3.8-slim-buster

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p model && \
    gdown "10jGKxCNeMGCFXdF410lX2oAbar-t_I8S" -O model/model.h5

CMD ["python3", "app.py"]