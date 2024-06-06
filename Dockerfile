FROM ubuntu:18.04
FROM python:3.9
EXPOSE 8080

RUN mkdir /app

COPY ./src /app/src
COPY ./static /app/src/static
COPY .env /app/.env
COPY config.json /app/config.json
COPY nginx.conf /app/nginx.conf
COPY serve.py /app/serve.py
COPY wsgi.py /app/wsgi.py
COPY requirements.txt /app/requirements.txt
RUN mkdir /app/src/logs

RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get install -y git wget nginx ca-certificates
RUN apt-get install ffmpeg libsm6 libxext6 -y -y

RUN pip install --upgrade pip
RUN git clone -b develop https://github.com/Rene-Michel99/Mask-RCNN-TF2.8
RUN pip install ./Mask-RCNN-TF2.8
RUN pip install -r /app/requirements.txt

ENV PATH="/app:${PATH}"

WORKDIR /app

ENTRYPOINT ["python3.9", "/app/serve.py"]