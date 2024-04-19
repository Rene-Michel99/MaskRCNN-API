FROM ubuntu:18.04
FROM python:3.9
EXPOSE 8080

RUN mkdir /app

COPY ./src /app/src
COPY nginx.conf /app/nginx.conf
COPY serve.py /app/serve.py
COPY wsgi.py /app/wsgi.py
RUN mkdir /app/src/logs

RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get install -y git wget nginx ca-certificates
RUN apt-get install ffmpeg libsm6 libxext6 -y -y

RUN git clone -b develop https://github.com/Rene-Michel99/Mask-RCNN-TF2.8
RUN pip install ./Mask-RCNN-TF2.8
RUN pip install -U Flask
RUN pip install flask-restful
RUN pip install scikit-learn
RUN pip install flask-cors
RUN pip install gevent gunicorn

ENV PATH="/app:${PATH}"

WORKDIR /app

CMD ["python3", "serve.py"]