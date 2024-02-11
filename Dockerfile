FROM ubuntu:18.04
FROM python:3.9
EXPOSE 8180

RUN mkdir /app
WORKDIR /app

COPY app.py ./app.py
COPY ./routes ./routes
COPY ./exceptions ./exceptions
RUN mkdir logs

RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get install -y git
RUN apt-get install ffmpeg libsm6 libxext6 -y -y

RUN git clone -b develop https://github.com/Rene-Michel99/Mask-RCNN-TF2.8
RUN mv ./Mask-RCNN-TF2.8/* ./
RUN rm -rf ./Mask-RCNN-TF2.8
RUN pip install -r requirements_sagemaker
RUN pip install -U Flask
RUN pip install flask-restful
RUN pip install scikit-learn
RUN pip install flask-cors

CMD ["python3", "app.py"]