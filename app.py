from flask import Flask
from flask_restful import Api

from AIManagers import MaskRCNNManager


# create image: docker image build -t ubuntu:maskrcnn-backend .
# docker run --name maskrcnn_container -d -p 8180:8180 ubuntu:maskrcnn-backend
#cp src/. container_id:/target

# creating the flask app
app = Flask(__name__)
# creating an API object
api = Api(app)

api.add_resource(MaskRCNNManager, '/maskrcnn', endpoint='maskrcnn')


if __name__ == '__main__': 
    app.run(host="0.0.0.0", port=8180, debug = True)
