import os
import uuid
import json
import logging
from flask import Flask, request
from flask_cors import CORS, cross_origin

from .models import ModelCache
from .routes import MaskRCNNInferenceRoute, MaskRCNNStatusRoute


# create image: docker image build -t maskrcnn-backend:latest .
# docker run -dp 127.0.0.1:8080:8080 maskrcnn-backend:latest
# docker run -it maskrcnn-backend:latest bash   
#cp src/. container_id:/target





class APIServer:
    def __init__(self):
        self.app = Flask(__name__)
        self.cors = CORS(self.app)
        self.log_dir = os.path.join(
            "logs",
            str(uuid.uuid4()),
        )

        self.app.route("/invocations", methods=["POST"])(self.inference)
        self.app.route("/ping", methods=["GET"])(self.status)

        self.logger = logging.getLogger('MaskRCNNBackend')
        self.logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()

        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        self.model_cache = ModelCache(self.logger, self.log_dir)
        self.inference_route = MaskRCNNInferenceRoute(self.logger)
        self.status_route = MaskRCNNStatusRoute()

        self.logger.info("Server is ready!")

    def run(self):
        self.app.run(host="0.0.0.0", port=8080, debug=True)

    @cross_origin()
    def inference(self):
        try:
            data = json.loads(request.data)
            model = self.model_cache.get_model_based_on_data(data)
            return self.inference_route.process(data, model), 200
        except Exception as ex:
            return self._parse_exception(ex)
    
    @cross_origin()
    def status(self):
        try:
            return self.status_route.process(self.model_cache)
        except Exception as ex:
            return self._parse_exception(ex)
    
    def _parse_exception(self, ex: Exception):
        self.logger.exception(ex)
        
        ex_message = ex.message if hasattr(ex, "message") else str(ex)
        error_code = ex.error_code if hasattr(ex, "error_code") else 500
        
        return {"error": ex_message}, error_code


api_server = APIServer()
