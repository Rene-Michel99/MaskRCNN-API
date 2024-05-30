import os
import time
import json
import socket
import logging
import multiprocessing
from flask import Flask, request
from flask_cors import CORS, cross_origin
from dotenv import load_dotenv

from .models import ModelCache, APIConfig
from .routes import MaskRCNNInferenceRoute, MaskRCNNStatusRoute, ConfigRoute, GetWorkersRoute


# docker image build -t maskrcnn-backend:latest .
# docker run -dp 127.0.0.1:8080:8080 maskrcnn-backend:latest
# docker run -it maskrcnn-backend:latest bash   
#cp src/. container_id:/target


class APIServer:
    def __init__(self):
        self.app = Flask(__name__)
        self.cors = CORS(self.app)
        load_dotenv()
        self.port = os.environ.get("SERVER_PORT", 8080)
        
        self.worker_name = self._get_worker_name()
        log_dir = os.path.join(
            os.getcwd(),
            "logs",
            self.worker_name,
        )
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        
        with open(os.path.join(log_dir, "pid"), "w") as f:
            f.write(str(os.getpid()))
        
        self.api_config = APIConfig(
            approx_epsilon=4,
            log_dir=log_dir,
            images_dir="./images",
            max_instances_model=int(os.environ.get("MODEL_MAX_QTY", 1)),
        )

        self.logger = self._build_logger(log_dir)

        self.model_cache = ModelCache(self.logger, self.api_config)

        self.inference_route = MaskRCNNInferenceRoute(self.logger, self.api_config)
        self.config_route = ConfigRoute(self.api_config, self.logger)
        self.status_route = MaskRCNNStatusRoute()
        self.get_workers_route = GetWorkersRoute()

        self.app.route("/inference", methods=["POST"])(self.inference)
        self.app.route("/status", methods=["GET"])(self.status)
        self.app.route("/updateConfig", methods=["PUT"])(self.update_config)
        self.app.route("/workers", methods=["GET"])(self.get_workers)

        self.logger.info(f"{self.worker_name} is ready listening on port {str(self.port)}")
    
    def _get_worker_name(self, timeout=10):
        client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print("Connecting to main server to get id...")
        client_sock.connect(('localhost', 3000))
        worker_name = None
        timer = 0
        while True:    
            try:
                msg = client_sock.recv(1024)
                worker_name = msg.decode()
                break
            except Exception as ex:
                timer += 1
                print(ex)
                time.sleep(1)
            
            if timer >= timeout:
                break
        
        client_sock.close()
        return worker_name
        
    def _build_logger(self, log_dir: str):
        logger = logging.getLogger(self.worker_name)
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()

        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        file_handler = logging.FileHandler(
            filename=os.path.join(log_dir, self.worker_name + ".log"),
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        return logger

    def run(self):
        self.app.run(host="0.0.0.0", port=self.port, debug=False)

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
            return self.status_route.process(self.model_cache), 200
        except Exception as ex:
            return self._parse_exception(ex)
    
    @cross_origin()
    def update_config(self):
        try:
            data = json.loads(request.data)
            return self.config_route.process(data), 203
        except Exception as ex:
            return self._parse_exception(ex)
    
    @cross_origin()
    def get_workers(self):
        try:
            return self.get_workers_route.process(), 200
        except Exception as ex:
            return self._parse_exception(ex)
    
    def _parse_exception(self, ex: Exception):
        self.logger.exception(ex)
        
        ex_message = ex.message if hasattr(ex, "message") else str(ex)
        error_code = ex.error_code if hasattr(ex, "error_code") else 500
        
        return {"error": ex_message}, error_code
