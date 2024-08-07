import os
import json
import logging
from flask_cors import CORS, cross_origin
from flask_swagger_ui import get_swaggerui_blueprint
from flask import Flask, request, send_from_directory
from dotenv import load_dotenv

from .utils import handle_exception
from .models import ModelCache, APIConfig
from .routes import MaskRCNNInferenceRoute, MaskRCNNGetClassesRoute, ConfigRoute, GetWorkersRoute, BlockRoute
from .handlers import BlockSystemHandler
from .services import ZMQClient


# docker image build -t maskrcnn:latest .
# docker run -dp 127.0.0.1:8080:8080 maskrcnn:latest
# docker run -it maskrcnn:latest bash   
#cp src/. container_id:/target


class APIServer:
    def __init__(self, worker_name: str, log_dir: str):
        self.app = Flask(__name__)
        self.cors = CORS(self.app)

        load_dotenv()
        self.port = os.environ.get("SERVER_PORT", 8080)
        
        self.worker_name = worker_name
        self._write_pid(log_dir)
        self.api_config = APIConfig(
            approx_epsilon=4,
            log_dir=log_dir,
            images_dir="./images",
            max_instances_model=int(os.environ.get("MODEL_MAX_QTY", 1)),
            split_images_above_maximum=bool(int(os.environ.get("SPLIT_IMAGES_ABOVE_MAXIMUM", 0)))
        )

        self.logger = self._build_logger(log_dir)
        self.model_cache = ModelCache(self.logger, self.api_config)
        self.block_system_handler = BlockSystemHandler(self.model_cache)
        self.zmq_client = ZMQClient(
            worker_name=self.worker_name,
            block_system=self.block_system_handler,
            logger=self.logger
        )

        self.inference_route = MaskRCNNInferenceRoute(self.logger, self.api_config)
        self.config_route = ConfigRoute(self.api_config, self.logger)
        self.get_classes_route = MaskRCNNGetClassesRoute(self.model_cache)
        self.get_workers_route = GetWorkersRoute(self.api_config.log_dir)
        self.block_route = BlockRoute(zmq_client=self.zmq_client, logger=self.logger)

        self.app.route("/inference", methods=["POST"])(self.inference)
        self.app.route("/classes", methods=["GET"])(self.get_classes)
        self.app.route("/updateConfig", methods=["PUT"])(self.update_config)
        self.app.route("/workers", methods=["GET"])(self.get_workers)
        self.app.route("/block", methods=["POST"])(self.block_system)
        self.app.route("/static/<path:path>")(self.get_static)
        self.app.register_blueprint(
            self._get_swagger_blueprint(),
            url_prefix="/doc"
        )
        self.logger.info(f"{self.worker_name} is ready listening on port {str(self.port)}")
        self.zmq_client.start_listen()
    
    def _write_pid(self, log_dir: str):
        with open(os.path.join(log_dir, "pid"), "w") as f:
            f.write(str(os.getpid()))
    
    def _get_swagger_blueprint(self):
        return get_swaggerui_blueprint(
            '/doc',
            '/static/swagger.yaml',
            config={
                'app_name': "MaskRCNNAPI"
            }
        )
        
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
    @handle_exception()
    def inference(self):
        data = json.loads(request.data)
        model = self.model_cache.get_model_based_on_data(data)
        return self.inference_route.process(data, model)
    
    @cross_origin()
    @handle_exception()
    def get_classes(self):
        return self.get_classes_route.process()
    
    @cross_origin()
    @handle_exception(success_code=203)
    def update_config(self):
        data = json.loads(request.data)
        return self.config_route.process(data)
    
    @cross_origin()
    @handle_exception()
    def get_workers(self):
        return self.get_workers_route.process()
    
    @cross_origin()
    @handle_exception(success_code=203)
    def block_system(self):
        return self.block_route.process()
    
    @cross_origin()
    @handle_exception()
    def get_static(self, path):
        return send_from_directory(path)
