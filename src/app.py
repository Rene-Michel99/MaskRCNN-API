import os
import uuid
import json
import shutil
import logging
import urllib.request
from flask import Flask, request
from flask_cors import CORS, cross_origin

from mrcnn.model import MaskRCNN
from mrcnn.Configs import Config
from .routes import MaskRCNNInferenceRoute, MaskRCNNStatusRoute, MaskRCNNTrainingRoute

# create image: docker image build -t ubuntu:maskrcnn-backend .
# docker run --name maskrcnn_container -d -p 8080:8080 ubuntu:maskrcnn-backend
# docker run -it ubuntu:maskrcnn-backend bash   
#cp src/. container_id:/target


WEIGHTS = {
    "alita": "./logs/weights/mask_rcnn_alita_and_poros_0004.h5",
    "coco": "./logs/weights/mask_rcnn_coco.h5",
}
CONFIGS = {
    "alita": Config(images_per_gpu=1, name="alita", num_classes=3, class_names=['BG', 'Idiomorfica', 'Subdiomorfica', 'Xenomorfica']),
    "coco": Config(images_per_gpu=1, name="coco", num_classes=80, class_names=[
        'BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
        'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
        'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
        'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
        'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard',
        'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
        'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
        'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
        'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
        'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
        'teddy bear', 'hair drier', 'toothbrush'
    ]),
}


class APIServer:
    def __init__(self):
        self.app = Flask(__name__)
        self.cache = {}
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

        self.inference_route = MaskRCNNInferenceRoute(configs=CONFIGS, weights=WEIGHTS, logger=self.logger)
        self.status_route = MaskRCNNStatusRoute()
        self.training_route = MaskRCNNTrainingRoute(WEIGHTS, CONFIGS, self.logger)

        self.logger.info("Server is ready!")

    def run(self):
        self.app.run(host="0.0.0.0", port=8080, debug=True)

    @cross_origin()
    def inference(self):
        try:
            data = json.loads(request.data)
            model = self._get_model_based_on_data(data)
            return self.inference_route.process(data, model), 200
        except Exception as ex:
            return self._parse_exception(ex)
    
    @cross_origin()
    def status(self):
        try:
            return self.status_route.process(self.cache)
        except Exception as ex:
            return self._parse_exception(ex)
    
    def _download_weights(self):
        if not os.path.exists("./logs"):
            os.system("mkdir ./logs")
            os.system(f"mkdir {self.log_dir}")
            os.system(f"mkdir logs/weights")
        
        for key in WEIGHTS.keys():
            if os.path.exists(WEIGHTS[key]):
                continue
        
            with urllib.request.urlopen(WEIGHTS_URLS[key]) as resp, open(WEIGHTS[key], 'wb') as out:
                shutil.copyfileobj(resp, out)
        
        self.logger.info("Weights downloaded!")
    
    def _get_model_based_on_data(self, data: dict):
        classes = data["classes"]
        self.logger.info("Request inference received for detection for {}".format(classes))
        for key, config in CONFIGS.items():
            found = sum([class_name.lower() == classes[0].lower() for class_name in config.CLASS_NAMES])
            if found:
                self.cache["last_model_config"] = key
                self.logger.info(f"Model weights to use is {key}")
                return self._get_model_from_cache(key)
    
    def _get_model_from_cache(self, key):
        model = self.cache.get(f"model_{key}")
        if model is not None:
            self.logger.info(f"Model loaded from cache {key}")
            return model
        
        self.logger.info(f"Creating and caching model for weights {key}")
        correct_config = CONFIGS.get(key)
        model = MaskRCNN(mode="inference", config=correct_config, model_dir=self.log_dir)
        model.load_weights(filepath=WEIGHTS[key], by_name=True)
        
        self.cache[f"model_{key}"] = model
        return model
    
    def _parse_exception(self, ex: Exception):
        self.logger.exception(ex)
        
        ex_message = ex.message if hasattr(ex, "message") else str(ex)
        error_code = ex.error_code if hasattr(ex, "error_code") else 500
        
        return {"error": ex_message}, error_code


api_server = APIServer()
#api_server.run()
