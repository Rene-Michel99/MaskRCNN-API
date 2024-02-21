import os
import json
import shutil
import logging
import urllib.request
from flask import Flask, request
from flask_cors import CORS, cross_origin

from mrcnn.model import MaskRCNN
from mrcnn.Configs import Config
from routes import MaskRCNNInferenceRoute, MaskRCNNStatusRoute, MaskRCNNTrainingRoute

# create image: docker image build -t ubuntu:maskrcnn-backend .
# docker run --name maskrcnn_container -d -p 8180:8180 ubuntu:maskrcnn-backend
#cp src/. container_id:/target


WEIGHTS = {
    "alita": "./logs/mask_rcnn_alita_and_poros_0004.h5",
    "coco": "./logs/mask_rcnn_coco.h5",
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

        self.app.route("/maskrcnn", methods=["POST"])(self.inference)
        self.app.route("/maskrcnn/<config>", methods=["GET"])(self.status)

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

        self._download_alita_weights()
        self.logger.info("Server is ready!")

    def run(self):
        self.app.run(host="0.0.0.0", port=8180, debug=True)

    @cross_origin()
    def inference(self):
        try:
            data = json.loads(request.data)
            model = self._get_model_based_on_data(data)
            return self.inference_route.process(data, model), 200
        except Exception as ex:
            return self._parse_exception(ex)
    
    @cross_origin()
    def status(self, config):
        try:
            model = self.cache.get(f"model_{config}")
            return self.status_route.process(model)
        except Exception as ex:
            return self._parse_exception(ex)
    
    def _download_alita_weights(self):
        alita_url_path = "https://github.com/Rene-Michel99/MaskRCNN-API/releases/download/weights/mask_rcnn_alita_and_poros_0004.h5"
        if os.path.exists(WEIGHTS["alita"]):
            return
        
        with urllib.request.urlopen(alita_url_path) as resp, open(WEIGHTS["alita"], 'wb') as out:
            shutil.copyfileobj(resp, out)
        
        self.logger.info("Alita weights downloaded!")
    
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
        if key == "coco":
            model = MaskRCNN(mode="inference", config=correct_config)
            model.load_weights(init_with="coco")
        elif key == "alita":
            model = MaskRCNN(mode="inference", config=correct_config)
            model.load_weights(filepath=WEIGHTS[key], by_name=True)
        
        self.cache[f"model_{key}"] = model
        return model
    
    def _parse_exception(self, ex: Exception):
        self.logger.exception(ex)
        
        ex_message = ex.message if hasattr(ex, "message") else str(ex)
        error_code = ex.error_code if hasattr(ex, "error_code") else 500
        
        return {"error": ex_message}, error_code


if __name__ == '__main__': 
    api_server = APIServer()
    api_server.run()
