import json
import logging
from flask import request
from flask_restful import Resource
from functools import lru_cache

from mrcnn.model import MaskRCNN
from mrcnn.Configs import Config
from routes import MaskRCNNInferenceRoute, MaskRCNNStatusRoute, MaskRCNNTrainingRoute

LAST_MODEL_CONFIG = ""
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

@lru_cache(maxsize=None)
def get_maskrcnn_model(data_source: str):
    correct_config = CONFIGS.get(data_source)

    model = None
    if data_source == "coco":
        model = MaskRCNN(mode="inference", config=correct_config)
        model.load_weights(init_with="coco")
    elif data_source == "alita":
        model = MaskRCNN(mode="inference", config=correct_config)
        model.load_weights(filepath=WEIGHTS[data_source], by_name=True)
    
    return model


class MaskRCNNManager(Resource):
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)

        self.CONFIGS = CONFIGS
        self.WEIGHTS = WEIGHTS

        self.logger = logging.getLogger('MaskRCNNBackend')
        self.logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()

        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        self.post_route = MaskRCNNInferenceRoute(configs=self.CONFIGS, weights=self.WEIGHTS, logger=self.logger)
        self.get_route = MaskRCNNStatusRoute()
        self.put_route = MaskRCNNTrainingRoute(self.WEIGHTS, self.CONFIGS, self.logger)

    # corresponds to the GET request.
    # this function is called whenever there
    # is a GET request for this resource
    def get(self):
        try:
            return self.get_route.process(get_maskrcnn_model(LAST_MODEL_CONFIG))
        except Exception as ex:
            return self._parse_exception(ex)
    
    # Corresponds to POST request 
    def post(self):
        try:
            data = json.loads(request.data)
            model = self._get_model_based_on_data(data)
            return self.post_route.process(data, model), 200, {'Access-Control-Allow-Origin': '*'}
        except Exception as ex:
            return self._parse_exception(ex)
    
    def put(self):
        try:
            data = json.loads(request.data)
            model = self._get_model_based_on_data(data)
            return self.put_route.process(model, data), 201
        except Exception as ex:
            return self._parse_exception(ex)
    
    def _get_model_based_on_data(self, data: dict):
        classes = data["classes"]
        for key, config in CONFIGS.items():
            found = [class_name.lower() == classes[0].lower() for class_name in config.CLASS_NAMES]
            if found:
                LAST_MODEL_CONFIG = key
                return get_maskrcnn_model(key)

    def _parse_exception(self, ex: Exception):
        self.logger.exception(ex)
        
        ex_message = ex.message if hasattr(ex, "message") else str(ex)
        error_code = ex.error_code if hasattr(ex, "error_code") else 500
        
        return {"error": ex_message}, error_code
