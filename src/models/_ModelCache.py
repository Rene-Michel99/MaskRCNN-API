from logging import Logger
from mrcnn.model import MaskRCNN
from mrcnn.Configs import Config

from ..exceptions import NotFoundException


WEIGHTS = {
    "alita": "./logs/weights/mask_rcnn_alita.h5",
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


class ModelCache:

    def __init__(self, logger: Logger, log_dir: str) -> None:
        self.cache = {}
        self.logger = logger
        self.log_dir = log_dir
    
    def get_model_based_on_data(self, data: dict) -> MaskRCNN:
        classes = data["classes"]
        self.logger.info("Request inference received for detection for {}".format(classes))
        for key, config in CONFIGS.items():
            found = sum([class_name.lower() == classes[0].lower() for class_name in config.CLASS_NAMES])
            if found:
                self.logger.info(f"Model weights to use is {key}")
                return self.get_model_from_cache(key)
        
        raise NotFoundException("There is no model with classes {}".format(classes))
    
    def get_model_from_cache(self, key) -> MaskRCNN:
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