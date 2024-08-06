from mrcnn.model import MaskRCNN
from mrcnn.Configs import Config

from ..exceptions import LockedException
from ._ModelLock import ModelLock
from ._ShapeClassifier import ShapeClassifier


class ModelWrapper(MaskRCNN):

    def __init__(self, mode: str, config: Config, model_dir: str, extra_config: dict=None):
        super().__init__(mode, config, model_dir)
        self.lock = ModelLock()
        self.shape_classifier = None
        
        if extra_config is not None and extra_config["name"] == "ShapeClassifier":
            self.shape_classifier = ShapeClassifier(
                weights_path=f"./logs/weights/{extra_config['weights']}",
                classes=extra_config["classes"],
                filter_by_class_name=extra_config.get("className"),
            )
    
    def detect(self, images: list, verbose=0) -> list:
        if self.lock.locked:
            raise LockedException("MaskRCNN model is already processing, try again later.")
        
        with self.lock:
            return super().detect(images, verbose)
    
    def get_extra_metrics(self, mask, class_name: str) -> dict:
        metrics = {}
        if self.shape_classifier is not None:
            shape = self.shape_classifier.predict(mask, class_name)
            if shape is not None:
                metrics["shape"] = shape
        
        return metrics
