from mrcnn.model import MaskRCNN
from mrcnn.Configs import Config

from ..exceptions import LockedException
from ._ModelLock import ModelLock


class ModelWrapper(MaskRCNN):

    def __init__(self, mode: str, config: Config, model_dir: str):
        super().__init__(mode, config, model_dir)
        self.lock = ModelLock()
    
    def detect(self, images: list, verbose=0) -> list:
        if self.lock.locked:
            raise LockedException("MaskRCNN model is already processing, try again later.")
        
        with self.lock:
            return super().detect(images, verbose)
