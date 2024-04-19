from logging import Logger

from mrcnn.model import MaskRCNN
from ..exceptions import BADRequestException, NOTFoundException


class MaskRCNNTrainingRoute(object):
    def __init__(self, available_weights: dict, available_configs: dict, logger: Logger):
        self.available_weights = available_weights
        self.available_configs = available_configs
        self.logger = logger
    
    def process(self, model: MaskRCNN, request: dict):
        self._validate_request(request)

        return {}
    
    def _validate_request(self, data: dict):
        pass