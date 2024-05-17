from logging import Logger

from ..models import APIConfig
from ..exceptions import BadRequestException


class ConfigRoute:
    
    def __init__(self, api_config: APIConfig, logger: Logger):
        self.api_config = api_config
        self.logger = logger
    
    def process(self, request: dict):
        self._validate_request(request)
        self.logger.info("Received request to change config {}".format(request))

        for key in request.keys():
            setattr(self.api_config, key, request[key])
        
        self.logger.info("Config changed!")
        return {}
        
    
    def _validate_request(self, request: dict):
        if not request:
            raise BadRequestException("Request is empty")
        
        for key in request.keys():
            has_key = hasattr(self.api_config, key)
            if not has_key:
                raise BadRequestException(f"{key} is not a valid config!")
            if has_key and key.startswith("_"):
                raise BadRequestException(f"{key} is not a valid config!")
            if has_key and hasattr(self.api_config, f"_{key}"):
                raise BadRequestException(f"{key} is not a valid config!")
