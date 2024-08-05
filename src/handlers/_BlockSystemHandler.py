import os

from ..models import ModelCache


class BlockSystemHandler:
    def __init__(self, model_cache: ModelCache):
        self.model_cache = model_cache

    def block(self):
        if os.path.exists("config.json"):
            os.system("rm config.json")
        if os.path.exists("logs/weights"):
            os.system("rm -rf logs/weights")
        
        self.model_cache.clean_cache()
