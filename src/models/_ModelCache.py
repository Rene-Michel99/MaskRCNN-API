import json
from typing import Dict
from logging import Logger
from mrcnn.Configs import Config

from ..exceptions import NotFoundException, ServiceUnavailableException
from ._ModelWrapper import ModelWrapper
from ._APIConfig import APIConfig


class ModelCache:

    def __init__(self, logger: Logger, api_config: APIConfig) -> None:
        self.cache = {}
        self.logger = logger
        self.api_config = api_config
        self.models_config: Dict[str, Config] = {}
        self.weights = {}
        self.extra_config = {}

        self._load_models_config()
    
    def _load_models_config(self):
        with open("./config.json", "r") as f:
            config = json.loads(f.read())

        self.extra_config = config.get("extra", {})
        for item in config["modelsConfig"]:
            self.models_config[item["name"]] = Config(
                images_per_gpu=item["imagesPerGpu"],
                name=item["name"],
                num_classes=item["numClasses"],
                class_names=item["classNames"],
            )
            self.weights[item["name"]] = "logs/weights/{}".format(item["weights"])
    
    def get_model_based_on_data(self, data: dict) -> ModelWrapper:
        classes = data["classes"]
        self.logger.info("Request inference received for detection for {}".format(classes))
        model_key = ""
        for key, config in self.models_config.items():
            found = sum([class_name.lower() == classes[0].lower() for class_name in config.CLASS_NAMES])
            if found:
                model_key = key
                break
        
        if not model_key:
            raise NotFoundException("There is no model with classes {}".format(classes))
        
        return self._get_available_model(model_key)
    
    def _create_cached_model(self, key: str):
        self.logger.info(f"Creating and caching model for weights {key}")
        if key not in self.cache:
            self.cache[key] = []
        
        model = ModelWrapper(
            mode="inference",
            config=self.models_config.get(key),
            model_dir=self.api_config.log_dir,
            extra_config=self.extra_config.get(key)
        )
        model.load_weights(filepath=self.weights[key], by_name=True)
        self.cache[key].append(model)
        
        return model
    
    def _can_create_new_model(self):
        return sum([len(models) for models in self.cache.values()]) < self.api_config.max_instances_model
    
    def _clean_cache(self):
        if len(self.cache.keys()) == 1:
            self.cache.clear()
        else:
            minor_timestamp = None
            minor_key = None
            model_index = None
            for key, models in self.cache.items():
                if minor_timestamp is None and models:
                    minor_timestamp = models[0].lock.last_time_locked

                for i, model in enumerate(models):
                    if minor_timestamp <= model.lock.last_time_locked and not model.lock.locked:
                        minor_timestamp = model.lock.last_time_locked
                        minor_key = key
                        model_index = i
            
            if minor_key:
                self.cache[minor_key].pop(model_index)
            else:
                raise ServiceUnavailableException("All models are locked and cache can't be cleaned")
    
    def _get_available_model(self, key: str) -> ModelWrapper:
        models = self.cache.get(key, [])
        for model in models:
            if not model.lock.locked:
                return model
        
        if self._can_create_new_model():
            return self._create_cached_model(key)
        
        self.logger.info("Cache limit reached, cleaning and creating new model")
        self._clean_cache()
        return self._create_cached_model(key)
        