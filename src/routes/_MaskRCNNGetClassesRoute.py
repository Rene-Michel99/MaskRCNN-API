from ..models import ModelCache


class MaskRCNNGetClassesRoute:

    def __init__(self, model_cache: ModelCache):
        self.model_cache = model_cache

    def process(self):
        if len(self.model_cache.models_config) == 0:
            return {
                "status": "No models in cache"
            }
        
        response = {
            "classes": []
        }
        for config in self.model_cache.models_config.values():
            response["classes"].extend(config.CLASS_NAMES)
        
        response["classes"] = list(set(response["classes"]))
        
        return response
