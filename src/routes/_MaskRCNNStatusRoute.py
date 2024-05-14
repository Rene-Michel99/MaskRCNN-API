from ..models import ModelCache


class MaskRCNNStatusRoute:
    
    def process(self, model_cache: ModelCache):
        if len(model_cache.cache.keys()) == 0:
            return {
                "status": "No models in cache"
            }
        
        response = {}
        for key, model in model_cache.cache.items():
            response[key] = {
                'is_compiled': model.is_compiled,
                'model_dir': model.model_dir,
                'mode': model.mode,
                'weights': model.using_weights,
                'class_names': model.config.CLASS_NAMES
            }
        
        return model
