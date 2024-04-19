
class MaskRCNNStatusRoute(object):
    
    def process(self, cache: dict):
        if len(cache.keys()) == 0:
            return {
                "status": "No models in cache"
            }
        
        response = {}
        for key, model in self.cache.items():
            response[key] = {
                'is_compiled': model.is_compiled,
                'model_dir': model.model_dir,
                'mode': model.mode,
                'weights': model.using_weights,
                'class_names': model.config.CLASS_NAMES
            }
        
        return model
