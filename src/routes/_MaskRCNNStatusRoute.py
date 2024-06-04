from ..models import ModelCache


class MaskRCNNStatusRoute:
    
    def process(self, model_cache: ModelCache):
        if len(model_cache.cache.keys()) == 0:
            return {
                "status": "No models in cache"
            }
        
        response = {}
        for key, models in model_cache.cache.items():
            response[key] = []
            for model in models:
                last_time_locked = model.lock.last_time_locked
                response[key].append({
                    'isCompiled': model.is_compiled,
                    'modelDir': model.model_dir,
                    'mode': model.mode,
                    'weights': model.using_weights,
                    'locked': model.lock.locked,
                    'lastTimeLocked': last_time_locked.isoformat() if last_time_locked else None,
                    'classNames': model.config.CLASS_NAMES
                })
        
        return response
