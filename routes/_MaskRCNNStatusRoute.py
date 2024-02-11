from mrcnn.model import MaskRCNN


class MaskRCNNStatusRoute(object):
    
    def process(self, model: MaskRCNN):
        if model is not None:
            return {
                'is_compiled': model.is_compiled,
                'model_dir': model.model_dir,
                'mode': model.mode,
                'weights': model.using_weights,
                'class_names': model.config.CLASS_NAMES
            }
        else:
            return {
                "is_compiled": False,
                "model_dil": None,
                "mode": None,
                "weights": None,
                "class_names": []
            }
