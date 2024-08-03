class APIConfig:

    def __init__(
            self,
            approx_epsilon: int,
            log_dir: str,
            images_dir: str,
            max_instances_model: int,
            split_images_above_maximum: bool
    ) -> None:

        self.approx_epsilon = approx_epsilon
        self.split_images_above_maximum = split_images_above_maximum
        self._log_dir = log_dir
        self._images_dir = images_dir
        self._max_instances_model = max_instances_model
    
    @property
    def images_dir(self):
        return self._images_dir
    
    @property
    def log_dir(self):
        return self._log_dir
    
    @property
    def max_instances_model(self):
        return self._max_instances_model
    
    @max_instances_model.setter
    def max_instances_model(self, value):
        pass
    
    @log_dir.setter
    def log_dir(self, value):
        pass

    @images_dir.setter
    def images_dir(self, value):
        pass
