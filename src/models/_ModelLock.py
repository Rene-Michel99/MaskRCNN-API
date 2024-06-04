from datetime import datetime


class ModelLock:
    def __init__(self):
        self.locked = False
        self._last_time_locked = None

    @property
    def last_time_locked(self):
        return self._last_time_locked
    
    @last_time_locked.setter
    def last_time_locked(self, value):
        pass
    
    def __enter__(self):
        self._last_time_locked = datetime.now()
        self.locked = True
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.locked = False
