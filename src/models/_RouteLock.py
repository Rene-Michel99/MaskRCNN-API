class RouteLock:
    def __init__(self):
        self.locked = False
    
    def __enter__(self):
        self.locked = True
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.locked = False
