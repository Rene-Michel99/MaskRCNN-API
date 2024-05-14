class NotFoundException(Exception):
    def __init__(self, message: str):
        super().__init__(message)

        self.message = message
        self.error_code = 404
