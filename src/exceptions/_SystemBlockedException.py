class SystemBlockedException(Exception):
    def __init__(self):
        super().__init__("Server is blocked or not configured")

        self.message = "Server is blocked or not configured"
        self.error_code = 403