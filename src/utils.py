from functools import wraps


def handle_exception(success_code=200, unknow_error_code=500):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs), success_code
            except Exception as ex:
                if hasattr(args[0], "logger"):
                    args[0].logger.exception(ex)
                
                ex_message = ex.message if hasattr(ex, "message") else str(ex)
                error_code = ex.error_code if hasattr(ex, "error_code") else unknow_error_code
                
                return {"error": ex_message}, error_code
        
        return wrapper
    
    return decorator
