import time

class RetryException(Exception):
    pass

# define the retry decorator
def retry(tries=3, delay_s=0.1, logger="default"):
    assert tries >= 2
    assert isinstance(tries, int)
    def decorator(method):
        def wrapper(*args, **kwargs):
            exp = None
            i = 0
            while True:
                i += 1
                try:
                    return method(*args, **kwargs)
                except Exception as ex:
                    if i >= tries:
                        raise ex
                    exp = ex
                    import traceback
                    import sys
                    exc_type, exc_value, exc_traceback = sys.exc_info()
                    s = traceback.format_exception(exc_type, exc_value, exc_traceback)
                    if logger == "default":
                        print(f"while retrying {method} in {__name__}:")
                        print("".join(s))
                    time.sleep(delay_s)
        return wrapper
    return decorator