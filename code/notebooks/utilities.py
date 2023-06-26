from functools import wraps
import typing

def prepare_arguments_for_lru_cache(func):
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        new_args = []
        for arg in args:
            if isinstance(arg, typing.Hashable):
                new_args.append(arg)
            elif isinstance(arg, list):
                new_args.append(tuple(arg))
        return func(*new_args, **kwargs)

    return wrapper