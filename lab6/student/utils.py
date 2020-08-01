# This script file is used as a function wrapper. Function Wrappers are often used when dealing with relatively complicated functions. In our case, the function wrapper is used to check a specific pre-condition: only one argument should be passed to the function.
def arguments_mutually_exclusive(func):
    def wrapper(self, **kwargs):
        if len(kwargs) != 1:
            raise ValueError('only a single argument must be given')
        return func(self, **kwargs)
    return wrapper
