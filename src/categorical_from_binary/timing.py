import time
from functools import wraps
from typing import Callable


def time_me(func: Callable) -> Callable:
    """
    A function decorator.

    Given function foo, do
        foo_with_time=timeit(foo)
    and this new function will return a tuple whose 1st element is the normal function result
    and whose second element is the elapsed time (in seconds)

    Usage:
        def double(x):
            return x*2

        result, time= time_me(double)(3)
        print(f"result={result}, time={time}")
        > result=9, time=9.5367431640625e-07
    """

    @wraps(func)
    def _time_it(*args, **kwargs):
        start = time.time()
        try:
            result = func(*args, **kwargs)
        finally:
            end = time.time()
            elapsed = end - start
            return result, elapsed

    return _time_it
