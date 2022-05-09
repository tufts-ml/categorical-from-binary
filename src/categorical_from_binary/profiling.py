import cProfile
import functools
import pstats
import tempfile


def profile_me(func):
    """
    A function decorator which can be used to profile code in line.
    Results are print to screen.

    The decorator currently eats the functions natural return value, but this could
    be adjusted (for an example, see categorical_from_binary.timing)

    Usage:
        profile_me(my_function)(arguments_of_function)

    Reference:
        https://towardsdatascience.com/how-to-profile-your-code-in-python-e70c834fad89
    """

    @functools.wraps(func)
    def wraps(*args, **kwargs):
        file = tempfile.mktemp()
        profiler = cProfile.Profile()
        profiler.runcall(func, *args, **kwargs)
        profiler.dump_stats(file)
        metrics = pstats.Stats(file)
        metrics.strip_dirs().sort_stats("time").print_stats(100)

    return wraps
