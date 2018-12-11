try:
    # python 3.7 deprecated getargspec
    from inspect import getfullargspec as getargspec
except ImportError:
    # python 2.7 - 3.6 backwards compatibility import
    from inspect import getargspec


def inspect_function_args(new_class, function, uses_calc_time):
    # inspect function to see if there are keyword arguments
    argspec = getargspec(function)
    kwargs = {}
    if argspec.defaults is not None:
        lowest_kwargs_position = len(argspec.args) - len(argspec.defaults)

    for i, arg in enumerate(argspec.args):
        if arg == 'time':
            if not uses_calc_time:
                raise ValueError("'time' is a restricted keyword.  Please"
                                 " use a different keyword.")
            else:
                new_class.uses_calc_time = True
        if argspec.defaults is not None and i >= lowest_kwargs_position:
            kwargs[arg] = argspec.defaults[i - lowest_kwargs_position]
    return new_class, kwargs
