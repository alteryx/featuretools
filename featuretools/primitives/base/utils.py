try:
    # python 3
    from inspect import signature
except ImportError:
    # python 2
    from funcsigs import signature


def inspect_function_args(new_class, function, uses_calc_time):
    # inspect function to see if there are keyword arguments
    kwargs = {}
    args = signature(function).parameters.values()
    for arg in args:
        if arg.name == 'time':
            if not uses_calc_time:
                raise ValueError("'time' is a restricted keyword.  Please"
                                 " use a different keyword.")
            else:
                new_class.uses_calc_time = True
        if arg.default is not arg.empty:
            kwargs[arg.name] = arg.default
    return new_class, kwargs
