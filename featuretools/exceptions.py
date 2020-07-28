class UnknownFeature(Exception):

    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)


class UnusedPrimitiveWarning(Warning):

    def __init__(self, *args, **kwargs):
        Warning.__init__(self, *args, **kwargs)
