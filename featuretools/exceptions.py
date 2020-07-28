class UnknownFeature(Exception):

    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)


class UnusedPrimitiveWarning(UserWarning):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
