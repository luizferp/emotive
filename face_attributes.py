# Face attributes supported by the MS Face API
ATTR_EMOTION = 'emotion'
ATTR_AGE = 'age'
ATTR_SMILE = 'smile'

class AttrDict(dict):
    """
    An specialization of <dict> that allows accessing dict keys as attributes
    """
    def __init__(self, *args, **kwArgs):
        super(AttrDict, self).__init__(*args, **kwArgs)
        self.__dict__ = self

    def __getattr__(self, arg):
        if arg in self.__dict__:
            return self.get(arg)
        return super(AttrDict, self).__getattribute__(arg)()

class AttrObj(object):
    def __init__(self, value):
        self.value = value

    def __getattr__(self):
        return self.value


class Emotion(AttrDict):
    pass


class Age(AttrObj):
    pass


class Smile(AttrObj):
    pass


ATTRIBUTES = {ATTR_EMOTION: Emotion, ATTR_AGE: Age, ATTR_SMILE: Smile}
