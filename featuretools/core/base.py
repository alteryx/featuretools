import copy
from builtins import object


class FTBase(object):

    def normalize(self, normalizer, remove_entityset=True):
        d = copy.copy(self.__dict__)
        if remove_entityset:
            from featuretools.entityset.entityset import EntitySet
            for k, v in d.items():
                if isinstance(v, EntitySet):
                    d[k] = v.id
        d = {k: normalizer(v) for k, v in d.items()}
        if hasattr(self, 'id'):
            d['id'] = self.id
        return d

    @classmethod
    def denormalize(cls, d, denormalizer=None, entityset=None):
        d = copy.copy(d)
        if denormalizer:
            d = {k: denormalizer(v, denormalizer=denormalizer, entityset=entityset)
                 for k, v in d.items()}

        if entityset and 'entityset' in d:
            d['entityset'] = entityset
        es = object.__new__(cls)
        es.__dict__ = d
        return es
