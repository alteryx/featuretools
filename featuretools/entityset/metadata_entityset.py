from featuretools.core.base import FTBase


class MetadataEntitySet(FTBase):
    id = None
    entities = []
    relationships = []
    name = None

    def __init__(self, id, entities=None, relationships=None, time_type=None):
        self.id = id
        self.entities = entities
        self.relationships = relationships
        self.time_type = time_type
