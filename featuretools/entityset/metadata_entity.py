import logging

from featuretools.core.base import FTBase

logger = logging.getLogger('featuretools.entityset')


class MetadataEntity(FTBase):
    """
    Stores all actual data for an entity
    """
    indexed_by = None

    def __init__(self, id, metadata_entityset, variables=None, name=None,
                 index=None, time_index=None, secondary_time_index=None,
                 last_time_index=None, encoding=None,
                 created_index=None):
        """ Create Entity

        Args:
            id (str): Id of Entity.
            metadata_entityset (MetadataEntitySet): MetadataEntityset for this MetadataEntity.
            variables (list[:class:`Variable`]) : List of variables which refer to each column in dataframe
            name (str): Name of entity.
            index (str): Name of id column in the dataframe.
            time_index (str): Name of time column in the dataframe.
            secondary_time_index (dict[str -> str]): Dictionary mapping columns
                in the dataframe to the time index column they are associated with.
            last_time_index (pd.Series): Time index of the last event for each
                instance across all child entities.
            encoding (str, optional)) : If None, will use 'ascii'. Another option is 'utf-8',
                or any encoding supported by pandas.
            created_index (bool, optional): Whether originally index was created or provided in dataframe

        """
        self.encoding = encoding
        self.indexed_by = {}
        self.created_index = created_index
        self.attempt_cast_index_to_int(index)
        self.last_time_index = last_time_index
        self.id = id
        self.name = name
        self.metadata_entityset = metadata_entityset
        self.index = index
        self.secondary_time_index = secondary_time_index
        self.time_index = time_index
        self.variables = variables
