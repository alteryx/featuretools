import featuretools as ft
import pandas as pd
import sqlalchemy as sa
from pdb import set_trace as bp

def entity_set_from_sql(id, connection):
    loader = EntitySetLoaderFromSQL(connection=connection, id=id)
    loader.load_all_tables()
    return loader.es

def _find_pk_name(table):
        column_names = table.columns.keys()
        for name in column_names:
            if table.columns[name].primary_key:
                return name
        return None

class EntitySetLoaderFromSQL:
    def __init__(self, id, connection):
        self.connection = connection
        self.metadata = sa.MetaData(connection)
        self.metadata.reflect(bind=self.connection.engine)
        self.table_names = self._get_table_names()
        self.es = ft.EntitySet(id=id)

    def _get_table_names(self):
        return list(self.metadata.tables.keys())

    def _load_table(self, table_name):
        table = sa.Table(table_name, self.metadata, autoload=True)
        index_name = _find_pk_name(table)
        df = pd.read_sql(table.select(), table.metadata.bind)
        return self.es.entity_from_dataframe(
            entity_id=table_name,
            dataframe=df,
            index=index_name
        )
    
    def load_all_tables(self):
        for table_name in self.table_names:
            self.es = self._load_table(table_name)