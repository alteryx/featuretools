import pandas as pd
import sqlalchemy as sa

import featuretools as ft


def from_sql(id, connection):
    loader = EntitySetLoaderFromSQL(connection=connection, id=id)
    loader.load_all_tables()
    loader.load_relationships()
    return loader.es


def _find_pk_name(table):
    column_names = table.columns.keys()
    pks = [name for name in column_names if table.columns[name].primary_key]
    if len(pks) > 1:
        raise RuntimeError('Composite primary key detected')
    elif len(pks) == 1:
        return pks[0]
    else:
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
        make_index = index_name is None
        if make_index:
            index_name = 'featuretools_sql_import_id'
        df = pd.read_sql(table.select(), table.metadata.bind)
        return self.es.entity_from_dataframe(
            entity_id=table_name,
            dataframe=df,
            index=index_name,
            make_index=make_index
        )

    def _column_fk_list(self, table, column_name):
        fks = table.columns[column_name].foreign_keys
        return list(filter(lambda fk: fk, fks))

    def _load_table_fks(self, table_name):
        table = sa.Table(table_name, self.metadata, autoload=True)
        column_names = table.columns.keys()
        fks = {column_name: self._column_fk_list(table, column_name) for column_name in column_names}
        return fks

    def load_relationships(self):
        for table_name in self.table_names:
            fk_dict = self._load_table_fks(table_name)
            child_table = table_name
            for column_name, fks in fk_dict.items():
                child_column = column_name
                child_table = table_name
                for fk in fks:
                    parent_column = fk.column.name
                    parent_table = fk.column.table.name
                    self._add_relationship(parent_table, parent_column, child_table, child_column)

    def load_all_tables(self):
        for table_name in self.table_names:
            self.es = self._load_table(table_name)

    def _add_relationship(self, parent_table, parent_column, child_table, child_column):
        r = ft.Relationship(self.es[parent_table][parent_column],
                            self.es[child_table][child_column])
        self.es = self.es.add_relationship(r)
