import featuretools as ft
from featuretools.config import config as ft_config
import os
import pandas as pd
import dask.dataframe as dd


def load_flight(entity_id='flight_dataset', nrows=None, force=False):
    '''
    Returns the flight dataset. Publishes the dataset if it has not been
    published before.

    Args:
        entity_id (str):  id of retail dataset on scheduler
    '''
    demo_save_path = make_flight_pathname(nrows)

    es = ft.EntitySet(entity_id)
    csv_s3 = "s3://featuretools-static/raw_flight_data/raw_csv_data/data_*.csv"

    if not os.path.isfile(demo_save_path) or force:
        df = dd.read_csv(csv_s3,
                         parse_dates=["FL_DATE"],
                         blocksize=None,
                         low_memory=False,
                         storage_options={'anon': True})
        if nrows:
            df = df.head(n=nrows, npartitions=-1, compute=False)
        df = df.compute()
        df.to_csv(demo_save_path, index=False)

    df = pd.read_csv(demo_save_path,
                     nrows=nrows,
                     parse_dates=["FL_DATE"])

    df.drop("Unnamed: 27", axis=1, inplace=True)
    df = df.reset_index(drop=True)
    df['trip_id'] = df.index.values

    es.entity_from_dataframe("trips",
                             index='trip_id',
                             dataframe=df,
                             time_index='FL_DATE')

    es.combine_variables("trips", "flight_id", ["UNIQUE_CARRIER", "TAIL_NUM", "FL_NUM"])

    es.normalize_entity(base_entity_id="trips",
                        new_entity_id="flights",
                        index="flight_id",
                        additional_variables=["UNIQUE_CARRIER", "TAIL_NUM", "FL_NUM", "ORIGIN_AIRPORT_ID",
                                              "ORIGIN_CITY_MARKET_ID", "DEST_AIRPORT_ID", "DEST_CITY_MARKET_ID"],
                        make_time_index=True)
    return es


def make_flight_pathname(nrows):
    file_name = 'raw_flight_data_' + str(nrows) + '.csv'
    return os.path.join(ft_config['csv_save_location'], file_name)