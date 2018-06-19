import os
import re
from builtins import str

import pandas as pd

from featuretools.config import config as ft_config
import featuretools as ft
import featuretools.variable_types as vtypes


def load_flight(use_cache=True,
                demo=True,
                only_return_data=False,
                month_filter=[1, 2],
                categorical_filter={'dest_city': ['Boston, MA'], 'origin_city': ['Boston, MA']}):
    """ Download, clean, and filter flight data from 2017.
        Input:
            month_filter (list[int]): Only use data from these months. Default is [1, 2].
            filterer (dict[str->str]): Use only specified categorical values.
                Default is {'dest_city': ['Boston, MA'], 'origin_city': ['Boston, MA']}
                which returns all flights in OR out of Boston. To skip, set to None.
            use_cache (bool): Use previously downloaded csv if possible.
            demo (bool): Use only two months of data. If false, use whole year.
            only_return_data (bool): Exit the function early and return a dataframe.


    """
    if demo:
        filename = 'flight_dataset_sample.csv'
        demo_save_path = os.path.join(ft_config['csv_save_location'], filename)
        csv_s3 = 'https://s3.amazonaws.com/featuretools-static/bots_flight_data_2017/data_2017_jan_feb.csv.zip'
    else:
        filename = 'flight_dataset_full.csv'
        demo_save_path = os.path.join(ft_config['csv_save_location'], filename)
        csv_s3 = 'https://s3.amazonaws.com/featuretools-static/bots_flight_data_2017/data_all_2017.csv.zip'

    if not use_cache or not os.path.isfile(demo_save_path):
        print('Downloading data from s3...')
        df = pd.read_csv(csv_s3)
        df.to_csv(demo_save_path, index=False)

    pd.options.display.max_columns = 200
    iter_csv = pd.read_csv(demo_save_path,
                           iterator=True,
                           chunksize=1000)
    print('Cleaning and Filtering data...')
    data = pd.concat([filter_data(clean_data(chunk),
                                  month_filter=month_filter,
                                  categorical_filter=filterer) for chunk in iter_csv])
    if only_return_data:
        return data

    es = make_es(data)

    return es


def make_es(data):
    print('Making EntitySet...')
    es = ft.EntitySet('Flight Data')

    variable_types = {'flight_num': vtypes.Categorical,
                      'distance_group': vtypes.Ordinal,
                      'cancelled': vtypes.Boolean,
                      'diverted': vtypes.Boolean}
    labely_columns = ['ArrDelay', 'DepDelay', 'CarrierDelay', 'WeatherDelay',
                      'NASDelay', 'SecurityDelay', 'LateAircraftDelay', 'Cancelled', 'Diverted',
                      'TaxiIn', 'TaxiOut', 'AirTime']

    es.entity_from_dataframe('trip_logs',
                             data,
                             index='trip_log_id',
                             make_index=True,
                             time_index='time_index',
                             secondary_time_index={'arr_time': labely_columns},
                             variable_types=variable_types)

    es.normalize_entity('trip_logs', 'flights', 'flight_id',
                        additional_variables=['origin', 'origin_city', 'origin_state',
                                              'dest', 'dest_city', 'dest_state',
                                              'distance_group', 'carrier', 'flight_num'])
    es.normalize_entity('flights', 'airlines', 'carrier',
                        make_time_index=False)

    es.normalize_entity('flights', 'airports', 'dest',
                        additional_variables=['dest_city', 'dest_state'],
                        make_time_index=False)
    return es


def clean_data(data):
    # Split columns by when we can know them
    main_info = ['FlightDate', 'Carrier', 'FlightNum', 'Origin',
                 'OriginCityName', 'OriginState', 'Dest', 'DestCityName',
                 'DestState', 'Distance', 'DistanceGroup', 'CRSDepTime', 'CRSElapsedTime']

    labely_columns = ['ArrDelay', 'DepDelay', 'CarrierDelay', 'WeatherDelay',
                      'NASDelay', 'SecurityDelay', 'LateAircraftDelay', 'Cancelled', 'Diverted',
                      'TaxiIn', 'TaxiOut', 'AirTime']

    # Fix times
    clean_data = data[main_info + labely_columns]

    clean_data['CRSDepTime'] = data['CRSDepTime'].apply(lambda x: str(x)) + data['FlightDate'].astype(str)

    clean_data.loc[:, 'CRSDepTime'] = pd.to_datetime(
        clean_data['CRSDepTime'], format='%H%M%Y-%m-%d', errors='coerce')

    clean_data.loc[:, 'DepTime'] = clean_data['CRSDepTime'] + \
        pd.to_timedelta(clean_data['DepDelay'], unit='m')
    clean_data.loc[:, 'ArrTime'] = clean_data['DepTime'] + pd.to_timedelta(
        clean_data['TaxiOut'] + clean_data['AirTime'] + clean_data['TaxiIn'], unit='m')
    clean_data.loc[:, 'CRSArrTime'] = clean_data['CRSDepTime'] + \
        pd.to_timedelta(clean_data['CRSElapsedTime'], unit='m')
    clean_data.loc[:, 'time_index'] = clean_data['DepTime'] - \
        pd.Timedelta('120d')
    # Fix labels: a null entry for a delay means no delay
    for col in labely_columns:
        clean_data.loc[:, col] = clean_data[col].fillna(0)

    clean_data = clean_data.dropna(
        axis='rows', subset=['CRSDepTime', 'CRSArrTime'])

    clean_data = clean_data.rename(
        columns={col: convert(col) for col in clean_data})
    clean_data = clean_data.rename(columns={'crs_arr_time': 'scheduled_arr_time',
                                            'crs_dep_time': 'scheduled_dep_time',
                                            'crs_elapsed_time': 'scheduled_elapsed_time',
                                            'nas_delay': 'national_airspace_delay',
                                            'origin_city_name': 'origin_city',
                                            'dest_city_name': 'dest_city'})
    clean_data.loc[:, 'flight_id'] = clean_data['carrier'] + '-' + \
        clean_data['flight_num'].apply(lambda x: str(x)) + ':' + clean_data['origin'] + '->' + clean_data['dest']

    return clean_data


def filter_data(clean_data,
                month_filter=[1, 2],
                categorical_filter={'origin_city': ['Boston, MA']}):
    if month_filter != []:
        tmp = clean_data['origin']
        tmp = False
        for month in month_filter:
            tmp = tmp | (clean_data['scheduled_dep_time'].apply(lambda x: x.month) == month)
        clean_data = clean_data[tmp]

    if categorical_filter is not None:
        tmp = clean_data['origin']
        tmp = False
        for key, values in categorical_filter.items():
            tmp = tmp | clean_data[key].isin(values)
        clean_data = clean_data[tmp]

    return clean_data


def convert(name):
    # Rename columns to underscore
    # Code via SO https://stackoverflow.com/questions/1175208/elegant-python-function-to-convert-camelcase-to-snake-case
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
