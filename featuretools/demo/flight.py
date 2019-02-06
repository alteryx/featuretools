import math
import re
from builtins import str

import pandas as pd
from tqdm import tqdm

import featuretools as ft
import featuretools.variable_types as vtypes


def load_flight(month_filter=None,
                categorical_filter=None,
                nrows=None,
                demo=True,
                return_single_table=False,
                verbose=False):
    '''
    Download, clean, and filter flight data from 2017.
    The original dataset can be found `here <https://www.transtats.bts.gov/DL_SelectFields.asp?DB_Short_Name=On-Time&Table_ID=236>`_.

    Args:

        month_filter (list[int]): Only use data from these months (example is ``[1, 2]``).
            To skip, set to None.
        categorical_filter (dict[str->str]): Use only specified categorical values.
            Example is ``{'dest_city': ['Boston, MA'], 'origin_city': ['Boston, MA']}``
            which returns all flights in OR out of Boston. To skip, set to None.
        nrows (int): Passed to nrows in ``pd.read_csv``. Used before filtering.
        demo (bool): Use only two months of data. If False, use the whole year.
        return_single_table (bool): Exit the function early and return a dataframe.
        verbose (bool): Show a progress bar while loading the data.

    Examples:

        .. ipython::
            :verbatim:

            In [1]: import featuretools as ft

            In [2]: es = ft.demo.load_flight(verbose=True,
               ...:                          month_filter=[1],
               ...:                          categorical_filter={'origin_city':['Boston, MA']})
            100%|xxxxxxxxxxxxxxxxxxxxxxxxx| 100/100 [01:16<00:00,  1.31it/s]

            In [3]: es
            Out[3]:
            Entityset: Flight Data
              Entities:
                airports [Rows: 55, Columns: 3]
                flights [Rows: 613, Columns: 9]
                trip_logs [Rows: 9456, Columns: 22]
                airlines [Rows: 10, Columns: 1]
              Relationships:
                trip_logs.flight_id -> flights.flight_id
                flights.carrier -> airlines.carrier
                flights.dest -> airports.dest
    '''

    key, csv_length = make_flight_pathname(demo=demo)

    print('Downloading data from s3...')
    bucket_name = 'featuretools-static'
    s3_url = "https://s3.amazonaws.com/{}/{}".format(bucket_name, key)

    chunksize = math.ceil(csv_length / 99)
    pd.options.display.max_columns = 200
    iter_csv = pd.read_csv(s3_url,
                           iterator=True,
                           nrows=nrows,
                           chunksize=chunksize)
    if verbose:
        iter_csv = tqdm(iter_csv, total=100)

    partial_df_list = []
    for chunk in iter_csv:
        df = filter_data(_clean_data(chunk),
                         month_filter=month_filter,
                         categorical_filter=categorical_filter)
        partial_df_list.append(df)
    data = pd.concat(partial_df_list)

    if return_single_table:
        return data

    es = make_es(data)

    return es


def make_es(data):
    es = ft.EntitySet('Flight Data')
    labely_columns = ['arr_delay', 'dep_delay', 'carrier_delay', 'weather_delay',
                      'national_airspace_delay', 'security_delay',
                      'late_aircraft_delay', 'cancelled', 'diverted',
                      'taxi_in', 'taxi_out', 'air_time', 'dep_time']

    variable_types = {'flight_num': vtypes.Categorical,
                      'distance_group': vtypes.Ordinal,
                      'cancelled': vtypes.Boolean,
                      'diverted': vtypes.Boolean}

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


def _clean_data(data):

    # Make column names snake case
    clean_data = data.rename(
        columns={col: convert(col) for col in data})

    # Chance crs -> "scheduled" and other minor clarifications
    clean_data = clean_data.rename(columns={'crs_arr_time': 'scheduled_arr_time',
                                            'crs_dep_time': 'scheduled_dep_time',
                                            'crs_elapsed_time': 'scheduled_elapsed_time',
                                            'nas_delay': 'national_airspace_delay',
                                            'origin_city_name': 'origin_city',
                                            'dest_city_name': 'dest_city'})

    # Combine strings like 0130 (1:30 AM) with dates (2017-01-01)
    clean_data['scheduled_dep_time'] = clean_data['scheduled_dep_time'].apply(lambda x: str(x)) + clean_data['flight_date'].astype('str')

    # Parse combined string as a date
    clean_data.loc[:, 'scheduled_dep_time'] = pd.to_datetime(
        clean_data['scheduled_dep_time'], format='%H%M%Y-%m-%d', errors='coerce')

    clean_data['scheduled_elapsed_time'] = pd.to_timedelta(clean_data['scheduled_elapsed_time'], unit='m')

    clean_data = _reconstruct_times(clean_data)

    # Create a time index 6 months before scheduled_dep
    clean_data.loc[:, 'time_index'] = clean_data['scheduled_dep_time'] - \
        pd.Timedelta('120d')

    # A null entry for a delay means no delay
    clean_data = _fill_labels(clean_data)

    # Nulls for scheduled values are too problematic. Remove them.
    clean_data = clean_data.dropna(
        axis='rows', subset=['scheduled_dep_time', 'scheduled_arr_time'])

    # Make a flight id. Define a flight as a combination of:
    # 1. carrier 2. flight number 3. origin airport 4. dest airport
    clean_data.loc[:, 'flight_id'] = clean_data['carrier'] + '-' + \
        clean_data['flight_num'].apply(lambda x: str(x)) + ':' + clean_data['origin'] + '->' + clean_data['dest']

    return clean_data


def _fill_labels(clean_data):
    labely_columns = ['arr_delay', 'dep_delay', 'carrier_delay', 'weather_delay',
                      'national_airspace_delay', 'security_delay',
                      'late_aircraft_delay', 'cancelled', 'diverted',
                      'taxi_in', 'taxi_out', 'air_time']
    for col in labely_columns:
        clean_data.loc[:, col] = clean_data[col].fillna(0)

    return clean_data


def _reconstruct_times(clean_data):
    """ Reconstruct departure_time, scheduled_dep_time,
        arrival_time and scheduled_arr_time by adding known delays
        to known times. We do:
            - dep_time is scheduled_dep + dep_delay
            - arr_time is dep_time + taxiing and air_time
            - scheduled arrival is scheduled_dep + scheduled_elapsed
    """
    clean_data.loc[:, 'dep_time'] = clean_data['scheduled_dep_time'] + \
        pd.to_timedelta(clean_data['dep_delay'], unit='m')

    clean_data.loc[:, 'arr_time'] = clean_data['dep_time'] + \
        pd.to_timedelta(clean_data['taxi_out'] +
                        clean_data['air_time'] +
                        clean_data['taxi_in'], unit='m')

    clean_data.loc[:, 'scheduled_arr_time'] = clean_data['scheduled_dep_time'] + \
        clean_data['scheduled_elapsed_time']
    return clean_data


def filter_data(clean_data,
                month_filter=None,
                categorical_filter=None):
    if month_filter is not None:
        tmp = clean_data['scheduled_dep_time'].dt.month.isin(month_filter)
        clean_data = clean_data[tmp]

    if categorical_filter is not None:
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


def make_flight_pathname(demo=True):
    if demo:
        filename = SMALL_FLIGHT_CSV
        key = 'bots_flight_data_2017/' + filename
        rows = 860457
    else:
        filename = BIG_FLIGHT_CSV
        key = 'bots_flight_data_2017/' + filename
        rows = 5162742

    return key, rows


SMALL_FLIGHT_CSV = 'data_2017_jan_feb.csv.zip'
BIG_FLIGHT_CSV = 'data_all_2017.csv.zip'
