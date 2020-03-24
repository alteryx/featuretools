import os, sys, logging

import pandas as pd
from dask import dataframe as dd
import dask
from dask.distributed import Client

import featuretools as ft


# DOCKER BUILD COMMAND: docker build -t py .
# DOCKER RUN COMMAND: docker run -it -p 8787:8787 -v /Users/nate.parsons/dev/featuretools/featuretools/dask-tests-tmp:/app -v /Users/nate.parsons/dev/featuretools/featuretools/dask-tests-tmp/data:/app/data py python -u /app/test_dask_file_io.py


if __name__ == '__main__':
    client = Client()

    # dask_application_file = os.path.join(os.path.dirname(__file__), 'data/home-credit-default-risk/application_train*.csv')
    dask_application_file = os.path.join(os.path.dirname(__file__), 'data/df-*.csv')
    pandas_application_file = os.path.join(os.path.dirname(__file__), 'data/home-credit-default-risk/application_train')

    num_repeats = 10
    new_cols = 10

    # READ IN PANDAS DATA
    print("Creating pandas dataframes")
    df1 = pd.read_csv(pandas_application_file + '.csv')
    print("Adding {} new columns".format(new_cols))    
    for i in range(new_cols):
        col_name = "new_col_{}".format(i)
        df1[col_name] = i
    print("Single file size: {}".format(df1.shape))
    print("Single file memory usage: {} MB".format(df1.memory_usage().sum()/1000000))
    print("Total estimated size: {} MB".format(df1.memory_usage().sum()/1000000 * num_repeats))


    # print("Creating new input files({} repeats)".format(num_repeats))
    # df = pd.read_csv(pandas_application_file + '.csv')
    # for i in range(num_repeats):
    #     df['SK_ID_CURR'] = df['SK_ID_CURR'] + 1000000 * (i)   
    #     output_path = os.path.join(os.path.dirname(__file__), 'data/df-{}.csv'.format(i))
    #     df.to_csv(output_path)

    print("Creating Dask dataframes: {}x".format(num_repeats))
    application_dd = dd.read_csv(dask_application_file)
    application_dd = client.persist(application_dd)
    from dask.distributed import wait
    wait(application_dd)
    breakpoint()
    print("Adding {} new columns".format(new_cols))    
    for i in range(new_cols):
        col_name = "new_col_{}".format(i)
        application_dd[col_name] = i
    breakpoint()
    print("Writing to CSV")
    output_path = os.path.join(os.path.dirname(__file__), 'data/dask_out-*.csv')
    application_dd.to_csv(output_path)


