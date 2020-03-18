# To run this test:
# 1. Download data from https://www.kaggle.com/c/home-credit-default-risk/data and unzip into data/home-credit-default-risk/ directory
# 2. Run home-credit-gen-data.py to generate addional datafiles
# 3. Run python test_agg.py to run this test

# NOTES
# -Number of partitions can be adjusted by changing the blocksize variable
# -Aggregation primitives can be run indvidually or at the same time, by uncommenting the proper code block near the end of the file
# -The size of the resulting feature matrix can be adjusted by changing the number of dataframes combines with `concat` commands

import os, sys, logging
from datetime import datetime

import pandas as pd
from dask import dataframe as dd
import dask
from dask.distributed import Client

import featuretools as ft
from featuretools.entityset import EntitySet

from memory_profiler import memory_usage

dask_application_file = os.path.join(os.path.dirname(__file__), 'data/home-credit-default-risk/application_train*.csv')
pandas_application_file = os.path.join(os.path.dirname(__file__), 'data/home-credit-default-risk/application_train')
bureau_file = os.path.join(os.path.dirname(__file__), 'data/home-credit-default-risk/bureau.csv')
previous_application_file = os.path.join(os.path.dirname(__file__), 'data/home-credit-default-risk/previous_application.csv')

# DOCKER BUILD COMMAND: docker build -t py .
# DOCKER RUN COMMAND: docker run -it -p 8787:8787 -v /Users/nate.parsons/dev/ft-dask-testing:/app -v /Users/nate.parsons/dev/ft-dask-testing/data:/app/data py python -u /app/test_loan.py
# DOCKER RUN COMMAND: docker run -it -p 8787:8787 -v /Users/nate.parsons/dev/ft-dask-testing:/app -v /Users/nate.parsons/dev/ft-dask-testing/data:/app/data py python -u /app/test_long.py
# DOCKER RUN COMMAND: docker run -it -p 8787:8787 -v /Users/nate.parsons/dev/ft-dask-testing:/app -v /Users/nate.parsons/dev/ft-dask-testing/data:/app/data py python -u /app/test_agg.py

def run_dask(trans_primitives, agg_primitives):
    # CREATE DASK DATAFRAMES
    print("Creating Dask dataframes")
    blocksize = '50MB'
    df1 = dd.read_csv(pandas_application_file + '.csv', blocksize=blocksize)
    df2 = dd.read_csv(pandas_application_file + '_2.csv', blocksize=blocksize)
    df3 = dd.read_csv(pandas_application_file + '_3.csv', blocksize=blocksize)
    df4 = dd.read_csv(pandas_application_file + '_4.csv', blocksize=blocksize)
    df5 = dd.read_csv(pandas_application_file + '_5.csv', blocksize=blocksize)
    df6 = dd.read_csv(pandas_application_file + '_6.csv', blocksize=blocksize)
    df7 = dd.read_csv(pandas_application_file + '_7.csv', blocksize=blocksize)
    df8 = dd.read_csv(pandas_application_file + '_8.csv', blocksize=blocksize)
    df9 = dd.read_csv(pandas_application_file + '_9.csv', blocksize=blocksize)
    application_dd = dd.concat([df1, df2])
    bureau_dd = dd.read_csv(bureau_file, blocksize=blocksize)
    previous_application_dd = dd.read_csv(previous_application_file, blocksize=blocksize)
    print('Application DF npartitions: {}'.format(application_dd.npartitions))
    print('Bureau DF npartitions: {}'.format(bureau_dd.npartitions))
    print('Previous Application DF npartitions: {}'.format(previous_application_dd.npartitions))
    # Create a bool column for testing
    bureau_dd['AMT_CREDIT_SUM_OVERDUE'] = bureau_dd['AMT_CREDIT_SUM_OVERDUE'].astype(bool)
    
    # CREATE DASK ENTITYSET
    print("Creating Dask entityset")
    start_es = datetime.now()
    dask_es = EntitySet(id='es')
    dask_es.entity_from_dataframe(
        entity_id="application",
        dataframe=application_dd,
        index="SK_ID_CURR",
    )
    dask_es.entity_from_dataframe(
        entity_id="bureau",
        dataframe=bureau_dd,
        index="SK_ID_BUREAU",
    )
    dask_es.entity_from_dataframe(
        entity_id="previous_application",
        dataframe=previous_application_dd,
        index="SK_ID_PREV",
    )
    dask_rel1 = ft.Relationship(dask_es["application"]["SK_ID_CURR"],
                        dask_es["bureau"]["SK_ID_CURR"])
    dask_rel2 = ft.Relationship(dask_es["application"]["SK_ID_CURR"],
                        dask_es["previous_application"]["SK_ID_CURR"])
    dask_es = dask_es.add_relationship(dask_rel1)
    dask_es = dask_es.add_relationship(dask_rel2)
    end_es = datetime.now()
    elapsed_es = end_es - start_es
    print(f"Entityset creation completed in {elapsed_es.total_seconds()} seconds")


    # RUN DFS WITH DASK ENTITYSET
    print("Run DFS with Dask entityset")
    start_dask = datetime.now()
    print(start_dask)
    dask_fm, _ = ft.dfs(entityset=dask_es,
                    target_entity="application",
                    trans_primitives=trans_primitives,
                    agg_primitives=agg_primitives,
                    verbose=True)
    end_dask = datetime.now()
    elapsed_dask = end_dask - start_dask
    print(f"Dask DFS completed in {elapsed_dask.total_seconds()} seconds")

    return dask_fm

    # # COMPUTE THE DASK FEATURE MATRIX
    # print("Computing Dask feature matrix")
    # compute_start = datetime.now()
    # computed_fm = dask_fm.compute()
    # compute_end = datetime.now()
    # compute_elapsed = compute_end - compute_start
    # print(f"Dask feature matrix computation complete in {compute_elapsed.total_seconds()} seconds")

    # return dask_fm, computed_fm


def run_pandas(trans_primitives, agg_primitives):
    # READ IN PANDAS DATA
    print("Creating pandas dataframes")
    df1 = pd.read_csv(pandas_application_file + '.csv')
    df2 = pd.read_csv(pandas_application_file + '_2.csv')
    df3 = pd.read_csv(pandas_application_file + '_3.csv')
    df4 = pd.read_csv(pandas_application_file + '_4.csv')
    df5 = pd.read_csv(pandas_application_file + '_5.csv')
    df6 = pd.read_csv(pandas_application_file + '_6.csv')
    df7 = pd.read_csv(pandas_application_file + '_7.csv')
    df8 = pd.read_csv(pandas_application_file + '_8.csv')
    df9 = pd.read_csv(pandas_application_file + '_9.csv')
    application_df = pd.concat([df1, df2])

    bureau_df = pd.read_csv(bureau_file)
    previous_application_df = pd.read_csv(previous_application_file)
    # Create a bool column for testing
    bureau_df['AMT_CREDIT_SUM_OVERDUE'] = bureau_df['AMT_CREDIT_SUM_OVERDUE'].astype(bool)

    # CREATE PANDAS ENTITYSET
    print("Creating pandas entityset")
    es = EntitySet(id='es')
    es.entity_from_dataframe(
        entity_id="application",
        dataframe=application_df,
        index="SK_ID_CURR",
    )
    es.entity_from_dataframe(
        entity_id="bureau",
        dataframe=bureau_df,
        index="SK_ID_BUREAU",
    )
    es.entity_from_dataframe(
        entity_id="previous_application",
        dataframe=previous_application_df,
        index="SK_ID_PREV",
    )
    rel1 = ft.Relationship(es["application"]["SK_ID_CURR"],
                        es["bureau"]["SK_ID_CURR"])
    rel2 = ft.Relationship(es["application"]["SK_ID_CURR"],
                        es["previous_application"]["SK_ID_CURR"])
    es = es.add_relationship(rel1)
    es = es.add_relationship(rel2)

    # RUN DFS WITH PANDAS ENTITYSET
    print("Run DFS with pandas entityset")
    start = datetime.now()
    print(start)
    fm, _ = ft.dfs(entityset=es,
               target_entity="application",
               trans_primitives=trans_primitives,
               agg_primitives=agg_primitives,
               verbose=True)
    end = datetime.now()
    elapsed = end - start
    print(f"Pandas DFS completed in {elapsed.total_seconds()} seconds")

    return fm


# dask_fm, computed_fm = run_dask()
# fm = run_pandas()

# pd.testing.assert_frame_equal(computed_fm.set_index('SK_ID_CURR').loc[fm.index][fm.columns], fm)

if __name__ == '__main__':
    # client = Client(n_workers=4, memory_limit='3GB')
    # client = Client(n_workers=1, threads_per_worker=4, processes=False, memory_limit='10GB')
    # client = Client(dashboard_address="127.0.0.1:8787", processes=False, silence_logs=logging.ERROR)
    # client = Client()
    # print(client)

    # Default primitives set
    agg_primitives = ['min', 'max', 'count', 'sum', 'mean', 'any', 'all', 'num_true']
    agg_primitives = ['min', 'max', 'count']
    # TO TEST FURTHER: std, avg_time_between, num_unique, entropy, median, skew, n_most_common
    trans_primitives = ['cum_sum', 'diff', 'is_weekend', 'year', 'day', 'negate', 'cum_min', 'cum_max', 'absolute']
    trans_primitives = ['cum_sum', 'diff', 'negate']

    # Test each agg primitive individually
    # for agg_primitive in agg_primitives:
    #     print(f"Testing primitive: {agg_primitive}")
    #     print("Generating Dask feature matrix")
    #     dask_fm = run_dask(trans_primitives, [agg_primitive])
    #     print(f"Writing dask feature matrix to CSV: {dask_fm.npartitions} partitions")
    #     start = datetime.now()
    #     output_path = os.path.join(os.path.dirname(__file__), 'data/feature_matrix-*.csv')
    #     dask_fm.to_csv(output_path)
    #     end = datetime.now()
    #     elapsed = end - start
    #     print(f"Write to CSV completed in {elapsed.total_seconds()} seconds")
    #     print("Computing dask feature matrix")
    #     dask_fm = dask_fm.compute()

    #     print("Generating pandas feature matrix")
    #     fm = run_pandas(trans_primitives, [agg_primitive])
    #     print(f"Feature matrix cols: {fm.columns}")
    #     print(fm.head())
    #     print(f"Pandas feature matrix memory usage: {fm.memory_usage().sum()/1000000} MB")
    #     print(f"Feature matrix shape: {fm.shape}")

    #     try:
    #         assert agg_primitive.upper() in "_".join(dask_fm.columns), f"`{agg_primitive.upper()}`` not found in dask features!"
    #         assert agg_primitive.upper() in "_".join(fm.columns), f"`{agg_primitive.upper()}`` not found in dask features!"
    #         pd.testing.assert_frame_equal(dask_fm.set_index('SK_ID_CURR').loc[fm.index][fm.columns], fm)
    #         print("Dataframes are equal")
    #     except:
    #         print("Something didn't work right")
    #         breakpoint()

    # Test all agg primitives at the same time
    print(f"Testing agg primitives: {agg_primitives}")
    print(f"Testing trans primitives: {trans_primitives}")
    print("Generating Dask feature matrix")
    dask_fm = run_dask(trans_primitives, agg_primitives)
    print(f"Writing dask feature matrix to CSV: {dask_fm.npartitions} partitions")
    start = datetime.now()
    output_path = os.path.join(os.path.dirname(__file__), 'data/feature_matrix-*.csv')
    dask_fm.to_csv(output_path)
    end = datetime.now()
    elapsed = end - start
    print(f"Write to CSV completed in {elapsed.total_seconds()} seconds")

    # print("Computing dask feature matrix")
    # dask_fm = dask_fm.compute()

    # dask_usage = memory_usage(run_dask)
    # print(f"Max Dask Memory Usage: {max(dask_usage)}")

    print("Generating pandas feature matrix")
    fm = run_pandas(trans_primitives, agg_primitives)
    print(f"Feature matrix cols: {fm.columns}")
    print(fm.head())
    print(f"Pandas feature matrix memory usage: {fm.memory_usage().sum()/1000000} MB")
    print(f"Feature matrix shape: {fm.shape}")

    # try:
    #     pd.testing.assert_frame_equal(dask_fm.set_index('SK_ID_CURR').loc[fm.index][fm.columns], fm)
    #     print("Dataframes are equal")
    # except:
    #     print("Something didn't work right")
    #     breakpoint()