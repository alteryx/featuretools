# flake8: noqa
import math
import psutil
from datetime import datetime

import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask.distributed import Client

import featuretools as ft
import featuretools.variable_types as vtypes


def run_test():
    print("Available Memory: {} Mb".format(psutil.virtual_memory().available / 1000000))
    try:
        client.close()
    except:
        pass
    client = Client(n_workers=1)
    print(client)

    print("Reading raw data...")
    start = datetime.now()
    blocksize = "40MB"
    # Read in the datasets and replace the anomalous values
    app_train = dd.read_csv('data/home-credit-default-risk/application_train.csv', blocksize=blocksize).replace({365243: np.nan})
    app_test = dd.read_csv('data/home-credit-default-risk/application_test.csv', blocksize=blocksize).replace({365243: np.nan})
    bureau = dd.read_csv('data/home-credit-default-risk/bureau.csv', blocksize=blocksize).replace({365243: np.nan})
    bureau_balance = dd.read_csv('data/home-credit-default-risk/bureau_balance.csv', blocksize=blocksize).replace({365243: np.nan})
    cash = dd.read_csv('data/home-credit-default-risk/POS_CASH_balance.csv', blocksize=blocksize).replace({365243: np.nan})
    credit = dd.read_csv('data/home-credit-default-risk/credit_card_balance.csv', blocksize=blocksize).replace({365243: np.nan})
    previous = dd.read_csv('data/home-credit-default-risk/previous_application.csv', blocksize=blocksize).replace({365243: np.nan})
    installments = dd.read_csv('data/home-credit-default-risk/installments_payments.csv', blocksize=blocksize).replace({365243: np.nan})
    end = datetime.now()
    elapsed = (end - start).total_seconds()
    print("Elapsed time: {} sec".format(elapsed))

    print("Preparing data...")
    start = datetime.now()
    app_test['TARGET'] = np.nan
    app = app_train.append(app_test[app_train.columns])

    for index in ['SK_ID_CURR', 'SK_ID_PREV', 'SK_ID_BUREAU']:
        for dataset in [app, bureau, bureau_balance, cash, credit, previous, installments]:
            if index in list(dataset.columns):
                dataset[index] = dataset[index].fillna(0).astype(np.int64)

    es = ft.EntitySet(id='clients')

    installments = installments.drop(columns=['SK_ID_CURR'])
    credit = credit.drop(columns=['SK_ID_CURR'])
    cash = cash.drop(columns=['SK_ID_CURR'])
    end = datetime.now()
    elapsed = (end - start).total_seconds()
    print("Elapsed time: {} sec".format(elapsed))

    print("Creating entityset...")
    start = datetime.now()
    app_vtypes = {
        'SK_ID_CURR': ft.variable_types.variable.Index,
        'AMT_ANNUITY': ft.variable_types.variable.Numeric,
        'AMT_CREDIT': ft.variable_types.variable.Numeric,
        'AMT_GOODS_PRICE': ft.variable_types.variable.Numeric,
        'AMT_INCOME_TOTAL': ft.variable_types.variable.Numeric,
        'AMT_REQ_CREDIT_BUREAU_DAY': ft.variable_types.variable.Numeric,
        'AMT_REQ_CREDIT_BUREAU_HOUR': ft.variable_types.variable.Numeric,
        'AMT_REQ_CREDIT_BUREAU_MON': ft.variable_types.variable.Numeric,
        'AMT_REQ_CREDIT_BUREAU_QRT': ft.variable_types.variable.Numeric,
        'AMT_REQ_CREDIT_BUREAU_WEEK': ft.variable_types.variable.Numeric,
        'AMT_REQ_CREDIT_BUREAU_YEAR': ft.variable_types.variable.Numeric,
        'APARTMENTS_AVG': ft.variable_types.variable.Numeric,
        'APARTMENTS_MEDI': ft.variable_types.variable.Numeric,
        'APARTMENTS_MODE': ft.variable_types.variable.Numeric,
        'BASEMENTAREA_AVG': ft.variable_types.variable.Numeric,
        'BASEMENTAREA_MEDI': ft.variable_types.variable.Numeric,
        'BASEMENTAREA_MODE': ft.variable_types.variable.Numeric,
        'CNT_CHILDREN': ft.variable_types.variable.Numeric,
        'CNT_FAM_MEMBERS': ft.variable_types.variable.Numeric,
        'CODE_GENDER': ft.variable_types.variable.Categorical,
        'COMMONAREA_AVG': ft.variable_types.variable.Numeric,
        'COMMONAREA_MEDI': ft.variable_types.variable.Numeric,
        'COMMONAREA_MODE': ft.variable_types.variable.Numeric,
        'DAYS_BIRTH': ft.variable_types.variable.Numeric,
        'DAYS_EMPLOYED': ft.variable_types.variable.Numeric,
        'DAYS_ID_PUBLISH': ft.variable_types.variable.Numeric,
        'DAYS_LAST_PHONE_CHANGE': ft.variable_types.variable.Numeric,
        'DAYS_REGISTRATION': ft.variable_types.variable.Numeric,
        'DEF_30_CNT_SOCIAL_CIRCLE': ft.variable_types.variable.Numeric,
        'DEF_60_CNT_SOCIAL_CIRCLE': ft.variable_types.variable.Numeric,
        'ELEVATORS_AVG': ft.variable_types.variable.Numeric,
        'ELEVATORS_MEDI': ft.variable_types.variable.Numeric,
        'ELEVATORS_MODE': ft.variable_types.variable.Numeric,
        'EMERGENCYSTATE_MODE': ft.variable_types.variable.Categorical,
        'ENTRANCES_AVG': ft.variable_types.variable.Numeric,
        'ENTRANCES_MEDI': ft.variable_types.variable.Numeric,
        'ENTRANCES_MODE': ft.variable_types.variable.Numeric,
        'EXT_SOURCE_1': ft.variable_types.variable.Numeric,
        'EXT_SOURCE_2': ft.variable_types.variable.Numeric,
        'EXT_SOURCE_3': ft.variable_types.variable.Numeric,
        'FLAG_CONT_MOBILE': ft.variable_types.variable.Boolean,
        'FLAG_DOCUMENT_10': ft.variable_types.variable.Boolean,
        'FLAG_DOCUMENT_11': ft.variable_types.variable.Boolean,
        'FLAG_DOCUMENT_12': ft.variable_types.variable.Boolean,
        'FLAG_DOCUMENT_13': ft.variable_types.variable.Boolean,
        'FLAG_DOCUMENT_14': ft.variable_types.variable.Boolean,
        'FLAG_DOCUMENT_15': ft.variable_types.variable.Boolean,
        'FLAG_DOCUMENT_16': ft.variable_types.variable.Boolean,
        'FLAG_DOCUMENT_17': ft.variable_types.variable.Boolean,
        'FLAG_DOCUMENT_18': ft.variable_types.variable.Boolean,
        'FLAG_DOCUMENT_19': ft.variable_types.variable.Boolean,
        'FLAG_DOCUMENT_2': ft.variable_types.variable.Boolean,
        'FLAG_DOCUMENT_20': ft.variable_types.variable.Boolean,
        'FLAG_DOCUMENT_21': ft.variable_types.variable.Boolean,
        'FLAG_DOCUMENT_3': ft.variable_types.variable.Boolean,
        'FLAG_DOCUMENT_4': ft.variable_types.variable.Boolean,
        'FLAG_DOCUMENT_5': ft.variable_types.variable.Boolean,
        'FLAG_DOCUMENT_6': ft.variable_types.variable.Boolean,
        'FLAG_DOCUMENT_7': ft.variable_types.variable.Boolean,
        'FLAG_DOCUMENT_8': ft.variable_types.variable.Boolean,
        'FLAG_DOCUMENT_9': ft.variable_types.variable.Boolean,
        'FLAG_EMAIL': ft.variable_types.variable.Boolean,
        'FLAG_EMP_PHONE': ft.variable_types.variable.Boolean,
        'FLAG_MOBIL': ft.variable_types.variable.Boolean,
        'FLAG_OWN_CAR': ft.variable_types.variable.Categorical,
        'FLAG_OWN_REALTY': ft.variable_types.variable.Categorical,
        'FLAG_PHONE': ft.variable_types.variable.Boolean,
        'FLAG_WORK_PHONE': ft.variable_types.variable.Boolean,
        'FLOORSMAX_AVG': ft.variable_types.variable.Numeric,
        'FLOORSMAX_MEDI': ft.variable_types.variable.Numeric,
        'FLOORSMAX_MODE': ft.variable_types.variable.Numeric,
        'FLOORSMIN_AVG': ft.variable_types.variable.Numeric,
        'FLOORSMIN_MEDI': ft.variable_types.variable.Numeric,
        'FLOORSMIN_MODE': ft.variable_types.variable.Numeric,
        'FONDKAPREMONT_MODE': ft.variable_types.variable.Categorical,
        'HOUR_APPR_PROCESS_START': ft.variable_types.variable.Numeric,
        'HOUSETYPE_MODE': ft.variable_types.variable.Categorical,
        'LANDAREA_AVG': ft.variable_types.variable.Numeric,
        'LANDAREA_MEDI': ft.variable_types.variable.Numeric,
        'LANDAREA_MODE': ft.variable_types.variable.Numeric,
        'LIVE_CITY_NOT_WORK_CITY': ft.variable_types.variable.Boolean,
        'LIVE_REGION_NOT_WORK_REGION': ft.variable_types.variable.Boolean,
        'LIVINGAPARTMENTS_AVG': ft.variable_types.variable.Numeric,
        'LIVINGAPARTMENTS_MEDI': ft.variable_types.variable.Numeric,
        'LIVINGAPARTMENTS_MODE': ft.variable_types.variable.Numeric,
        'LIVINGAREA_AVG': ft.variable_types.variable.Numeric,
        'LIVINGAREA_MEDI': ft.variable_types.variable.Numeric,
        'LIVINGAREA_MODE': ft.variable_types.variable.Numeric,
        'NAME_CONTRACT_TYPE': ft.variable_types.variable.Categorical,
        'NAME_EDUCATION_TYPE': ft.variable_types.variable.Categorical,
        'NAME_FAMILY_STATUS': ft.variable_types.variable.Categorical,
        'NAME_HOUSING_TYPE': ft.variable_types.variable.Categorical,
        'NAME_INCOME_TYPE': ft.variable_types.variable.Categorical,
        'NAME_TYPE_SUITE': ft.variable_types.variable.Categorical,
        'NONLIVINGAPARTMENTS_AVG': ft.variable_types.variable.Numeric,
        'NONLIVINGAPARTMENTS_MEDI': ft.variable_types.variable.Numeric,
        'NONLIVINGAPARTMENTS_MODE': ft.variable_types.variable.Numeric,
        'NONLIVINGAREA_AVG': ft.variable_types.variable.Numeric,
        'NONLIVINGAREA_MEDI': ft.variable_types.variable.Numeric,
        'NONLIVINGAREA_MODE': ft.variable_types.variable.Numeric,
        'OBS_30_CNT_SOCIAL_CIRCLE': ft.variable_types.variable.Numeric,
        'OBS_60_CNT_SOCIAL_CIRCLE': ft.variable_types.variable.Numeric,
        'OCCUPATION_TYPE': ft.variable_types.variable.Categorical,
        'ORGANIZATION_TYPE': ft.variable_types.variable.Categorical,
        'OWN_CAR_AGE': ft.variable_types.variable.Numeric,
        'REGION_POPULATION_RELATIVE': ft.variable_types.variable.Numeric,
        'REGION_RATING_CLIENT': ft.variable_types.variable.Numeric,
        'REGION_RATING_CLIENT_W_CITY': ft.variable_types.variable.Numeric,
        'REG_CITY_NOT_LIVE_CITY': ft.variable_types.variable.Boolean,
        'REG_CITY_NOT_WORK_CITY': ft.variable_types.variable.Boolean,
        'REG_REGION_NOT_LIVE_REGION': ft.variable_types.variable.Boolean,
        'REG_REGION_NOT_WORK_REGION': ft.variable_types.variable.Boolean,
        'TARGET': ft.variable_types.variable.Numeric,
        'TOTALAREA_MODE': ft.variable_types.variable.Numeric,
        'WALLSMATERIAL_MODE': ft.variable_types.variable.Categorical,
        'WEEKDAY_APPR_PROCESS_START': ft.variable_types.variable.Categorical,
        'YEARS_BEGINEXPLUATATION_AVG': ft.variable_types.variable.Numeric,
        'YEARS_BEGINEXPLUATATION_MEDI': ft.variable_types.variable.Numeric,
        'YEARS_BEGINEXPLUATATION_MODE': ft.variable_types.variable.Numeric,
        'YEARS_BUILD_AVG': ft.variable_types.variable.Numeric,
        'YEARS_BUILD_MEDI': ft.variable_types.variable.Numeric,
        'YEARS_BUILD_MODE': ft.variable_types.variable.Numeric
    }

    bureau_vtypes = {
        'SK_ID_BUREAU': ft.variable_types.variable.Index,
        'SK_ID_CURR': ft.variable_types.variable.Id,
        'CREDIT_ACTIVE': ft.variable_types.variable.Categorical,
        'CREDIT_CURRENCY': ft.variable_types.variable.Categorical,
        'DAYS_CREDIT': ft.variable_types.variable.Numeric,
        'CREDIT_DAY_OVERDUE': ft.variable_types.variable.Numeric,
        'DAYS_CREDIT_ENDDATE': ft.variable_types.variable.Numeric,
        'DAYS_ENDDATE_FACT': ft.variable_types.variable.Numeric,
        'AMT_CREDIT_MAX_OVERDUE': ft.variable_types.variable.Numeric,
        'CNT_CREDIT_PROLONG': ft.variable_types.variable.Numeric,
        'AMT_CREDIT_SUM': ft.variable_types.variable.Numeric,
        'AMT_CREDIT_SUM_DEBT': ft.variable_types.variable.Numeric,
        'AMT_CREDIT_SUM_LIMIT': ft.variable_types.variable.Numeric,
        'AMT_CREDIT_SUM_OVERDUE': ft.variable_types.variable.Numeric,
        'CREDIT_TYPE': ft.variable_types.variable.Categorical,
        'DAYS_CREDIT_UPDATE': ft.variable_types.variable.Numeric,
        'AMT_ANNUITY': ft.variable_types.variable.Numeric
    }

    previous_vtypes = {
        'SK_ID_PREV': ft.variable_types.variable.Index,
        'SK_ID_CURR': ft.variable_types.variable.Id,
        'NAME_CONTRACT_TYPE': ft.variable_types.variable.Categorical,
        'AMT_ANNUITY': ft.variable_types.variable.Numeric,
        'AMT_APPLICATION': ft.variable_types.variable.Numeric,
        'AMT_CREDIT': ft.variable_types.variable.Numeric,
        'AMT_DOWN_PAYMENT': ft.variable_types.variable.Numeric,
        'AMT_GOODS_PRICE': ft.variable_types.variable.Numeric,
        'WEEKDAY_APPR_PROCESS_START': ft.variable_types.variable.Categorical,
        'HOUR_APPR_PROCESS_START': ft.variable_types.variable.Numeric,
        'FLAG_LAST_APPL_PER_CONTRACT': ft.variable_types.variable.Categorical,
        'NFLAG_LAST_APPL_IN_DAY': ft.variable_types.variable.Boolean,
        'RATE_DOWN_PAYMENT': ft.variable_types.variable.Numeric,
        'RATE_INTEREST_PRIMARY': ft.variable_types.variable.Numeric,
        'RATE_INTEREST_PRIVILEGED': ft.variable_types.variable.Numeric,
        'NAME_CASH_LOAN_PURPOSE': ft.variable_types.variable.Categorical,
        'NAME_CONTRACT_STATUS': ft.variable_types.variable.Categorical,
        'DAYS_DECISION': ft.variable_types.variable.Numeric,
        'NAME_PAYMENT_TYPE': ft.variable_types.variable.Categorical,
        'CODE_REJECT_REASON': ft.variable_types.variable.Categorical,
        'NAME_TYPE_SUITE': ft.variable_types.variable.Categorical,
        'NAME_CLIENT_TYPE': ft.variable_types.variable.Categorical,
        'NAME_GOODS_CATEGORY': ft.variable_types.variable.Categorical,
        'NAME_PORTFOLIO': ft.variable_types.variable.Categorical,
        'NAME_PRODUCT_TYPE': ft.variable_types.variable.Categorical,
        'CHANNEL_TYPE': ft.variable_types.variable.Categorical,
        'SELLERPLACE_AREA': ft.variable_types.variable.Numeric,
        'NAME_SELLER_INDUSTRY': ft.variable_types.variable.Categorical,
        'CNT_PAYMENT': ft.variable_types.variable.Numeric,
        'NAME_YIELD_GROUP': ft.variable_types.variable.Categorical,
        'PRODUCT_COMBINATION': ft.variable_types.variable.Categorical,
        'DAYS_FIRST_DRAWING': ft.variable_types.variable.Numeric,
        'DAYS_FIRST_DUE': ft.variable_types.variable.Numeric,
        'DAYS_LAST_DUE_1ST_VERSION': ft.variable_types.variable.Numeric,
        'DAYS_LAST_DUE': ft.variable_types.variable.Numeric,
        'DAYS_TERMINATION': ft.variable_types.variable.Numeric,
        'NFLAG_INSURED_ON_APPROVAL': ft.variable_types.variable.Numeric
    }

    bureau_balance_vtypes = {
        'bureaubalance_index': ft.variable_types.variable.Index,
        'SK_ID_BUREAU': ft.variable_types.variable.Id,
        'MONTHS_BALANCE': ft.variable_types.variable.Numeric,
        'STATUS': ft.variable_types.variable.Categorical
    }

    cash_vtypes = {
        'cash_index': ft.variable_types.variable.Index,
        'SK_ID_PREV': ft.variable_types.variable.Id,
        'MONTHS_BALANCE': ft.variable_types.variable.Numeric,
        'CNT_INSTALMENT': ft.variable_types.variable.Numeric,
        'CNT_INSTALMENT_FUTURE': ft.variable_types.variable.Numeric,
        'NAME_CONTRACT_STATUS': ft.variable_types.variable.Categorical,
        'SK_DPD': ft.variable_types.variable.Numeric,
        'SK_DPD_DEF': ft.variable_types.variable.Numeric
    }

    installments_vtypes = {
        'installments_index': ft.variable_types.variable.Index,
        'SK_ID_PREV': ft.variable_types.variable.Id,
        'NUM_INSTALMENT_VERSION': ft.variable_types.variable.Numeric,
        'NUM_INSTALMENT_NUMBER': ft.variable_types.variable.Numeric,
        'DAYS_INSTALMENT': ft.variable_types.variable.Numeric,
        'DAYS_ENTRY_PAYMENT': ft.variable_types.variable.Numeric,
        'AMT_INSTALMENT': ft.variable_types.variable.Numeric,
        'AMT_PAYMENT': ft.variable_types.variable.Numeric
    }

    credit_vtypes = {
        'credit_index': ft.variable_types.variable.Index,
        'SK_ID_PREV': ft.variable_types.variable.Id,
        'MONTHS_BALANCE': ft.variable_types.variable.Numeric,
        'AMT_BALANCE': ft.variable_types.variable.Numeric,
        'AMT_CREDIT_LIMIT_ACTUAL': ft.variable_types.variable.Numeric,
        'AMT_DRAWINGS_ATM_CURRENT': ft.variable_types.variable.Numeric,
        'AMT_DRAWINGS_CURRENT': ft.variable_types.variable.Numeric,
        'AMT_DRAWINGS_OTHER_CURRENT': ft.variable_types.variable.Numeric,
        'AMT_DRAWINGS_POS_CURRENT': ft.variable_types.variable.Numeric,
        'AMT_INST_MIN_REGULARITY': ft.variable_types.variable.Numeric,
        'AMT_PAYMENT_CURRENT': ft.variable_types.variable.Numeric,
        'AMT_PAYMENT_TOTAL_CURRENT': ft.variable_types.variable.Numeric,
        'AMT_RECEIVABLE_PRINCIPAL': ft.variable_types.variable.Numeric,
        'AMT_RECIVABLE': ft.variable_types.variable.Numeric,
        'AMT_TOTAL_RECEIVABLE': ft.variable_types.variable.Numeric,
        'CNT_DRAWINGS_ATM_CURRENT': ft.variable_types.variable.Numeric,
        'CNT_DRAWINGS_CURRENT': ft.variable_types.variable.Numeric,
        'CNT_DRAWINGS_OTHER_CURRENT': ft.variable_types.variable.Numeric,
        'CNT_DRAWINGS_POS_CURRENT': ft.variable_types.variable.Numeric,
        'CNT_INSTALMENT_MATURE_CUM': ft.variable_types.variable.Numeric,
        'NAME_CONTRACT_STATUS': ft.variable_types.variable.Categorical,
        'SK_DPD': ft.variable_types.variable.Numeric,
        'SK_DPD_DEF': ft.variable_types.variable.Numeric
    }

    # Entities with a unique index
    es = es.entity_from_dataframe(entity_id='app', dataframe=app, index='SK_ID_CURR',
                                  variable_types=app_vtypes)

    es = es.entity_from_dataframe(entity_id='bureau', dataframe=bureau, index='SK_ID_BUREAU',
                                  variable_types=bureau_vtypes)

    es = es.entity_from_dataframe(entity_id='previous', dataframe=previous, index='SK_ID_PREV',
                                  variable_types=previous_vtypes)

    # Entities that do not have a unique index
    es = es.entity_from_dataframe(entity_id='bureau_balance', dataframe=bureau_balance,
                                  make_index=True, index='bureaubalance_index',
                                  variable_types=bureau_balance_vtypes)

    es = es.entity_from_dataframe(entity_id='cash', dataframe=cash,
                                  make_index=True, index='cash_index',
                                  variable_types=cash_vtypes)

    es = es.entity_from_dataframe(entity_id='installments', dataframe=installments,
                                  make_index=True, index='installments_index',
                                  variable_types=installments_vtypes)

    es = es.entity_from_dataframe(entity_id='credit', dataframe=credit,
                                  make_index=True, index='credit_index',
                                  variable_types=credit_vtypes)

    print(es)
    end = datetime.now()
    elapsed = (end - start).total_seconds()
    print("Elapsed time: {} sec".format(elapsed))

    start = datetime.now()
    print("Adding relationships...")
    # Relationship between app_train and bureau
    r_app_bureau = ft.Relationship(es['app']['SK_ID_CURR'], es['bureau']['SK_ID_CURR'])

    # Relationship between bureau and bureau balance
    r_bureau_balance = ft.Relationship(es['bureau']['SK_ID_BUREAU'], es['bureau_balance']['SK_ID_BUREAU'])

    # Relationship between current app and previous apps
    r_app_previous = ft.Relationship(es['app']['SK_ID_CURR'], es['previous']['SK_ID_CURR'])

    # Relationships between previous apps and cash, installments, and credit
    r_previous_cash = ft.Relationship(es['previous']['SK_ID_PREV'], es['cash']['SK_ID_PREV'])
    r_previous_installments = ft.Relationship(es['previous']['SK_ID_PREV'], es['installments']['SK_ID_PREV'])
    r_previous_credit = ft.Relationship(es['previous']['SK_ID_PREV'], es['credit']['SK_ID_PREV'])

    # Add in the defined relationships
    es = es.add_relationships([r_app_bureau, r_bureau_balance, r_app_previous,
                               r_previous_cash, r_previous_installments, r_previous_credit])
    # Print out the EntitySet
    print(es)
    end = datetime.now()
    elapsed = (end - start).total_seconds()
    print("Elapsed time: {} sec".format(elapsed))

    # Primitives supported by Dask implementation
    agg_primitives =  ["sum", "max", "min", "mean", "count", "percent_true", "num_unique"]
    trans_primitives = ['percentile', 'and']
    agg_primitives =  ["sum", "max", "min", "mean"]
    trans_primitives = ['and']

    print("Running DFS...")
    start = datetime.now()
    cutoff_times = app['SK_ID_CURR'].to_frame().rename(columns={"SK_ID_CURR":"instance_id"})
    cutoff_times["time"] = datetime.now()
    cutoff_times = cutoff_times.compute()
    
    features = ft.dfs(entityset=es, target_entity='app',
                      trans_primitives=trans_primitives,
                      agg_primitives=agg_primitives,
                      where_primitives=[], seed_features=[],
                      max_depth=2, verbose=1, features_only=True, cutoff_time=cutoff_times)

    new_partitions = es['app'].df.npartitions * math.ceil(len(features) / len(es['app'].df.columns))
    print("New Partitions: {}".format(new_partitions))
    es['app'].df = es['app'].df.repartition(npartitions=new_partitions)

    # DFS with specified primitives
    fm, features = ft.dfs(entityset=es, target_entity='app',
                          trans_primitives=trans_primitives,
                          agg_primitives=agg_primitives,
                          where_primitives=[], seed_features=[],
                          max_depth=2, verbose=1, cutoff_time=cutoff_times)
    end = datetime.now()
    elapsed = (end - start).total_seconds()
    print("Elapsed time: {} sec".format(elapsed))

    print("Write fm to csv...")
    start = datetime.now()
    fm.to_csv("dask-fm-test/*.csv")
    end = datetime.now()
    elapsed = (end - start).total_seconds()
    print("Elapsed time: {} sec".format(elapsed))
    return
    print("Computing feature matrix...")
    start = datetime.now()
    fm_computed = fm.compute()
    end = datetime.now()
    elapsed = (end - start).total_seconds()
    print("Elapsed time: {} sec".format(elapsed))
    print("Shape: {}".format(fm_computed.shape))
    print("Memory: {} MB".format(fm_computed.memory_usage().sum() / 1000000))

    print("Partition Ratio:", math.ceil(len(features) / len(es['app'].df.columns)))
    print("Column Ratio:", len(fm_computed.columns) / len(app.columns))
    print("Memory Ratio:", fm_computed.memory_usage().sum() / app.compute().memory_usage().sum())

    client.close()

if __name__ == "__main__":
    run_test()
