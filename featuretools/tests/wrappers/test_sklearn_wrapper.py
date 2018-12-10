# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from featuretools.demo.mock_customer import load_mock_customer
from featuretools.wrappers import DFSTransformer


def select_numeric(df):
    return df.select_dtypes(exclude=['object'])


@pytest.fixture(scope='module')
def es():
    es = load_mock_customer(n_customers=15,
                            n_products=15,
                            n_sessions=75,
                            n_transactions=1000,
                            random_seed=0,
                            return_entityset=True)
    return es


@pytest.fixture(scope='module')
def df(es):
    df = es['customers'].df
    df['target'] = np.random.randint(1, 3, df.shape[0])  # 1 or 2 values
    return df


@pytest.fixture(scope='module')
def pipeline(es):
    pipeline = Pipeline(steps=[
        ('ft', DFSTransformer(entityset=es,
                              target_entity="customers",
                              max_features=20)),
        ("numeric", FunctionTransformer(select_numeric, validate=False)),
        ('imp', SimpleImputer()),
        ('et', ExtraTreesClassifier(n_estimators=10))
    ])
    return pipeline


def test_sklearn_transformer(es, df):
    # Using with transformers
    pipeline = Pipeline(steps=[
        ('ft', DFSTransformer(entityset=es,
                              target_entity="customers")),
        ("numeric", FunctionTransformer(select_numeric, validate=False)),
        ('sc', StandardScaler()),
    ])

    X_train = pipeline.fit(df['customer_id']).transform(df['customer_id'])

    assert X_train.shape[0] == 15


def test_sklearn_estimator(df, pipeline):
    # Using with estimator
    pipeline.fit(df['customer_id'].values, y=df.target.values) \
            .predict(df['customer_id'].values)
    result = pipeline.score(df['customer_id'].values, df.target.values)

    assert isinstance(result, (float))

    # Pickling / Unpickling Pipeline
    # TODO fix this
    # s = pickle.dumps(pipeline)
    # pipe_pickled = pickle.loads(s)
    # result = pipe_pickled.score(df['customer_id'].values, df.target.values)
    # assert isinstance(result, (float))


def test_sklearn_cross_val_score(df, pipeline):
    # Using with cross_val_score
    results = cross_val_score(pipeline,
                              X=df['customer_id'].values,
                              y=df.target.values,
                              cv=2,
                              scoring="accuracy")

    assert isinstance(results[0], (float))
    assert isinstance(results[1], (float))


def test_sklearn_gridsearchcv(df, pipeline):
    # Using with GridSearchCV
    params = {
        'et__max_depth': [5, 10]
    }
    grid = GridSearchCV(estimator=pipeline,
                        param_grid=params,
                        cv=3)
    grid.fit(df['customer_id'].values, df.target.values)

    assert len(grid.predict(df['customer_id'].values)) == 15


def test_sklearn_cuttoff(pipeline):
    # Using cuttof_time to filter data
    ct = pd.DataFrame()
    ct['customer_id'] = [1, 2, 3]
    ct['time'] = pd.to_datetime(['2014-1-1 04:00',
                                 '2014-1-1 04:00',
                                 '2014-1-1 04:00'])
    ct['label'] = [True, True, False]

    results = pipeline.fit(ct, y=ct.label).predict(ct)

    assert len(results) == 3
