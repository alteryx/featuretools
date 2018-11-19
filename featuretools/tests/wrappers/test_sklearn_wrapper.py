import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import pickle

from featuretools.demo.mock_customer import load_mock_customer
from featuretools.wrappers.sklearn import DFSTransformer


def test_sklearn_wrapper():

    # Load example data

    es = load_mock_customer(n_customers=15, n_products=15, n_sessions=75,
                                    n_transactions=1000, random_seed=0,
                                    return_entityset=True)

    df = es['customers'].df
    df['target'] = np.random.randint(1,3, df.shape[0])


    # Using with transformers

    pipeline = Pipeline(steps=[
        ('ft', DFSTransformer(entityset=es, target_entity="customers",
                                max_features=2, n_jobs=1)),
        ('sc', StandardScaler()),
    ])

    X_train = pipeline.fit(df['customer_id'].values).transform(
                                                        df['customer_id'].values)

    assert X_train.shape[0] == 15
    assert X_train.shape[1] == 2


    # Using with estimator

    pipeline = Pipeline(steps=[
        ('ft', DFSTransformer(entityset=es, target_entity="customers",
                                max_features=2, n_jobs=1)),
        ('et', ExtraTreesClassifier(n_estimators=100))
    ])

    pipeline.fit(df['customer_id'].values, y=df.target.values).predict(
                                                        df['customer_id'].values)
    result = pipeline.score(df['customer_id'].values, df.target.values)

    assert isinstance(result, (float))


    # Pickling / Unpickling Pipeline

    s = pickle.dumps(pipeline)
    pipe_pickled = pickle.loads(s)
    result = pipe_pickled.score(df['customer_id'].values, df.target.values)
    assert isinstance(result, (float))


    # Using with cross_val_score

    results = cross_val_score(pipeline,
                              X = df['customer_id'].values,
                              y = df.target.values,
                              cv=2,
                              scoring="accuracy")

    assert isinstance(results[0], (float))
    assert isinstance(results[1], (float))


    # Using with GridSearchCV

    params = {
        'et__max_depth': [5, 10]
    }

    grid = GridSearchCV(estimator=pipeline,
                        param_grid=params,
                        cv=3)

    grid.fit(df['customer_id'].values, df.target.values)
    assert len(grid.predict(df['customer_id'].values)) == 15
