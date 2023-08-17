<p align="center">
<img width=50% src="https://www.featuretools.com/wp-content/uploads/2017/12/FeatureLabs-Logo-Tangerine-800.png" alt="Featuretools" />
</p>
<p align="center">
<i>"One of the holy grails of machine learning is to automate more and more of the feature engineering process."</i> ― Pedro Domingos, <a href="https://bit.ly/things_to_know_ml">A Few Useful Things to Know about Machine Learning</a>
</p>

<p align="center">
    <a href="https://github.com/alteryx/featuretools/actions/workflows/tests_with_latest_deps.yaml" alt="Tests" target="_blank">
        <img src="https://github.com/alteryx/featuretools/actions/workflows/tests_with_latest_deps.yaml/badge.svg?branch=main" alt="Tests" />
    </a>
    <a href="https://codecov.io/gh/alteryx/featuretools">
        <img src="https://codecov.io/gh/alteryx/featuretools/branch/main/graph/badge.svg"/>
    </a>
    <a href='https://featuretools.alteryx.com/en/stable/?badge=stable'>
        <img src='https://readthedocs.com/projects/feature-labs-inc-featuretools/badge/?version=stable' alt='Documentation Status' />
    </a>
    <a href="https://badge.fury.io/py/featuretools" target="_blank">
        <img src="https://badge.fury.io/py/featuretools.svg?maxAge=2592000" alt="PyPI Version" />
    </a>
    <a href="https://anaconda.org/conda-forge/featuretools" target="_blank">
        <img src="https://anaconda.org/conda-forge/featuretools/badges/version.svg" alt="Anaconda Version" />
    </a>
    <a href="https://stackoverflow.com/questions/tagged/featuretools" target="_blank">
        <img src="http://img.shields.io/badge/questions-on_stackoverflow-blue.svg" alt="StackOverflow" />
    </a>
    <a href="https://pepy.tech/project/featuretools" target="_blank">
        <img src="https://static.pepy.tech/badge/featuretools/month" alt="PyPI Downloads" />
    </a>
</p>
<hr>

[Featuretools](https://www.featuretools.com) is a python library for automated feature engineering. See the [documentation](https://docs.featuretools.com) for more information.

## Installation
Install with pip

```
python -m pip install featuretools
```

or from the Conda-forge channel on [conda](https://anaconda.org/conda-forge/featuretools):

```
conda install -c conda-forge featuretools
```

### Add-ons

You can install add-ons individually or all at once by running

```
python -m pip install "featuretools[complete]"
```

**Update checker** - Receive automatic notifications of new Featuretools releases

```
python -m pip install "featuretools[updater]"
```

**Premium Primitives** - Use Premium Primitives, including Natural Language Processing primitives:

```
python -m pip install "featuretools[premium]"
```

**TSFresh Primitives** - Use 60+ primitives from [tsfresh](https://tsfresh.readthedocs.io/en/latest/) within Featuretools

```
python -m pip install "featuretools[tsfresh]"
```

**Dask Support** - Use Dask Dataframes to create EntitySets or run DFS with njobs > 1

```
python -m pip install "featuretools[dask]"
```

**SQL** - Automatic EntitySet generation from relational data stored in a SQL database: 

```
python -m pip install "featuretools[sql]"
```
## Example
Below is an example of using Deep Feature Synthesis (DFS) to perform automated feature engineering. In this example, we apply DFS to a multi-table dataset consisting of timestamped customer transactions.

```python
>> import featuretools as ft
>> es = ft.demo.load_mock_customer(return_entityset=True)
>> es.plot()
```

<img src="https://github.com/alteryx/featuretools/blob/main/docs/source/_static/images/entity_set.png?raw=true" width="350">

Featuretools can automatically create a single table of features for any "target dataframe"
```python
>> feature_matrix, features_defs = ft.dfs(entityset=es, target_dataframe_name="customers")
>> feature_matrix.head(5)
```

```
            zip_code  COUNT(transactions)  COUNT(sessions)  SUM(transactions.amount) MODE(sessions.device)  MIN(transactions.amount)  MAX(transactions.amount)  YEAR(join_date)  SKEW(transactions.amount)  DAY(join_date)                   ...                     SUM(sessions.MIN(transactions.amount))  MAX(sessions.SKEW(transactions.amount))  MAX(sessions.MIN(transactions.amount))  SUM(sessions.MEAN(transactions.amount))  STD(sessions.SUM(transactions.amount))  STD(sessions.MEAN(transactions.amount))  SKEW(sessions.MEAN(transactions.amount))  STD(sessions.MAX(transactions.amount))  NUM_UNIQUE(sessions.DAY(session_start))  MIN(sessions.SKEW(transactions.amount))
customer_id                                                                                                                                                                                                                                  ...
1              60091                  131               10                  10236.77               desktop                      5.60                    149.95             2008                   0.070041               1                   ...                                                     169.77                                 0.610052                                   41.95                               791.976505                              175.939423                                 9.299023                                 -0.377150                                5.857976                                        1                                -0.395358
2              02139                  122                8                   9118.81                mobile                      5.81                    149.15             2008                   0.028647              20                   ...                                                     114.85                                 0.492531                                   42.96                               596.243506                              230.333502                                10.925037                                  0.962350                                7.420480                                        1                                -0.470007
3              02139                   78                5                   5758.24               desktop                      6.78                    147.73             2008                   0.070814              10                   ...                                                      64.98                                 0.645728                                   21.77                               369.770121                              471.048551                                 9.819148                                 -0.244976                               12.537259                                        1                                -0.630425
4              60091                  111                8                   8205.28               desktop                      5.73                    149.56             2008                   0.087986              30                   ...                                                      83.53                                 0.516262                                   17.27                               584.673126                              322.883448                                13.065436                                 -0.548969                               12.738488                                        1                                -0.497169
5              02139                   58                4                   4571.37                tablet                      5.91                    148.17             2008                   0.085883              19                   ...                                                      73.09                                 0.830112                                   27.46                               313.448942                              198.522508                                 8.950528                                  0.098885                                5.599228                                        1                                -0.396571

[5 rows x 69 columns]
```
We now have a feature vector for each customer that can be used for machine learning. See the [documentation on Deep Feature Synthesis](https://featuretools.alteryx.com/en/stable/getting_started/afe.html) for more examples.

Featuretools contains many different types of built-in primitives for creating features. If the primitive you need is not included, Featuretools also allows you to [define your own custom primitives](https://featuretools.alteryx.com/en/stable/getting_started/primitives.html#defining-custom-primitives).

## Demos
**Predict Next Purchase**

[Repository](https://github.com/alteryx/open_source_demos/blob/main/predict-next-purchase/) | [Notebook](https://github.com/alteryx/open_source_demos/blob/main/predict-next-purchase/Tutorial.ipynb)

In this demonstration, we use a multi-table dataset of 3 million online grocery orders from Instacart to predict what a customer will buy next. We show how to generate features with automated feature engineering and build an accurate machine learning pipeline using Featuretools, which can be reused for multiple prediction problems. For more advanced users, we show how to scale that pipeline to a large dataset using Dask.

For more examples of how to use Featuretools, check out our [demos](https://www.featuretools.com/demos) page.

## Testing & Development

The Featuretools community welcomes pull requests. Instructions for testing and development are available [here.](https://featuretools.alteryx.com/en/stable/install.html#development)

## Support
The Featuretools community is happy to provide support to users of Featuretools. Project support can be found in four places depending on the type of question:

1. For usage questions, use [Stack Overflow](https://stackoverflow.com/questions/tagged/featuretools) with the `featuretools` tag.
2. For bugs, issues, or feature requests start a [Github issue](https://github.com/alteryx/featuretools/issues).
3. For discussion regarding development on the core library, use [Slack](https://join.slack.com/t/alteryx-oss/shared_invite/zt-182tyvuxv-NzIn6eiCEf8TBziuKp0bNA).
4. For everything else, the core developers can be reached by email at open_source_support@alteryx.com

## Citing Featuretools

If you use Featuretools, please consider citing the following paper:

James Max Kanter, Kalyan Veeramachaneni. [Deep feature synthesis: Towards automating data science endeavors.](https://dai.lids.mit.edu/wp-content/uploads/2017/10/DSAA_DSM_2015.pdf) *IEEE DSAA 2015*.

BibTeX entry:

```bibtex
@inproceedings{kanter2015deep,
  author    = {James Max Kanter and Kalyan Veeramachaneni},
  title     = {Deep feature synthesis: Towards automating data science endeavors},
  booktitle = {2015 {IEEE} International Conference on Data Science and Advanced Analytics, DSAA 2015, Paris, France, October 19-21, 2015},
  pages     = {1--10},
  year      = {2015},
  organization={IEEE}
}
```

## Built at Alteryx

**Featuretools** is an open source project maintained by [Alteryx](https://www.alteryx.com). To see the other open source projects we’re working on visit [Alteryx Open Source](https://www.alteryx.com/open-source). If building impactful data science pipelines is important to you or your business, please get in touch.

<p align="center">
  <a href="https://www.alteryx.com/open-source">
    <img src="https://alteryx-oss-web-images.s3.amazonaws.com/OpenSource_Logo-01.png" alt="Alteryx Open Source" width="800"/>
  </a>
</p>
