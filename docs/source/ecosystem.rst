======================
Featuretools Ecosystem
======================
New projects are regularly being built on top of Featuretools. These projects not only validate the importance of feature engineering, but also signals that Featuretools is providing a useful set of functionality to users.

On this page, we have a list of libraries, use case / demos, and tutorials that leverage Featuretools. It is far from an exhaustive list. If you would like to add a project, please contact us or submit a pull request on GitHub.

---------
Libraries
---------
`Featuretools for Spark`_
=========================
- This is a python library written to scale Featuretools with `Spark`_. While we have developed an `internal tutorial`_, this is external resource you can use.

.. _`Featuretools for Spark`: https://github.com/pan5431333/featuretools4s
.. _`internal tutorial`: https://medium.com/feature-labs-engineering/featuretools-on-spark-e5aa67eaf807
.. _`Spark`: https://spark.apache.org/

`Featuretools for R`_
=====================
- Many data scientists use **R** instead of **Python** for their day-to-day tasks. This library provides an **R** interface for Featuretools.

.. _`Featuretools for R`: https://github.com/magnusfurugard/featuretoolsR

`MLBlocks`_
===========
- MLBlocks is a framework for creating end-to-end machine learning pipelines. MLBlocks contains a primitive which uses Featuretools.

.. _`MLBlocks`: https://github.com/HDI-Project/MLBlocks

`Cardea`_
=========
- Cardea is a machine learning library built on top of the FHIR data schema. It uses a number of **automl** tools, including Featuretools.

.. _`Cardea`: https://github.com/D3-AI/Cardea

-----------------
Demos & Use Cases
-----------------
`Predict customer lifetime value`_
==================================
- A common use case for machine learning is to predict customer lifetime value. This article walks through the importance of this prediction, and uses Featuretools in the process.

.. _`Predict customer lifetime value`: https://towardsdatascience.com/automating-interpretable-feature-engineering-for-predicting-clv-87ece7da9b36

`Predict NHL playoff matches`_
==============================
- Many users of `Kaggle`_ are eager to use Featuretools to improve their model performance. In this blog post, a Kaggle user takes a NHL dataset containing game information, and uses Featuretools to create a model.

.. _`Predict NHL playoff matches`: https://towardsdatascience.com/automated-feature-engineering-for-predictive-modeling-d8c9fa4e478b
.. _`Kaggle`: https://www.kaggle.com/

`Predict poverty of households in Costa Rica`_
==============================================
- Social programs have a difficult time determining the right people to give aid. Using a dataset of Costa Rican household characteristics, this Kaggle kernel predicts the poverty of households.

.. _`Predict poverty of households in Costa Rica`: https://www.kaggle.com/willkoehrsen/featuretools-for-good

.. note::

    For demos written by Feature Labs, see `featuretools.com/demos <https://www.featuretools.com/demos/>`_

---------
Tutorials
---------
`Automated Feature Engineering in Python`_
==========================================
- This article provides a walk-through of how to use a retail dataset with DFS.

.. _`Automated Feature Engineering in Python`: https://towardsdatascience.com/automated-feature-engineering-in-python-99baf11cc219

`A Hands-On Guide to Automated Feature Engineering`_
====================================================

- A **in-depth** tutorial that works through using Featuretools.

.. _`A Hands-On Guide to Automated Feature Engineering`: https://www.analyticsvidhya.com/blog/2018/08/guide-automated-feature-engineering-featuretools-python/

`Simple Automatic Feature Engineering`_
=======================================
- A walk-through which takes a user-generated dataset, and create a classifier to predict clients who make large orders.

.. _`Simple Automatic Feature Engineering`: https://medium.com/@rrfd/simple-automatic-feature-engineering-using-featuretools-in-python-for-classification-b1308040e183

`Introduction to Automated Feature Engineering Using DFS`_
==========================================================
- Using the loan dataset, we see how Featuretools helps automate the manual process of feature engineering.

.. _`Introduction to Automated Feature Engineering Using DFS`: https://heartbeat.fritz.ai/introduction-to-automated-feature-engineering-using-deep-feature-synthesis-dfs-3feb69a7c00b

`Automated Feature Engineering Workshop`_
=========================================
- From the 2017 Data Summer Conference, an automated feature engineering workshop using Featuretools.

.. _`Automated Feature Engineering Workshop`: https://github.com/fred-navruzov/featuretools-workshop

`Tutorial in Japanese`_
=======================
- A tutorial of Featuretools that works with a feature selection library `Boruta`_ and a hyper parameter tuning library `Optuna`_.

.. _`Tutorial in Japanese`: https://dev.classmethod.jp/machine-learning/yoshim-featuretools-boruta-optuna/
.. _`Optuna`: https://github.com/pfnet/optuna
.. _`Boruta`: https://github.com/scikit-learn-contrib/boruta_py
