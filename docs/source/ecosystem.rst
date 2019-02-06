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
- This is a python library written to scale FeatureTools with Spark. While we have developed an `internal tutorial`_, this is external resource you can use.

.. _`Featuretools for Spark`: https://github.com/pan5431333/featuretools4s
.. _`internal tutorial`: https://github.com/pan5431333/featuretools4s

`Featuretools for R`_
=====================
- Many data scientists use **R** instead of **Python** for their day-to-day tasks. This library provides an R interface for featuretools.

.. _`Featuretools for R`: https://github.com/magnusfurugard/featuretoolsR

`MLBlocks`_
===========
- MLBlocks is a framework for creating end-to-end machine learning library pipelines. A primitive in MLBlocks which allows your to use Featuretools.

.. _`MLBlocks`: https://github.com/HDI-Project/MLBlocks

`Cardea`_
=========
- Cardea is a machine learning library built on top of the FHIR data schema. It uses a number of **automl** tools, including featureotools.

.. _`Cardea`: https://github.com/D3-AI/Cardea


-----------------
Demos & Use Cases
-----------------
`Predict customer lifetime value`_
==================================
- A common use case for machine learning is to create a model for predicting customer lifetime value. This article describes the business value of this prediction, explains the machine learning methodology, and show how well the results performed.

.. _`Predict customer lifetime value`: https://towardsdatascience.com/automating-interpretable-feature-engineering-for-predicting-clv-87ece7da9b36


`Predict NHL playoff matches`_
==============================
- Many users of Kagglers are eager to use Featuretools to improve their model performance. In this blog post, we see how a user takes a NHL dataset containing game information, and uses Featuretools to create a sklearn model.

.. _`Predict NHL playoff matches`: https://towardsdatascience.com/automated-feature-engineering-for-predictive-modeling-d8c9fa4e478b

`Predict poverty of households in Costa Rica`_
==============================================
- Social programs have a difficult time determining that the right people are given enough aid. Using a dataset of Costa Rican household characteristics, this Kaggle kernel predicts the poverty of households.

.. _`Predict poverty of households in Costa Rica`: https://www.kaggle.com/willkoehrsen/featuretools-for-good

- For demos written by Feature Labs, checkout our `demos <https://www.featuretools.com/demos/>`_.

---------
Tutorials
---------
`Automated Feature Engineering in Python`_
==========================================
- This article provides a walkthrough of how to user our demo retail dataset and use DFS for automated feature engineering.

.. _`Automated Feature Engineering in Python`: https://towardsdatascience.com/automated-feature-engineering-in-python-99baf11cc219

`A Hands-On Guide to Automated Feature Engineering using featuretools in Python`_
=================================================================================
- A **in-depth** tutorial that works through using Featuretools and explains the importance of feature engineering.

.. _`A Hands-On Guide to Automated Feature Engineering using Featuretools in Python`: https://www.analyticsvidhya.com/blog/2018/08/guide-automated-feature-engineering-featuretools-python/

`Simple Automatic Feature Engineering`_
=======================================
- A walkthrough which takes a user-generated dataset, and create a classifier to predict clients who make large orders. It also shows how to understand which features were the most important in classifing these clients.

.. _`Simple Automatic Feature Engineering`: https://medium.com/@rrfd/simple-automatic-feature-engineering-using-featuretools-in-python-for-classification-b1308040e183

`Introduction to Automated Feature Engineering Using DFS`_
==========================================================
- A comprehensive guide on feature engineering, and how to use Featuretools. Using the loan dataset, we see how Featuretools helps automate the manual process of feature engineering.

.. _`Introduction to Automated Feature Engineering Using DFS`: https://heartbeat.fritz.ai/introduction-to-automated-feature-engineering-using-deep-feature-synthesis-dfs-3feb69a7c00b


`Automated Feature Engineering Workshop`_
=========================================
- From the 2017 Data Summer Conference, an automated feature engineering workshop using Featuretools.

.. _`Automated Feature Engineering Workshop`: https://github.com/fred-navruzov/featuretools-workshop

`Tutorial in Japanese`_
=======================
- A tutorial of Featuretools that works with a hyper parameter tuning library `Optuna`_, and a feature selection library `Boruta`_.

.. _`Tutorial in Japanese`: https://dev.classmethod.jp/machine-learning/yoshim-featuretools-boruta-optuna/
.. _`Optuna`: https://github.com/pfnet/optuna
.. _`Boruta`: https://github.com/scikit-learn-contrib/boruta_py
