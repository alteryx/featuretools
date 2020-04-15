:description: A list of libraries, use cases / demos, and tutorials that leverage Featuretools

===============================
Featuretools External Ecosystem
===============================

New projects are regularly being built on top of Featuretools, highlighting the importance of automated feature engineering. On this page, we have a list of libraries, use cases / demos, and tutorials that leverage Featuretools. If you would like to add a project, please contact us or submit a pull request on `GitHub`_.

.. _`GitHub`: https://github.com/FeatureLabs/featuretools

.. note::

    We are proud and excited to share the work of people using Featuretools, but we cannot endorse or provide support for the tools on this page.

---------
Libraries
---------


`MLBlocks`_
===========
- MLBlocks is a simple framework for composing end-to-end tunable Machine Learning Pipelines by seamlessly combining tools from any python library with a simple, common and uniform interface. MLBlocks contains a primitive that uses Featuretools.

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
- A common use case for machine learning is to predict customer lifetime value. This article walks through the importance of this prediction problem using Featuretools in the process.

.. _`Predict customer lifetime value`: https://towardsdatascience.com/automating-interpretable-feature-engineering-for-predicting-clv-87ece7da9b36

`Predict NHL playoff matches`_
==============================
- Many users of `Kaggle`_ are eager to use Featuretools to improve their model performance. In this blog post, a Kaggle user takes a dataset of plays from National Hockey League games and creates a model to predict if a game is a playoff match.

.. _`Predict NHL playoff matches`: https://towardsdatascience.com/automated-feature-engineering-for-predictive-modeling-d8c9fa4e478b
.. _`Kaggle`: https://www.kaggle.com/

`Predict poverty of households in Costa Rica`_
==============================================
- Social programs have a difficult time determining the right people to give aid. Using a dataset of Costa Rican household characteristics, this Kaggle kernel predicts the poverty of households.

.. _`Predict poverty of households in Costa Rica`: https://www.kaggle.com/willkoehrsen/featuretools-for-good

`Predicting Functional Threshold Power (FTP)`_
==============================================
- This notebook and accompanying report evaluates the use of machine learning for predicting a cyclist’s FTP using data collected from previous training sessions. Featuretools is used to generate a set of independent variables that capture changes in performance over time.

.. _`Predicting Functional Threshold Power (FTP)`: https://github.com/jrkinley/ftp_proba

.. note::

    For more demos written by `Feature Labs <https://www.featurelabs.com>`_, see `featuretools.com/demos <https://www.featuretools.com/demos/>`_

---------
Tutorials
---------
`Automated Feature Engineering in Python`_
==========================================
- This article provides a walk-through of how to use a retail dataset with DFS.

.. _`Automated Feature Engineering in Python`: https://towardsdatascience.com/automated-feature-engineering-in-python-99baf11cc219

`A Hands-On Guide to Automated Feature Engineering`_
====================================================
- A **in-depth** tutorial that works through using Featuretools to predict future product sales at "BigMart".

.. _`A Hands-On Guide to Automated Feature Engineering`: https://www.analyticsvidhya.com/blog/2018/08/guide-automated-feature-engineering-featuretools-python/

`Simple Automatic Feature Engineering`_
=======================================
- A walk-through that applies Featuretools to a sample dataset and creates a classifier to predict clients who make large orders.

.. _`Simple Automatic Feature Engineering`: https://medium.com/@rrfd/simple-automatic-feature-engineering-using-featuretools-in-python-for-classification-b1308040e183

`Introduction to Automated Feature Engineering Using DFS`_
==========================================================
- This article demonstrates using Featuretools helps automate the manual process of feature engineering on a dataset of home loans.

.. _`Introduction to Automated Feature Engineering Using DFS`: https://heartbeat.fritz.ai/introduction-to-automated-feature-engineering-using-deep-feature-synthesis-dfs-3feb69a7c00b

`Automated Feature Engineering Workshop`_
=========================================
- An automated feature engineering workshop using Featuretools hosted at the 2017 Data Summer Conference.

.. _`Automated Feature Engineering Workshop`: https://github.com/fred-navruzov/featuretools-workshop

`Tutorial in Japanese`_
=======================
- A tutorial of Featuretools that demonstrates integrating with the feature selection library `Boruta`_ and the hyper parameter tuning library `Optuna`_.

.. _`Tutorial in Japanese`: https://dev.classmethod.jp/machine-learning/yoshim-featuretools-boruta-optuna/
.. _`Optuna`: https://github.com/pfnet/optuna
.. _`Boruta`: https://github.com/scikit-learn-contrib/boruta_py

`Building a Churn Prediction Model using Featuretools`_
=======================================================
- A video tutorial that shows how to build a churn prediction model using Featuretools along with `Spark`_, `XGBoost`_, and `Google Cloud Platform`_.

.. _`Building a Churn Prediction Model using Featuretools`: https://youtu.be/ZwwneZ6iU3Y
.. _`Spark`: https://spark.apache.org/
.. _`XGBoost`: https://github.com/dmlc/xgboost
.. _`Google Cloud Platform`: https://cloud.google.com/

`Automated Feature Engineering Workshop in Russian`_
====================================================
- A video tutorial that shows how to predict if an applicant is capable of repaying a loan using Featuretools.

.. _`Automated Feature Engineering Workshop in Russian`: https://youtu.be/R0-mnamKxqY
