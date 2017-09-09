What is Feature Engineering?
============================

In order to prepare raw data for modeling or machine learning, we first have to perform :term:`feature engineering`. This task involves using human intuition and expertise to transform a dataset into explanatory or predictive signals.

.. "Coming up with features is difficult, time-consuming, requires expert knowledge. "Applied machine learning" is basically feature engineering." --



Feature Engineering Examples
****************************

*The average flight delay for flights out of Boston Logan Airport*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
**How to calculate**

1.  We have to figure out the amount of time a plane was delayed. To do this, we subtract the scheduled departure time from the actual departure.

2.  We have to find the subset of flights in the dataset departing from Boston Logan. Then we average their delays. We can decide between using mean or median for our average.


*Percentage of a user's activity from a mobile device in last week*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
**How to calculate**

1.  We must split up our dataset such that we have all of the visits for each user in one place.

2.  For each user, we have to sum the total activity, as well as the sum just the mobile device activity

3.  We calculate the feature for each user by dividing the mobile activity by the total activity


Why structure and automate?
***************************
Feature engineering is time consuming, it has the biggest impact on the successful of data science projects.

Today, people perform feature engineering by writing adhoc scripts with general purpose tools like Pandas or SQL. As a result,

* There's no shared approach to implement this common process
* It is difficult to reuse feature implementations on new datasets
* There is a high barrier to entry to perform machine learning for people without experience



