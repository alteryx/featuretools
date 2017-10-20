.. _changelog:

Changelog
---------

**v0.1.11** October XX, 2017
    * Package linting (:pr:`#7`)
    * Custom primitive creation functions (:pr:`13`)
    * Split requirements to separate files and pin to latest versions (:pr:`15`)
    * Replace ``selection.select_high_variance_features`` and ``selection.select_percent_null`` with ``selection.remove_low_information_features`` (:pr:`18`)

**v0.1.10** October 12, 2017
    * NumTrue primitive added and docstring of other primitives updated (:pr:`11`)
    * fixed hash issue with same base features (:pr:`8`)
    * Head fix (:pr:`9`)
    * Fix training window (:pr:`10`)
    * Add associative attribute to primitives (:pr:`3`)
    * Add status badges, fix license in setup.py (:pr:`1`)
    * fixed head printout and flight demo index (:pr:`2`)

**v0.1.9** September 8, 2017
    * Documentation improvements
    * New ``featuretools.demo.load_mock_customer`` function


**v0.1.8** September 1, 2017
    * Bug fixes
    * Added ``Percentile`` transform primitive

**v0.1.7** August 17, 2017
    * Performance improvements for approximate in ``calculate_feature_matrix`` and ``dfs``
    * Added ``Week`` transform primitive

**v0.1.6** July 26, 2017

    * Added ``load_features`` and ``save_features`` to persist and reload features
    * Added save_progress argument to ``calculate_feature_matrix``
    * Added approximate parameter to ``calculate_feature_matrix`` and ``dfs``
    * Added ``load_flight`` to ft.demo

**v0.1.5** July 11, 2017

    * Windows support

**v0.1.3** July 10, 2017

    * Renamed feature submodule to primitives
    * Renamed prediction_entity arguments to target_entity
    * Added training_window parameter to ``calculate_feature_matrix``


**v0.1.2** July 3rd, 2017

    * Initial release

.. command
.. git log --pretty=oneline --abbrev-commit
