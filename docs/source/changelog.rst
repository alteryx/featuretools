.. _changelog:

Changelog
---------
**v0.1.20** Apr 13, 2018
    * Primitives as strings in DFS parameters (:pr:`129`)
    * Integer time index bugfixes (:pr:`128`)
    * Add make_temporal_cutoffs utility function (:pr:`126`)
    * Show all entities, switch shape display to row/col (:pr:`124`)
    * Improved chunking when calculating feature matrices  (:pr:`121`)
    * fixed num characters nan fix (:pr:`118`)
    * modify ignore_variables docstring (:pr:`117`)

**v0.1.19** Mar 21, 2018
    * More descriptive DFS progress bar (:pr:`69`)
    * Convert text variable to string before NumWords (:pr:`106`)
    * EntitySet.concat() reindexes relationships (:pr:`96`)
    * Keep non-feature columns when encoding feature matrix (:pr:`111`)
    * Uses full entity update for dependencies of uses_full_entity features (:pr:`110`)
    * Update column names in retail demo (:pr:`104`)
    * Handle Transform features that need access to all values of entity (:pr:`91`)

**v0.1.18** Feb 27, 2018
    * fixes related instances bug (:pr:`97`)
    * Adding non-feature columns to calculated feature matrix (:pr:`78`)
    * Relax numpy version req (:pr:`82`)
    * Remove `entity_from_csv`, tests, and lint (:pr:`71`)

**v0.1.17** Jan 18, 2018
    * LatLong type (:pr:`57`)
    * Last time index fixes (:pr:`70`)
    * Make median agg primitives ignore nans by default (:pr:`61`)
    * Remove Python 3.4 support (:pr:`64`)
    * Change `normalize_entity` to update `secondary_time_index` (:pr:`59`)
    * Unpin requirements (:pr:`53`)
    * associative -> commutative (:pr:`56`)
    * Add Words and Chars primitives (:pr:`51`)

**v0.1.16** Dec 19, 2017
    * fix EntitySet.combine_variables and standardize encode_features (:pr:`47`)
    * Python 3 compatibility (:pr:`16`)

**v0.1.15** Dec 18, 2017
    * Fix variable type in demo data (:pr:`37`)
    * Custom primitive kwarg fix (:pr:`38`)
    * Changed order and text of arguments in make_trans_primitive docstring (:pr:`42`)

**v0.1.14** November 20, 2017
    * Last time index (:pr:`33`)
    * Update Scipy version to 1.0.0 (:pr:`31`)


**v0.1.13** November 1, 2017
    * Add MANIFEST.in (:pr:`26`)

**v0.1.11** October 31, 2017
    * Package linting (:pr:`7`)
    * Custom primitive creation functions (:pr:`13`)
    * Split requirements to separate files and pin to latest versions (:pr:`15`)
    * Select low information features (:pr:`18`)
    * Fix docs typos (:pr:`19`)
    * Fixed Diff primitive for rare nan case (:pr:`21`)
    * added some mising doc strings (:pr:`23`)
    * Trend fix (:pr:`22`)
    * Remove as_dir=False option from EntitySet.to_pickle() (:pr:`20`)
    * Entity Normalization Preserves Types of Copy & Additional Variables (:pr:`25`)

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
