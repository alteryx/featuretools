.. _changelog:

Changelog
---------
**v0.4.1** Nov 29, 2018
    * Resolve bug preventing using first column as index by default (:pr:`308`)
    * Handle return type when creating features from Id variables (:pr:`318`)
    * Make id an optional parameter of EntitySet constructor (:pr:`324`)
    * Handle primitives with same function being applied to same column (:pr:`321`)
    * Update requirements (:pr:`328`)
    * Clean up DFS arguments (:pr:`319`)
    * Clean up Pandas Backend (:pr:`302`)
    * Update properties of cumulative transform primitives (:pr:`320`)
    * Feature stability between versions documentation (:pr:`316`)
    * Add download count to GitHub readme (:pr:`310`)
    * Fixed #297 update tests to check error strings (:pr:`303`)
    * Remove usage of fixtures in agg primitive tests (:pr:`325`)

**v0.4.0** Oct 31, 2018
    * Remove ft.utils.gen_utils.getsize and make pympler a test requirement (:pr:`299`)
    * Update requirements.txt (:pr:`298`)
    * Refactor EntitySet.find_path(...) (:pr:`295`)
    * Clean up unused methods (:pr:`293`)
    * Remove unused parents property of Entity (:pr:`283`)
    * Removed relationships parameter (:pr:`284`)
    * Improve time index validation (:pr:`285`)
    * Encode features with "unknown" class in categorical (:pr:`287`)
    * Allow where clauses on direct features in Deep Feature Synthesis (:pr:`279`)
    * Change to fullargsspec (:pr:`288`)
    * Parallel verbose fixes (:pr:`282`)
    * Update tests for python 3.7 (:pr:`277`)
    * Check duplicate rows cutoff times (:pr:`276`)
    * Load retail demo data using compressed file (:pr:`271`)

**v0.3.1** Sept 28, 2018
    * Handling time rewrite (:pr:`245`)
    * Update deep_feature_synthesis.py (:pr:`249`)
    * Handling return type when creating features from DatetimeTimeIndex (:pr:`266`)
    * Update retail.py (:pr:`259`)
    * Improve Consistency of Transform Primitives (:pr:`236`)
    * Update demo docstrings (:pr:`268`)
    * Handle non-string column names (:pr:`255`)
    * Clean up merging of aggregation primitives (:pr:`250`)
    * Add tests for Entity methods (:pr:`262`)
    * Handle no child data when calculating aggregation features with multiple arguments (:pr:`264`)
    * Add `is_string` utils function (:pr:`260`)
    * Update python versions to match docker container (:pr:`261`)
    * Handle where clause when no child data (:pr:`258`)
    * No longer cache demo csvs, remove config file (:pr:`257`)
    * Avoid stacking "expanding" primitives (:pr:`238`)
    * Use randomly generated names in retail csv (:pr:`233`)
    * Update README.md (:pr:`243`)

**v0.3.0** Aug 27, 2018
    * Improve performance of all feature calculations (:pr:`224`)
    * Update agg primitives to use more efficient functions (:pr:`215`)
    * Optimize metadata calculation (:pr:`229`)
    * More robust handling when no data at a cutoff time (:pr:`234`)
    * Workaround categorical merge (:pr:`231`)
    * Switch which CSV is associated with which variable (:pr:`228`)
    * Remove unused kwargs from query_by_values, filter_and_sort (:pr:`225`)
    * Remove convert_links_to_integers (:pr:`219`)
    * Add conda install instructions (:pr:`223`, :pr:`227`)
    * Add example of using Dask to parallelize to docs  (:pr:`221`)

**v0.2.2** Aug 20, 2018
    * Remove unnecessary check no related instances call and refactor (:pr:`209`)
    * Improve memory usage through support for pandas categorical types (:pr:`196`)
    * Bump minimum pandas version from 0.20.3 to 0.23.0 (:pr:`216`)
    * Better parallel memory warnings (:pr:`208`, :pr:`214`)
    * Update demo datasets (:pr:`187`, :pr:`201`, :pr:`207`)
    * Make primitive lookup case insensitive  (:pr:`213`)
    * Use capital name (:pr:`211`)
    * Set class name for Min (:pr:`206`)
    * Remove ``variable_types`` from normalize entity (:pr:`205`)
    * Handle parquet serialization with last time index (:pr:`204`)
    * Reset index of cutoff times in calculate feature matrix (:pr:`198`)
    * Check argument types for .normalize_entity (:pr:`195`)
    * Type checking ignore entities.  (:pr:`193`)

**v0.2.1** July 2, 2018
    * Cpu count fix (:pr:`176`)
    * Update flight (:pr:`175`)
    * Move feature matrix calculation helper functions to separate file (:pr:`177`)

**v0.2.0** June 22, 2018
    * Multiprocessing (:pr:`170`)
    * Handle unicode encoding in repr throughout Featuretools (:pr:`161`)
    * Clean up EntitySet class (:pr:`145`)
    * Add support for building and uploading conda package (:pr:`167`)
    * Parquet serialization (:pr:`152`)
    * Remove variable stats (:pr:`171`)
    * Make sure index variable comes first (:pr:`168`)
    * No last time index update on normalize (:pr:`169`)
    * Remove list of times as on option for `cutoff_time` in `calculate_feature_matrix` (:pr:`165`)
    * Config does error checking to see if it can write to disk (:pr:`162`)


**v0.1.21** May 30, 2018
    * Support Pandas 0.23.0 (:pr:`153`, :pr:`154`, :pr:`155`, :pr:`159`)
    * No EntitySet required in loading/saving features (:pr:`141`)
    * Use s3 demo csv with better column names (:pr:`139`)
    * more reasonable start parameter (:pr:`149`)
    * add issue template (:pr:`133`)
    * Improve tests (:pr:`136`, :pr:`137`, :pr:`144`, :pr:`147`)
    * Remove unused functions (:pr:`140`, :pr:`143`, :pr:`146`)
    * Update documentation after recent changes / removals (:pr:`157`)
    * Rename demo retail csv file (:pr:`148`)
    * Add names for binary (:pr:`142`)
    * EntitySet repr to use get_name rather than id (:pr:`134`)
    * Ensure config dir is writable (:pr:`135`)

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
