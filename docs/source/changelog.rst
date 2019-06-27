.. _changelog:

Changelog
---------
**Future Release**
    * Enhancements
        * Speedup groupby transform calculations (:pr:`609`)
        * Generate features along all paths when there are multiple paths between entities (:pr:`600`, :pr:`608`)
    * Fixes
        * Select columns of dataframe using a list (:pr:`615`)
        * Change type of features calculated on Index features to Categorical (:pr:`602`)
        * Filter dataframes through forward relationships (:pr:`625`)
        * Specify Dask version in requirements for python 2 (:pr:`627`)
        * Keep dataframe sorted by time during feature calculation (:pr:`626`)
    * Changes
        * Remove unused variance_selection.py file (:pr:`613`)
        * Remove Timedelta data param (:pr:`619`)
        * Remove DaysSince primitive (:pr:`628`)
    * Documentation Changes
        * Add installation instructions for add-on libraries (:pr:`617`)
    * Testing Changes
        * Miscellaneous changes (:pr:`595`, :pr:`612`)

    Thanks to the following people for contributing to this release:
    :user:`CJStadler`, :user:`kmax12`, :user:`rwedge`, :user:`gsheni`, :user:`kkleidal`

**v0.9.0** June 19, 2019
    * Enhancements
        * Add unit parameter to timesince primitives (:pr:`558`)
        * Add ability to install optional add on libraries (:pr:`551`)
        * Load and save features from open files and strings (:pr:`566`)
        * Support custom variable types (:pr:`571`)
        * Support entitysets which have multiple paths between two entities (:pr:`572`, :pr:`544`)
        * Added show_info function, more output information added to CLI `featuretools info` (:pr:`525`)
    * Fixes
        * Normalize_entity specifies error when 'make_time_index' is an invalid string (:pr:`550`)
        * Schema version added for entityset serialization (:pr:`586`)
        * Renamed features have names correctly serialized (:pr:`585`)
        * Improved error message for index/time_index being the same column in normalize_entity and entity_from_dataframe (:pr:`583`)
        * Removed all mentions of allow_where (:pr:`587`, :pr:`588`)
        * Removed unused variable in normalize entity (:pr:`589`)
        * Change time since return type to numeric (:pr:`606`)
    * Changes
        * Refactor get_pandas_data_slice to take single entity (:pr:`547`)
        * Updates TimeSincePrevious and Diff Primitives (:pr:`561`)
        * Remove unecessary time_last variable (:pr:`546`)
    * Documentation Changes
        * Add Featuretools Enterprise to documentation (:pr:`563`)
        * Miscellaneous changes (:pr:`552`, :pr:`573`, :pr:`577`, :pr:`599`)
    * Testing Changes
        * Miscellaneous changes (:pr:`559`, :pr:`569`, :pr:`570`, :pr:`574`, :pr:`584`, :pr:`590`)

    Thanks to the following people for contributing to this release:
    :user:`alexjwang`, :user:`allisonportis`, :user:`CJStadler`, :user:`ctduffy`, :user:`gsheni`, :user:`kmax12`, :user:`rwedge`

**v0.8.0** May 17, 2019
    * Rename NUnique to NumUnique (:pr:`510`)
    * Serialize features as JSON (:pr:`532`)
    * Drop all variables at once in normalize_entity (:pr:`533`)
    * Remove unnecessary sorting from normalize_entity (:pr:`535`)
    * Features cache their names (:pr:`536`)
    * Only calculate features for instances before cutoff (:pr:`523`)
    * Remove all relative imports (:pr:`530`)
    * Added FullName Variable Type (:pr:`506`)
    * Add error message when target entity does not exist (:pr:`520`)
    * New demo links (:pr:`542`)
    * Remove duplicate features check in DFS (:pr:`538`)
    * featuretools_primitives entry point expects list of primitive classes (:pr:`529`)
    * Update ALL_VARIABLE_TYPES list (:pr:`526`)
    * More Informative N Jobs Prints and Warnings (:pr:`511`)
    * Update sklearn version requirements (:pr:`541`)
    * Update Makefile (:pr:`519`)
    * Remove unused parameter in Entity._handle_time (:pr:`524`)
    * Remove build_ext code from setup.py (:pr:`513`)
    * Documentation updates (:pr:`512`, :pr:`514`, :pr:`515`, :pr:`521`, :pr:`522`, :pr:`527`, :pr:`545`)
    * Testing updates (:pr:`509`, :pr:`516`, :pr:`517`, :pr:`539`)

    Thanks to the following people for contributing to this release: :user:`bphi`, :user:`CharlesBradshaw`, :user:`CJStadler`, :user:`glentennis`, :user:`gsheni`, :user:`kmax12`, :user:`rwedge`

**Breaking Changes**

* ``NUnique`` has been renamed to ``NumUnique``.

    Previous behavior

    .. code-block:: python

        from featuretools.primitives import NUnique

    New behavior

    .. code-block:: python

        from featuretools.primitives import NumUnique

**v0.7.1** Apr 24, 2019
    * Automatically generate feature name for controllable primitives (:pr:`481`)
    * Primitive docstring updates (:pr:`489`, :pr:`492`, :pr:`494`, :pr:`495`)
    * Change primitive functions that returned strings to return functions (:pr:`499`)
    * CLI customizable via entrypoints (:pr:`493`)
    * Improve calculation of aggregation features on grandchildren (:pr:`479`)
    * Refactor entrypoints to use decorator (:pr:`483`)
    * Include doctests in testing suite (:pr:`491`)
    * Documentation updates (:pr:`490`)
    * Update how standard primitives are imported internally (:pr:`482`)

    Thanks to the following people for contributing to this release: :user:`bukosabino`, :user:`CharlesBradshaw`, :user:`glentennis`, :user:`gsheni`, :user:`jeff-hernandez`, :user:`kmax12`, :user:`minkvsky`, :user:`rwedge`, :user:`thehomebrewnerd`

**v0.7.0** Mar 29, 2019
    * Improve Entity Set Serialization (:pr:`361`)
    * Support calling a primitive instance's function directly (:pr:`461`, :pr:`468`)
    * Support other libraries extending featuretools functionality via entrypoints (:pr:`452`)
    * Remove featuretools install command (:pr:`475`)
    * Add GroupByTransformFeature (:pr:`455`, :pr:`472`, :pr:`476`)
    * Update Haversine Primitive (:pr:`435`, :pr:`462`)
    * Add commutative argument to SubtractNumeric and DivideNumeric primitives (:pr:`457`)
    * Add FilePath variable_type (:pr:`470`)
    * Add PhoneNumber, DateOfBirth, URL variable types (:pr:`447`)
    * Generalize infer_variable_type, convert_variable_data and convert_all_variable_data methods (:pr:`423`)
    * Documentation updates (:pr:`438`, :pr:`446`, :pr:`458`, :pr:`469`)
    * Testing updates (:pr:`440`, :pr:`444`, :pr:`445`, :pr:`459`)

    Thanks to the following people for contributing to this release: :user:`bukosabino`, :user:`CharlesBradshaw`, :user:`ColCarroll`, :user:`glentennis`, :user:`grayskripko`, :user:`gsheni`, :user:`jeff-hernandez`, :user:`jrkinley`, :user:`kmax12`, :user:`RogerTangos`, :user:`rwedge`

**Breaking Changes**

* ``ft.dfs`` now has a ``groupby_trans_primitives`` parameter that DFS uses to automatically construct features that group by an ID column and then apply a transform primitive to search group. This change applies to the following primitives: ``CumSum``, ``CumCount``, ``CumMean``, ``CumMin``, and ``CumMax``.

    Previous behavior

    .. code-block:: python

        ft.dfs(entityset=es,
               target_entity='customers',
               trans_primitives=["cum_mean"])

    New behavior

    .. code-block:: python

        ft.dfs(entityset=es,
               target_entity='customers',
               groupby_trans_primitives=["cum_mean"])

* Related to the above change, cumulative transform features are now defined using a new feature class, ``GroupByTransformFeature``.

    Previous behavior

    .. code-block:: python

        ft.Feature([base_feature, groupby_feature], primitive=CumulativePrimitive)


    New behavior

    .. code-block:: python

        ft.Feature(base_feature, groupby=groupby_feature, primitive=CumulativePrimitive)


**v0.6.1** Feb 15, 2019
    * Cumulative primitives (:pr:`410`)
    * Entity.query_by_values now preserves row order of underlying data (:pr:`428`)
    * Implementing Country Code and Sub Region Codes as variable types (:pr:`430`)
    * Added IPAddress and EmailAddress variable types (:pr:`426`)
    * Install data and dependencies (:pr:`403`)
    * Add TimeSinceFirst, fix TimeSinceLast (:pr:`388`)
    * Allow user to pass in desired feature return types (:pr:`372`)
    * Add new configuration object (:pr:`401`)
    * Replace NUnique get_function (:pr:`434`)
    * _calculate_idenity_features now only returns the features asked for, instead of the entire entity (:pr:`429`)
    * Primitive function name uniqueness (:pr:`424`)
    * Update NumCharacters and NumWords primitives (:pr:`419`)
    * Removed Variable.dtype (:pr:`416`, :pr:`433`)
    * Change to zipcode rep, str for pandas (:pr:`418`)
    * Remove pandas version upper bound (:pr:`408`)
    * Make S3 dependencies optional (:pr:`404`)
    * Check that agg_primitives and trans_primitives are right primitive type (:pr:`397`)
    * Mean primitive changes (:pr:`395`)
    * Fix transform stacking on multi-output aggregation (:pr:`394`)
    * Fix list_primitives (:pr:`391`)
    * Handle graphviz dependency (:pr:`389`, :pr:`396`, :pr:`398`)
    * Testing updates (:pr:`402`, :pr:`417`, :pr:`433`)
    * Documentation updates (:pr:`400`, :pr:`409`, :pr:`415`, :pr:`417`, :pr:`420`, :pr:`421`, :pr:`422`, :pr:`431`)


    Thanks to the following people for contributing to this release:  :user:`CharlesBradshaw`, :user:`csala`, :user:`floscha`, :user:`gsheni`, :user:`jxwolstenholme`, :user:`kmax12`, :user:`RogerTangos`, :user:`rwedge`

**v0.6.0** Jan 30, 2018
    * Primitive refactor (:pr:`364`)
    * Mean ignore NaNs (:pr:`379`)
    * Plotting entitysets (:pr:`382`)
    * Add seed features later in DFS process (:pr:`357`)
    * Multiple output column features (:pr:`376`)
    * Add ZipCode Variable Type (:pr:`367`)
    * Add `primitive.get_filepath` and example of primitive loading data from external files (:pr:`380`)
    * Transform primitives take series as input (:pr:`385`)
    * Update dependency requirements (:pr:`378`, :pr:`383`, :pr:`386`)
    * Add modulo to override tests (:pr:`384`)
    * Update documentation (:pr:`368`, :pr:`377`)
    * Update README.md (:pr:`366`, :pr:`373`)
    * Update CI tests (:pr:`359`, :pr:`360`, :pr:`375`)

    Thanks to the following people for contributing to this release: :user:`floscha`, :user:`gsheni`, :user:`kmax12`, :user:`RogerTangos`, :user:`rwedge`

**v0.5.1** Dec 17, 2018
    * Add missing dependencies (:pr:`353`)
    * Move comment to note in documentation (:pr:`352`)

**v0.5.0** Dec 17, 2018
    * Add specific error for duplicate additional/copy_variables in normalize_entity (:pr:`348`)
    * Removed EntitySet._import_from_dataframe (:pr:`346`)
    * Removed time_index_reduce parameter (:pr:`344`)
    * Allow installation of additional primitives (:pr:`326`)
    * Fix DatetimeIndex variable conversion (:pr:`342`)
    * Update Sklearn DFS Transformer (:pr:`343`)
    * Clean up entity creation logic (:pr:`336`)
    * remove casting to list in transform feature calculation (:pr:`330`)
    * Fix sklearn wrapper (:pr:`335`)
    * Add readme to pypi
    * Update conda docs after move to conda-forge (:pr:`334`)
    * Add wrapper for scikit-learn Pipelines (:pr:`323`)
    * Remove parse_date_cols parameter from EntitySet._import_from_dataframe (:pr:`333`)

    Thanks to the following people for contributing to this release: :user:`bukosabino`, :user:`georgewambold`, :user:`gsheni`, :user:`jeff-hernandez`, :user:`kmax12`, and :user:`rwedge`.

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
