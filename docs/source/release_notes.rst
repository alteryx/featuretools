.. _release_notes:

Release Notes
-------------

Future Release
==============
    * Enhancements
    * Fixes
    * Changes
        * Restrict numpy to <2.0.0 (:pr:`2743`)
    * Documentation Changes
        * Update API Docs to include previously missing primitives (:pr:`2737`)
    * Testing Changes

    Thanks to the following people for contributing to this release:
    :user:`thehomebrewnerd`

v1.31.0 May 14, 2024
====================
    * Enhancements
        * Add support for Python 3.12 (:pr:`2713`)
    * Fixes
        * Move ``flatten_list`` util function into ``feature_discovery`` module to fix import bug (:pr:`2702`)
    * Changes
        * Temporarily restrict Dask version (:pr:`2694`)
        * Remove support for creating ``EntitySets`` from Dask or Pyspark dataframes (:pr:`2705`)
        * Bump minimum versions of ``tqdm`` and ``pip`` in requirements files (:pr:`2716`)
        * Use ``filter`` arg in call to ``tarfile.extractall`` to safely deserialize EntitySets (:pr:`2722`)
    * Testing Changes
        * Fix serialization test to work with pytest 8.1.1 (:pr:`2694`)
        * Update to allow minimum dependency checker to run properly (:pr:`2709`)
        * Update pull request check CI action (:pr:`2720`)
        * Update release notes updated check CI action (:pr:`2726`)

    Thanks to the following people for contributing to this release:
    :user:`thehomebrewnerd`

Breaking Changes
++++++++++++++++
* With this release of Featuretools, EntitySets can no longer be created from Dask or Pyspark dataframes. The behavior when using pandas
  dataframes to create EntitySets remains unchanged.


v1.30.0 Feb 26, 2024
====================
    * Changes
        * Update min requirements for numpy, pandas and Woodwork (:pr:`2681`)
        * Update release notes version for release(:pr:`2689`)
    * Testing Changes
        * Update ``make_ecommerce_entityset`` to work without Dask (:pr:`2677`)

    Thanks to the following people for contributing to this release:
    :user:`tamargrey`, :user:`thehomebrewnerd`

v1.29.0 Feb 16, 2024
====================
    .. warning::
        This release of Featuretools will not support Python 3.8

    * Fixes
        * Fix dependency issues (:pr:`2644`, :pr:`2656`)
        * Add workaround for pandas 2.2.0 bug with nunique and unpin pandas deps (:pr:`2657`)
    * Changes
        * Fix deprecation warnings with is_categorical_dtype (:pr:`2641`)
        * Remove woodwork, pyarrow, numpy, and pandas pins for spark installation (:pr:`2661`)
    * Documentation Changes
        * Update Featuretools logo to display properly in dark mode (:pr:`2632`)
        * Remove references to premium primitives while release isnt possible (:pr:`2674`)
    * Testing Changes
        * Update tests for compatibility with new versions of ``holidays`` (:pr:`2636`)
        * Update ruff to 0.1.6 and use ruff linter/formatter (:pr:`2639`)
        * Update ``release.yaml`` to use trusted publisher for PyPI releases (:pr:`2646`, :pr:`2653`, :pr:`2654`)
        * Update dependency checkers and tests to include Dask (:pr:`2658`)
        * Fix the tests that run with Woodwork main so they can be triggered (:pr:`2657`)
        * Fix minimum dependency checker action (:pr:`2664`)
        * Fix Slack alert for tests with Woodwork main branch (:pr:`2668`)

    Thanks to the following people for contributing to this release:
    :user:`gsheni`, :user:`thehomebrewnerd`, :user:`tamargrey`, :user:`LakshmanKishore`


v1.28.0 Oct 26, 2023
====================
    * Fixes
        * Fix bug with default value in ``PercentTrue`` primitive (:pr:`2627`)
    * Changes
        * Refactor ``featuretools/tests/primitive_tests/utils.py`` to leverage list comprehensions for improved Pythonic quality (:pr:`2607`)
        * Refactor ``can_stack_primitive_on_inputs`` (:pr:`2522`)
        * Update s3 bucket for docs image (:pr:`2593`)
        * Temporarily restrict pandas max version to ``<2.1.0`` and pyarrow to ``<13.0.0`` (:pr:`2609`)
        * Update for compatibility with pandas version ``2.1.0`` and remove pandas upper version restriction (:pr:`2616`)
    * Documentation Changes
        * Fix badge on README for tests (:pr:`2598`)
        * Update readthedocs config to use build.os (:pr:`2601`)
    * Testing Changes
        * Update airflow looking glass performance tests workflow (:pr:`2615`)
        * Removed old performance testing workflow (:pr:`2620`)

    Thanks to the following people for contributing to this release:
    :user:`gsheni`, :user:`petejanuszewski1`, :user:`thehomebrewnerd`, :user:`tosemml`

v1.27.0 Jul 24, 2023
====================
    * Enhancements
        * Add support for Python 3.11 (:pr:`2583`)
        * Add support for ``pandas`` v2.0 (:pr:`2585`)
    * Changes
        * Remove natural language primitives add-on (:pr:`2570`)
        * Updates to address various warnings (:pr:`2589`)
    * Testing Changes
        * Run looking glass performance tests on merge via Airflow (:pr:`2575`)

    Thanks to the following people for contributing to this release:
    :user:`gsheni`, :user:`petejanuszewski1`, :user:`sbadithe`, :user:`thehomebrewnerd`

v1.26.0 Apr 27, 2023
====================
    * Enhancements
        * Introduce New Single-Table DFS Algorithm (:pr:`2516`). This includes **experimental** functionality and is not officially supported.
        * Add premium primitives install command (:pr:`2545`)
    * Fixes
        * Fix Description of ``DaysInMonth`` (:pr:`2547`)
    * Changes
        * Make Dask an optional dependency (:pr:`2560`)

    Thanks to the following people for contributing to this release:
    :user:`dvreed77`, :user:`gsheni`, :user:`thehomebrewnerd`

Breaking Changes
++++++++++++++++
* Dask is now an optional dependency of Featuretools. Users that run ``calculate_feature_matrix`` with ``n_jobs`` set
  to anything other than 1, will now need to install Dask prior to running ``calculate_feature_matrix``. The required Dask
  dependencies can be installed with ``pip install "featuretools[dask]"``.

v1.25.0 Apr 13, 2023
====================
    * Enhancements
        * Add ``MaxCount``, ``MedianCount``, ``MaxMinDelta``, ``NUniqueDays``, ``NMostCommonFrequency``,
            ``NUniqueDaysOfCalendarYear``, ``NUniqueDaysOfMonth``, ``NUniqueMonths``,
            ``NUniqueWeeks``, ``IsFirstWeekOfMonth`` (:pr:`2533`)
        * Add ``HasNoDuplicates``, ``NthWeekOfMonth``, ``IsMonotonicallyDecreasing``, ``IsMonotonicallyIncreasing``,
            ``IsUnique`` (:pr:`2537`)
    * Fixes
        * Fix release notes header version (:pr:`2544`)
    * Changes
        * Restrict pandas to < 2.0.0 (:pr:`2533`)
        * Upgrade minimum pandas to 1.5.0 (:pr:`2537`)
        * Removed the ``Correlation`` and ``AutoCorrelation`` primitive as these could lead to data leakage (:pr:`2537`)
        * Remove IntegerNullable support for ``Kurtosis`` primitive  (:pr:`2537`)

    Thanks to the following people for contributing to this release:
    :user:`gsheni`

v1.24.0 Mar 28, 2023
====================
    * Enhancements
        * Add ``AverageCountPerUnique``, ``CountryCodeToContinent``, ``FileExtension``, ``FirstLastTimeDelta``, ``SavgolFilter``,
            ``CumulativeTimeSinceLastFalse``, ``CumulativeTimeSinceLastTrue``, ``PercentChange``, ``PercentUnique`` (:pr:`2485`)
        * Add ``FullNameToFirstName``, ``FullNameToLastName``, ``FullNameToTitle``, ``AutoCorrelation``,
            ``Correlation``, ``DateFirstEvent`` (:pr:`2507`)
        * Add ``Kurtosis``, ``MinCount``, ``NumFalseSinceLastTrue``, ``NumPeaks``,
            ``NumTrueSinceLastFalse``, ``NumZeroCrossings`` (:pr:`2514`)
    * Fixes
        * Pin github-action-check-linked-issues to 1.4.5 (:pr:`2497`)
        * Support Woodwork's update numeric inference (integers as strings) (:pr:`2505`)
        * Update ``SubtractNumeric`` Primitive with commutative class property (:pr:`2527`)
    * Changes
        * Separate Makefile command for core requirements, test requirements and dev requirements (:pr:`2518`)

    Thanks to the following people for contributing to this release:
    :user:`dvreed77`, :user:`gsheni`, :user:`ozzieD`

v1.23.0 Feb 15, 2023
====================
    * Changes
        * Change ``TotalWordLength`` and ``UpperCaseWordCount`` to return ``IntegerNullable`` (:pr:`2474`)
    * Testing Changes
       * Add GitHub Actions cache to speed up workflows (:pr:`2475`)
       * Fix latest dependency checker install command (:pr:`2476`)
       * Add pull request check for linked issues to CI workflow (:pr:`2477`, :pr:`2481`)
       * Remove make package from lint workflow (:pr:`2479`)

    Thanks to the following people for contributing to this release:
    :user:`dvreed77`, :user:`gsheni`, :user:`sbadithe`

v1.22.0 Jan 31, 2023
====================
    * Enhancements
        * Add ``AbsoluteDiff``, ``SameAsPrevious``, ``Variance``, ``Season``, ``UpperCaseWordCount`` transform primitives (:pr:`2460`)
    * Fixes
        * Fix bug with consecutive spaces in ``NumWords`` (:pr:`2459`)
        * Fix for compatibility with ``holidays`` v0.19.0 (:pr:`2471`)
    * Changes
        * Specify black and ruff config arguments in pre-commit-config (:pr:`2456`)
        * ``NumCharacters`` returns null given null input (:pr:`2463`)
    * Documentation Changes
        * Update ``release.md`` with instructions for launching Looking Glass performance test runs (:pr:`2461`)
        * Pin ``jupyter-client==7.4.9`` to fix broken documentation build (:pr:`2463`)
        * Unpin jupyter-client documentation requirement (:pr:`2468`)
    * Testing Changes
        * Add test suites for ``NumWords`` and ``NumCharacters`` primitives (:pr:`2459`, :pr:`2463`)

    Thanks to the following people for contributing to this release:
    :user:`gsheni`, :user:`rwedge`, :user:`sbadithe`, :user:`thehomebrewnerd`

v1.21.0 Jan 18, 2023
====================
    * Enhancements
        * Add `get_recommended_primitives` function to featuretools (:pr:`2398`)
    * Changes
        * Update build_docs workflow to only run for Python 3.8 and Python 3.10 (:pr:`2447`)
    * Documentation Changes
        * Minor fix to release notes (:pr:`2444`)
    * Testing Changes
        * Add test that checks for Natural Language primitives timing out against edge-case input (:pr:`2429`)
        * Fix test compatibility with composeml 0.10 (:pr:`2439`)
        * Minimum dependency unit test jobs do not abort if one job fails (:pr:`2437`)
        * Run Looking Glass performance tests on merge to main (:pr:`2440`, :pr:`2441`)
        * Add ruff for linting and replace isort/flake8 (:pr:`2448`)

    Thanks to the following people for contributing to this release:
    :user:`gsheni`, :user:`ozzieD`, :user:`rwedge`, :user:`sbadithe`, :user:`thehomebrewnerd`

v1.20.0 Jan 5, 2023
===================
    * Enhancements
        * Add ``TimeSinceLastFalse``, ``TimeSinceLastMax``, ``TimeSinceLastMin``, and ``TimeSinceLastTrue`` primitives (:pr:`2418`)
        * Add ``MaxConsecutiveFalse``, ``MaxConsecutiveNegatives``, ``MaxConsecutivePositives``, ``MaxConsecutiveTrue``, ``MaxConsecutiveZeros``, ``NumConsecutiveGreaterMean``, ``NumConsecutiveLessMean`` (:pr:`2420`)
    * Fixes
        * Fix typo in ``_handle_binary_comparison`` function name and update ``set_feature_names`` docstring (:pr:`2388`)
        * Only allow Datetime time index as input to ``RateOfChange`` primitive (:pr:`2408`)
        * Prevent catastrophic backtracking in regex for ``NumberOfWordsInQuotes`` (:pr:`2413`)
        * Fix to eliminate fragmentation ``PerformanceWarning`` in ``feature_set_calculator.py`` (:pr:`2424`)
        * Fix serialization of ``NumberOfCommonWords`` feature with custom word_set (:pr:`2432`)
        * Improve edge case handling in NaturalLanguage primitives by standardizing delimiter regex (:pr:`2423`)
        * Remove support for ``Datetime`` and ``Ordinal`` inputs in several primitives to prevent creation of Features that cannot be calculated (:pr:`2434`)
    * Changes
        * Refactor ``_all_direct_and_same_path`` by deleting call to ``_features_have_same_path`` (:pr:`2400`)
        * Refactor ``_build_transform_features`` by iterating over ``input_features`` once (:pr:`2400`)
        * Iterate only once over ``ignore_columns`` in ``DeepFeatureSynthesis`` init (:pr:`2397`)
        * Resolve empty Pandas series warnings (:pr:`2403`)
        * Initialize Woodwork with ``init_with_partial_schama`` instead of ``init`` in ``EntitySet.add_last_time_indexes`` (:pr:`2409`)
        * Updates for compatibility with numpy 1.24.0 (:pr:`2414`)
        * The ``delimiter_regex`` parameter for ``TotalWordLength`` has been renamed to ``do_not_count`` (:pr:`2423`)
    * Documentation Changes
        *  Remove unused sections from 1.19.0 notes (:pr:`2396`)

   Thanks to the following people for contributing to this release:
   :user:`gsheni`, :user:`rwedge`, :user:`sbadithe`, :user:`thehomebrewnerd`


Breaking Changes
++++++++++++++++
* The ``delimiter_regex`` parameter for ``TotalWordLength`` has been renamed to ``do_not_count``.
  Old saved features that had a non-default value for the parameter will no longer load.
* Support for ``Datetime`` and ``Ordinal`` inputs has been removed from the ``LessThanScalar``,
  ``GreaterThanScalar``, ``LessThanEqualToScalar`` and ``GreaterThanEqualToScalar`` primitives.

v1.19.0 Dec 9, 2022
===================
    * Enhancements
        * Add ``OneDigitPostalCode`` and ``TwoDigitPostalCode`` primitives (:pr:`2365`)
        * Add ``ExpandingCount``, ``ExpandingMin``, ``ExpandingMean``, ``ExpandingMax``, ``ExpandingSTD``, and ``ExpandingTrend`` primitives (:pr:`2343`)
    * Fixes
        * Fix DeepFeatureSynthesis to consider the ``base_of_exclude`` family of attributes when creating transform features(:pr:`2380`)
        * Fix bug with negative version numbers in ``test_version`` (:pr:`2389`)
        * Fix bug in ``MultiplyNumericBoolean`` primitive that can cause an error with certain input dtype combinations (:pr:`2393`)
    * Testing Changes
        * Fix version comparison in ``test_holiday_out_of_range`` (:pr:`2382`)

    Thanks to the following people for contributing to this release:
    :user:`sbadithe`, :user:`thehomebrewnerd`

v1.18.0 Nov 15, 2022
====================
    * Enhancements
        * Add ``RollingOutlierCount`` primitive (:pr:`2129`)
        * Add ``RateOfChange`` primitive (:pr:`2359`)
    * Fixes
        * Sets ``uses_full_dataframe`` for ``Rolling*`` and ``Exponential*`` primitives (:pr:`2354`)
        * Updates for compatibility with upcoming Woodwork release 0.21.0 (:pr:`2363`)
        * Updates demo dataset location to use new links (:pr:`2366`)
        * Fix ``test_holiday_out_of_range`` after ``holidays`` release 0.17 (:pr:`2373`)
    * Changes
        * Remove click and CLI functions (``list-primitives``, ``info``) (:pr:`2353`, :pr:`2358`)
    * Documentation Changes
        * Build docs in parallel with Sphinx (:pr:`2351`)
        * Use non-editable install to allow local docs build (:pr:`2367`)
        * Remove primitives.featurelabs.com website from documentation (:pr:`2369`)
    * Testing Changes
        * Replace use of pytest's tmpdir fixture with tmp_path (:pr:`2344`)

    Thanks to the following people for contributing to this release:
    :user:`gsheni`, :user:`rwedge`, :user:`sbadithe`, :user:`tamargrey`, :user:`thehomebrewnerd`

Breaking Changes
++++++++++++++++
* The featuretools CLI has been completely removed.

v1.17.0 Oct 31, 2022
====================
    * Enhancements
        * Add featuretools-sklearn-transformer as an extra installation option (:pr:`2335`)
        * Add CountAboveMean, CountBelowMean, CountGreaterThan, CountInsideNthSTD, CountInsideRange, CountLessThan, CountOutsideNthSTD, CountOutsideRange (:pr:`2336`)
    * Changes
        * Restructure primitives directory to use individual primitives files (:pr:`2331`)
        * Restrict 2022.10.1 for dask and distributed (:pr:`2347`)
    * Documentation Changes
        * Add Featuretools-SQL to Install page on documentation (:pr:`2337`)
        * Fixes broken link in Featuretools documentation (:pr:`2339`)

    Thanks to the following people for contributing to this release:
    :user:`gsheni`, :user:`rwedge`, :user:`sbadithe`, :user:`thehomebrewnerd`

v1.16.0 Oct 24, 2022
====================
    * Enhancements
        * Add ExponentialWeighted primitives and DateToTimeZone primitive (:pr:`2318`)
        * Add 14 natural language primitives from ``nlp_primitives`` library (:pr:`2328`)
    * Documentation Changes
        * Fix typos in ``aggregation_primitive_base.py`` and ``features_deserializer.py`` (:pr:`2317`) (:pr:`2324`)
        * Update SQL integration documentation to reflect Snowflake compatibility (:pr:`2313`)
    * Testing Changes
        * Add Windows install test (:pr:`2330`)

    Thanks to the following people for contributing to this release:
    :user:`gsheni`, :user:`sbadithe`, :user:`thehomebrewnerd`

v1.15.0 Oct 6, 2022
===================
    * Enhancements
        * Add ``series_library`` attribute to ``EntitySet`` dictionary (:pr:`2257`)
        * Leverage ``Library`` Enum inheriting from ``str`` (:pr:`2275`)
    * Changes
        * Change default gap for Rolling* primitives from 0 to 1 to prevent accidental leakage (:pr:`2282`)
        * Updates for pandas 1.5.0 compatibility (:pr:`2290`, :pr:`2291`, :pr:`2308`)
        * Exclude documentation files from release workflow (:pr:`2295`)
        * Bump requirements for optional pyspark dependency (:pr:`2299`)
        * Bump ``scipy`` and ``woodwork[spark]`` dependencies (:pr:`2306`)
    * Documentation Changes
        * Add documentation describing how to use ``featuretools_sql`` with ``featuretools`` (:pr:`2262`)
        * Remove ``featuretools_sql`` as a docs requirement (:pr:`2302`)
        * Fix typo in ``DiffDatetime`` doctest (:pr:`2314`)
        * Fix typo in ``EntitySet`` documentation (:pr:`2315`)
    * Testing Changes
        * Remove graphviz version restrictions in Windows CI tests (:pr:`2285`)
        * Run CI tests with ``pytest -n auto`` (:pr:`2298`, :pr:`2310`)

    Thanks to the following people for contributing to this release:
    :user:`gsheni`, :user:`rwedge`, :user:`sbadithe`, :user:`thehomebrewnerd`

Breaking Changes
++++++++++++++++
* The ``EntitySet`` schema has been updated to include a ``series_library`` attribute
* The default behavior of the ``Rolling*`` primitives has changed in this release. If this primitive was used without
  defining the ``gap`` value, the feature values returned with this release will be different than feature values from
  prior releases.

v1.14.0 Sep 1, 2022
===================
    * Enhancements
        * Replace ``NumericLag`` with ``Lag`` primitive (:pr:`2252`)
        * Refactor build_features to speed up long running DFS calls by 50% (:pr:`2224`)
    * Fixes
        * Fix compatibility issues with holidays 0.15 (:pr:`2254`)
    * Changes
        * Update release notes to make clear conda release portion (:pr:`2249`)
        * Use pyproject.toml only (move away from setup.cfg) (:pr:`2260`, :pr:`2263`, :pr:`2265`)
        * Add entry point instructions for pyproject.toml project (:pr:`2272`)
    * Documentation Changes
        * Fix to remove warning from Using Spark EntitySets Guide (:pr:`2258`)
    * Testing Changes
        * Add tests/profiling/dfs_profile.py (:pr:`2224`)
        * Add workflow to test featuretools without test dependencies (:pr:`2274`)

    Thanks to the following people for contributing to this release:
    :user:`cp2boston`, :user:`gsheni`, :user:`ozzieD`, :user:`stefaniesmith`, :user:`thehomebrewnerd`

v1.13.0 Aug 18, 2022
====================
    * Fixes
        * Allow boolean columns to be included in remove_highly_correlated_features (:pr:`2231`)
    * Changes
        * Refactor schema version checking to use `packaging` method (:pr:`2230`)
        * Extract duplicated logic for Rolling primitives into a general utility function (:pr:`2218`)
        * Set pandas version to >=1.4.0 (:pr:`2246`)
        * Remove workaround in `roll_series_with_gap` caused by pandas version < 1.4.0 (:pr:`2246`)
    * Documentation Changes
        * Add line breaks between sections of IsFederalHoliday primitive docstring (:pr:`2235`)
    * Testing Changes
        * Update create feedstock PR forked repo to use (:pr:`2223`, :pr:`2237`)
        * Update development requirements and use latest for documentation (:pr:`2225`)

    Thanks to the following people for contributing to this release:
    :user:`gsheni`, :user:`ozzieD`, :user:`sbadithe`, :user:`tamargrey`

v1.12.1 Aug 4, 2022
===================
    * Fixes
        * Update ``Trend`` and ``RollingTrend`` primitives to work with ``IntegerNullable`` inputs (:pr:`2204`)
        * ``camel_and_title_to_snake`` handles snake case strings with numbers (:pr:`2220`)
        * Change ``_get_description`` to split on blank lines to avoid truncating primitive descriptions (:pr:`2219`)
    * Documentation Changes
        * Add instructions to add new users to featuretools feedstock (:pr:`2215`)
    * Testing Changes
        * Add create feedstock PR workflow (:pr:`2181`)
        * Add performance tests for python 3.9 and 3.10 (:pr:`2198`, :pr:`2208`)
        * Add test to ensure primitive docstrings use standardized verbs (:pr:`2200`)
        * Configure codecov to avoid premature PR comments (:pr:`2209`)

    Thanks to the following people for contributing to this release:
    :user:`gsheni`, :user:`rwedge`, :user:`sbadithe`, :user:`tamargrey`, :user:`thehomebrewnerd`

v1.12.0 Jul 19, 2022
====================
    .. warning::
        This release of Featuretools will not support Python 3.7

    * Enhancements
        * Add ``IsWorkingHours`` and ``IsLunchTime`` transform primitives (:pr:`2130`)
        * Add periods parameter to ``Diff`` and add ``DiffDatetime`` primitive (:pr:`2155`)
        * Add ``RollingTrend`` primitive (:pr:`2170`)
    * Fixes
        * Resolves Woodwork integration test failure and removes Python version check for codecov (:pr:`2182`)
    * Changes
        * Drop Python 3.7 support (:pr:`2169`, :pr:`2186`)
        * Add pre-commit hooks for linting (:pr:`2177`)
    * Documentation Changes
        * Augment single table entry in DFS to include information about passing in a dictionary for `dataframes` argument (:pr:`2160`)
    * Testing Changes
        * Standardize imports across test files to simplify accessing featuretools functions (:pr:`2166`)
        * Split spark tests into multiple CI jobs to speed up runtime (:pr:`2183`)

    Thanks to the following people for contributing to this release:
    :user:`dvreed77`, :user:`gsheni`, :user:`ozzieD`, :user:`rwedge`, :user:`sbadithe`

v1.11.1 Jul 5, 2022
===================
    * Fixes
        * Remove 24th hour from PartOfDay primitive and add 0th hour (:pr:`2167`)

    Thanks to the following people for contributing to this release:
    :user:`tamargrey`

v1.11.0 Jun 30, 2022
====================
    * Enhancements
        * Add datetime and string types as valid arguments to dfs ``cutoff_time`` (:pr:`2147`)
        * Add ``PartOfDay`` transform primitive (:pr:`2128`)
        * Add ``IsYearEnd``, ``IsYearStart`` transform primitives (:pr:`2124`)
        * Add ``Feature.set_feature_names`` method to directly set output column names for multi-output features (:pr:`2142`)
        * Include np.nan testing for ``DayOfYear`` and ``DaysInMonth`` primitives (:pr:`2146`)
        * Allow dfs kwargs to be passed into ``get_valid_primitives`` (:pr:`2157`)
    * Changes
        * Improve serialization and deserialization to reduce storage of duplicate primitive information (:pr:`2136`, :pr:`2127`, :pr:`2144`)
        * Sort core requirements and test requirements in setup cfg (:pr:`2152`)
    * Testing Changes
        * Fix pandas warning and reduce dask .apply warnings (:pr:`2145`)
        * Pin graphviz version used in windows tests (:pr:`2159`)

    Thanks to the following people for contributing to this release:
    :user:`gsheni`, :user:`ozzieD`, :user:`rwedge`, :user:`sbadithe`, :user:`tamargrey`, :user:`thehomebrewnerd`

v1.10.0 Jun 23, 2022
====================
    * Enhancements
        * Add ``DayOfYear``, ``DaysInMonth``, ``Quarter``, ``IsLeapYear``, ``IsQuarterEnd``, ``IsQuarterStart`` transform primitives (:pr:`2110`, :pr:`2117`)
        * Add ``IsMonthEnd``, ``IsMonthStart`` transform primitives (:pr:`2121`)
        * Move ``Quarter`` test cases (:pr:`2123`)
        * Add ``summarize_primitives`` function for getting metrics about available primitives (:pr:`2099`)
    * Changes
        * Changes for compatibility with numpy 1.23.0 (:pr:`2135`, :pr:`2137`)
    * Documentation Changes
        * Update contributing.md to add pandoc (:pr:`2103`, :pr:`2104`)
        * Update NLP primitives section of API reference (:pr:`2109`)
        * Fixing release notes formatting (:pr:`2139`)
    * Testing Changes
        * Latest dependency checker installs spark dependencies (:pr:`2112`)
        * Fix test failures with pyspark v3.3.0 (:pr:`2114`, :pr:`2120`)

    Thanks to the following people for contributing to this release:
    :user:`gsheni`, :user:`ozzieD`, :user:`rwedge`, :user:`sbadithe`, :user:`thehomebrewnerd`

v1.9.2 Jun 10, 2022
===================
    * Fixes
        * Add feature origin information to all multi-output feature columns (:pr:`2102`)
    * Documentation Changes
        * Update contributing.md to add pandoc (:pr:`2103`)

    Thanks to the following people for contributing to this release:
    :user:`gsheni`, :user:`thehomebrewnerd`

v1.9.1 May 27, 2022
===================
    * Enhancements
        * Update ``DateToHoliday`` and ``DistanceToHoliday`` primitives to work with timezone-aware inputs (:pr:`2056`)
    * Changes
        * Delete setup.py, MANIFEST.in and move configuration to pyproject.toml (:pr:`2046`)
    * Documentation Changes
        * Update slack invite link to new (:pr:`2044`)
        * Add slack and stackoverflow icon to footer (:pr:`2087`)
        * Update dead links in docs and docstrings (:pr:`2092`, :pr:`2095`)
    * Testing Changes
        * Skip test for ``normalize_dataframe`` due to different error coming from Woodwork in 0.16.3 (:pr:`2052`)
        * Fix Woodwork install in test with Woodwork main branch (:pr:`2055`)
        * Use codecov action v3 (:pr:`2039`)
        * Add workflow to kickoff EvalML unit tests with Featuretools main (:pr:`2072`)
        * Rename yml to yaml for GitHub Actions workflows (:pr:`2073`, :pr:`2077`)
        * Update Dask test fixtures to prevent flaky behavior (:pr:`2079`)
        * Update Makefile with better pkg command (:pr:`2081`)
        * Add scheduled workflow that checks for broken links in documentation (:pr:`2084`)

    Thanks to the following people for contributing to this release:
    :user:`gsheni`, :user:`rwedge`, :user:`thehomebrewnerd`

v1.9.0 Apr 27, 2022
===================
    * Enhancements
        * Improve ``UnusedPrimitiveWarning`` with additional information (:pr:`2003`)
        * Update DFS primitive matching to use all inputs defined in primitive ``input_types`` (:pr:`2019`)
        * Add ``MultiplyNumericBoolean`` primitive (:pr:`2035`)
    * Fixes
        * Fix issue with Ordinal inputs to binary comparison primitives (:pr:`2024`, :pr:`2025`)
    * Changes
        * Updated autonormalize version requirement (:pr:`2002`)
        * Remove extra NaN checking in LatLong primitives (:pr:`1924`)
        * Normalize LatLong NaN values during EntitySet creation (:pr:`1924`)
        * Pass primitive dictionaries into ``check_primitive`` to avoid repetitive calls (:pr:`2016`)
        * Remove ``Boolean`` and ``BooleanNullable`` from ``MultiplyNumeric`` primitive inputs (:pr:`2022`)
        * Update serialization for compatibility with Woodwork version 0.16.1 (:pr:`2030`)
    * Documentation Changes
        * Update README text to Alteryx (:pr:`2010`, :pr:`2015`)
    * Testing Changes
        * Update unit tests with Woodwork main branch workflow name (:pr:`2033`)
        * Add slack alert for failing unit tests with Woodwork main branch (:pr:`2040`)

    Thanks to the following people for contributing to this release:
    :user:`dvreed77`, :user:`gsheni`, :user:`ozzieD`, :user:`rwedge`, :user:`thehomebrewnerd`

Note
++++
* The update to the DFS algorithm in this release may cause the number of features returned
  by ``ft.dfs`` to increase in some cases.

v1.8.0 Mar 31, 2022
===================
    * Changes
        * Removed ``make_trans_primitive`` and ``make_agg_primitive`` utility functions (:pr:`1970`)
    * Documentation Changes
        * Update project urls in setup cfg to include Twitter and Slack (:pr:`1981`)
        * Update nbconvert to version 6.4.5 to fix docs build issue (:pr:`1984`)
        * Update ReadMe to have centered badges and add docs badge (:pr:`1993`)
        * Add M1 installation instructions to docs and contributing (:pr:`1997`)
    * Testing Changes
        * Updated scheduled workflows to only run on Alteryx owned repos (:pr:`1973`)
        * Updated minimum dependency checker to use new version with write file support (:pr:`1975`, :pr:`1976`)
        * Add black linting package and remove autopep8 (:pr:`1978`)
        * Update tests for compatibility with Woodwork version 0.15.0 (:pr:`1984`)

    Thanks to the following people for contributing to this release:
    :user:`gsheni`, :user:`thehomebrewnerd`

Breaking Changes
++++++++++++++++
* The utility functions ``make_trans_primitive`` and ``make_agg_primitive`` have been removed. To create custom
  primitives, define the primitive class directly.

v1.7.0 Mar 16, 2022
===================
    * Enhancements
        * Add support for Python 3.10 (:pr:`1940`)
        * Added the SquareRoot, NaturalLogarithm, Sine, Cosine and Tangent primitives (:pr:`1948`)
    * Fixes
        * Updated the conda install commands to specify the channel (:pr:`1917`)
    * Changes
        * Update error message when DFS returns an empty list of features (:pr:`1919`)
        * Remove ``list_variable_types`` and related directories (:pr:`1929`)
        * Transition to use pyproject.toml and setup.cfg (moving away from setup.py) (:pr:`1941`, :pr:`1950`, :pr:`1952`, :pr:`1954`, :pr:`1957`, :pr:`1964`)
        * Replace Koalas with pandas API on Spark (:pr:`1949`)
    * Documentation Changes
        * Add time series guide (:pr:`1896`)
        * Update minimum nlp_primitives requirement for docs (:pr:`1925`)
        * Add GitHub URL for PyPi (:pr:`1928`)
        * Add backport release support (:pr:`1932`)
        * Update instructions in ``release.md`` (:pr:`1963`)
    * Testing Changes
        * Update test cases to cover __main__.py file (:pr:`1927`)
        * Upgrade moto requirement (:pr:`1929`, :pr:`1938`)
        * Add Python 3.9 linting, install complete, and docs build CI tests (:pr:`1934`)
        * Add CI workflow to test with latest woodwork main branch (:pr:`1936`)
        * Add lower bound for wheel for minimum dependency checker and limit lint CI tests to Python 3.10 (:pr:`1945`)
        * Fix non-deterministic test in ``test_es.py`` (:pr:`1961`)

    Thanks to the following people for contributing to this release:
    :user:`andriyor`, :user:`gsheni`, :user:`jeff-hernandez`, :user:`kushal-gopal`, :user:`mingdavidqi`, :user:`rwedge`, :user:`tamargrey`, :user:`thehomebrewnerd`, :user:`tvdboom`

Breaking Changes
++++++++++++++++
* The deprecated utility ``list_variable_types`` has been removed from Featuretools.

v1.6.0 Feb 17, 2022
===================
    * Enhancements
        * Add ``IsFederalHoliday`` transform primitive (:pr:`1912`)
    * Fixes
        * Fix to catch new ``NotImplementedError`` raised by ``holidays`` library for unknown country (:pr:`1907`)
    * Changes
        * Remove outdated pandas workaround code (:pr:`1906`)
    * Documentation Changes
        * Add in-line tabs and copy-paste functionality to docs (:pr:`1905`)
    * Testing Changes
        * Fix URL deserialization file (:pr:`1909`)

    Thanks to the following people for contributing to this release:
    :user:`jeff-hernandez`, :user:`rwedge`, :user:`thehomebrewnerd`


v1.5.0 Feb 14, 2022
===================
    .. warning::
        Featuretools may not support Python 3.7 in next non-bugfix release.

    * Enhancements
        * Add ability to use offset alias strings as inputs to rolling primitives (:pr:`1809`)
        * Update to add support for pandas version 1.4.0 (:pr:`1881`, :pr:`1895`)
    * Fixes
        * Fix ``featuretools_primitives`` entry point (:pr:`1891`)
    * Changes
        * Allow only snake camel and title case for primitives (:pr:`1854`)
        * Add autonormalize as an add-on library (:pr:`1840`)
        * Add DateToHoliday Transform Primitive (:pr:`1848`)
        * Add DistanceToHoliday Transform Primitive (:pr:`1853`)
        * Temporarily restrict pandas and koalas max versions (:pr:`1863`)
        * Add ``__setitem__`` method to overload ``add_dataframe`` method on EntitySet (:pr:`1862`)
        * Add support for woodwork 0.12.0 (:pr:`1872`, :pr:`1897`)
        * Split Datetime and LatLong primitives into separate files (:pr:`1861`)
        * Null values will not be included in index of normalized dataframe (:pr:`1897`)
    * Documentation Changes
        * Bump ipython version (:pr:`1857`)
        * Update README.md with Alteryx link (:pr:`1886`)
    * Testing Changes
        * Add check for package conflicts with install workflow (:pr:`1843`)
        * Change auto approve workflow to use assignee (:pr:`1843`)
        * Update auto approve workflow to delete branch and change on trigger (:pr:`1852`)
        * Upgrade tests to use compose version 0.8.0 (:pr:`1856`)
        * Updated deep feature synthesis and feature serialization tests to use new primitive files (:pr:`1861`)

    Thanks to the following people for contributing to this release:
    :user:`dvreed77`, :user:`gsheni`, :user:`jacobboney`, :user:`jeff-hernandez`, :user:`rwedge`, :user:`tamargrey`, :user:`thehomebrewnerd`, :user:`tuethan1999`

Breaking Changes
++++++++++++++++
* When using ``normalize_dataframe`` to create a new dataframe, the new dataframe's index will not include a null value.

v1.4.0 Jan 10, 2022
===================
    * Enhancements
        * Add LatLong transform primitives - GeoMidpoint, IsInGeoBox, CityblockDistance (:pr:`1814`)
        * Add issue templates for bugs, feature requests and documentation improvements (:pr:`1834`)
    * Fixes
        * Fix bug where Woodwork initialization could fail on feature matrix if cutoff times caused null values to be introduced (:pr:`1810`)
    * Changes
        * Skip code coverage for specific dask usage lines (:pr:`1829`)
        * Increase minimum required numpy version to 1.21.0, scipy to 1.3.3, koalas to 1.8.1 (:pr:`1833`)
        * Remove pyyaml as a requirement (:pr:`1833`)
    * Documentation Changes
        * Remove testing on conda forge in release.md (:pr:`1811`)
    * Testing Changes
        * Enable auto-merge for minimum and latest dependency merge requests (:pr:`1818`, :pr:`1821`, :pr:`1822`)
        * Change auto approve workfow to use PR number and run every 30 minutes (:pr:`1827`)
        * Add auto approve workflow to run when unit tests complete (:pr:`1837`)
        * Test deserializing from S3 with mocked S3 fixtures only (:pr:`1825`)
        * Remove fastparquet as a test requirement (:pr:`1833`)

    Thanks to the following people for contributing to this release:
    :user:`davesque`, :user:`gsheni`, :user:`rwedge`, :user:`thehomebrewnerd`

v1.3.0 Dec 2, 2021
==================
    * Enhancements
        * Add ``NumericLag`` transform primitive (:pr:`1797`)
    * Changes
        * Update pip to 21.3.1 for test requirements (:pr:`1789`)
    * Documentation Changes
        * Add Docker install instructions and documentation on the install page. (:pr:`1785`)
        * Update install page on documentation with correct python version (:pr:`1784`)
        * Fix formatting in Improving Computational Performance guide (:pr:`1786`)

    Thanks to the following people for contributing to this release:
    :user:`gsheni`, :user:`HenryRocha`, :user:`tamargrey` :user:`thehomebrewnerd`

v1.2.0 Nov 15, 2021
===================
    * Enhancements
        * Add Rolling Transform primitives with integer parameters (:pr:`1770`)
    * Fixes
        * Handle new graphviz FORMATS import (:pr:`1770`)
    * Changes
        * Add new version of featuretools_tsfresh_primitives as an add-on library (:pr:`1772`)
        * Add ``load_weather`` as demo dataset for time series :pr:`1777`

    Thanks to the following people for contributing to this release:
    :user:`gsheni`, :user:`tamargrey`

v1.1.0 Nov 2, 2021
==================
    * Fixes
        * Check ``base_of_exclude`` attribute on primitive instead feature class (:pr:`1749`)
        * Pin upper bound for pyspark (:pr:`1748`)
        * Fix ``get_unused_primitives`` only recognizes lowercase primitive strings (:pr:`1733`)
        * Require newer versions of dask and distributed (:pr:`1762`)
        * Fix bug with pass-through columns of cutoff_time df when n_jobs > 1 (:pr:`1765`)
    * Changes
        * Add new version of nlp_primitives as an add-on library (:pr:`1743`)
        * Change name of date_of_birth (column name) to birthday in mock dataset (:pr:`1754`)
    * Documentation Changes
        * Upgrade Sphinx and fix docs configuration error (:pr:`1760`)
    * Testing Changes
        * Modify CI to run unit test with latest dependencies on python 3.9 (:pr:`1738`)
        * Added Python version standardizer to Jupyter notebook linting (:pr:`1741`)

    Thanks to the following people for contributing to this release:
    :user:`bchen1116`, :user:`gsheni`, :user:`HenryRocha`, :user:`jeff-hernandez`, :user:`ridicolos`, :user:`rwedge`

v1.0.0 Oct 12, 2021
===================
    * Enhancements
        * Add support for creating EntitySets from Woodwork DataTables (:pr:`1277`)
        * Add ``EntitySet.__deepcopy__`` that retains Woodwork typing information (:pr:`1465`)
        * Add ``EntitySet.__getstate__`` and ``EntitySet.__setstate__`` to preserve typing when pickling (:pr:`1581`)
        * Returned feature matrix has woodwork typing information (:pr:`1664`)
    * Fixes
        * Fix ``DFSTransformer`` Documentation for Featuretools 1.0 (:pr:`1605`)
        * Fix ``calculate_feature_matrix`` time type check and ``encode_features`` for synthesis tests (:pr:`1580`)
        * Revert reordering of categories in ``Equal`` and ``NotEqual`` primitives (:pr:`1640`)
        * Fix bug in ``EntitySet.add_relationship`` that caused ``foreign_key`` tag to be lost (:pr:`1675`)
        * Update DFS to not build features on last time index columns in dataframes (:pr:`1695`)
    * Changes
        * Remove ``add_interesting_values`` from ``Entity`` (:pr:`1269`)
        * Move ``set_secondary_time_index`` method from ``Entity`` to ``EntitySet`` (:pr:`1280`)
        * Refactor Relationship creation process (:pr:`1370`)
        * Replaced ``Entity.update_data`` with ``EntitySet.update_dataframe`` (:pr:`1398`)
        * Move validation check for uniform time index to ``EntitySet`` (:pr:`1400`)
        * Replace ``Entity`` objects in ``EntitySet`` with Woodwork dataframes (:pr:`1405`)
        * Refactor ``EntitySet.plot`` to work with Woodwork dataframes (:pr:`1468`)
        * Move ``last_time_index`` to be a column on the DataFrame (:pr:`1456`)
        * Update serialization/deserialization to work with Woodwork (:pr:`1452`)
        * Refactor ``EntitySet.query_by_values`` to work with Woodwork dataframes (:pr:`1467`)
        * Replace ``list_variable_types`` with ``list_logical_types`` (:pr:`1477`)
        * Allow deep EntitySet equality check (:pr:`1480`)
        * Update ``EntitySet.concat`` to work with Woodwork DataFrames (:pr:`1490`)
        * Add function to list semantic tags (:pr:`1486`)
        * Initialize Woodwork on feature matrix in ``remove_highly_correlated_features`` if necessary (:pr:`1618`)
        * Remove categorical-encoding as an add-on library (will be added back later) (:pr:`1632`)
        * Remove autonormalize as an add-on library (will be added back later) (:pr:`1636`)
        * Remove tsfresh, nlp_primitives, sklearn_transformer as an add-on library (will be added back later) (:pr:`1638`)
        * Update input and return types for ``CumCount`` primitive (:pr:`1651`)
        * Standardize imports of Woodwork (:pr:`1526`)
        * Rename target entity to target dataframe (:pr:`1506`)
        * Replace ``entity_from_dataframe`` with ``add_dataframe`` (:pr:`1504`)
        * Create features from Woodwork columns (:pr:`1582`)
        * Move default variable description logic to ``generate_description`` (:pr:`1403`)
        * Update Woodwork to version 0.4.0 with ``LogicalType.transform`` and LogicalType instances (:pr:`1451`)
        * Update Woodwork to version 0.4.1 with Ordinal order values and whitespace serialization fix (:pr:`1478`)
        * Use ``ColumnSchema`` for primitive input and return types (:pr:`1411`)
        * Update features to use Woodwork and remove ``Entity`` and ``Variable`` classes (:pr:`1501`)
        * Re-add ``make_index`` functionality to EntitySet (:pr:`1507`)
        * Use ``ColumnSchema`` in DFS primitive matching (:pr:`1523`)
        * Updates from Featuretools v0.26.0 (:pr:`1539`)
        * Leverage Woodwork better in ``add_interesting_values`` (:pr:`1550`)
        * Update ``calculate_feature_matrix`` to use Woodwork (:pr:`1533`)
        * Update Woodwork to version 0.6.0 with changed categorical inference (:pr:`1597`)
        * Update ``nlp-primitives`` requirement for Featuretools 1.0 (:pr:`1609`)
        * Remove remaining references to ``Entity`` and ``Variable`` in code (:pr:`1612`)
        * Update Woodwork to version 0.7.1 with changed initialization (:pr:`1648`)
        * Removes outdated workaround code related to a since-resolved pandas issue (:pr:`1677`)
        * Remove unused ``_dataframes_equal`` and ``camel_to_snake`` functions (:pr:`1683`)
        * Update Woodwork to version 0.8.0 for improved performance (:pr:`1689`)
        * Remove redundant typecasting in ``encode_features`` (:pr:`1694`)
        * Speed up ``encode_features`` if not inplace, some space cost (:pr:`1699`)
        * Clean up comments and commented out code (:pr:`1701`)
        * Update Woodwork to version 0.8.1 for improved performance (:pr:`1702`)
    * Documentation Changes
        * Add a Woodwork Typing in Featuretools guide (:pr:`1589`)
        * Add a resource guide for transitioning to Featuretools 1.0 (:pr:`1627`)
        * Update ``using_entitysets`` page to use Woodwork (:pr:`1532`)
        * Update FAQ page to use Woodwork integration (:pr:`1649`)
        * Update DFS page to be Jupyter notebook and use Woodwork integration (:pr:`1557`)
        * Update Feature Primitives page to be Jupyter notebook and use Woodwork integration (:pr:`1556`)
        * Update Handling Time page to be Jupyter notebook and use Woodwork integration (:pr:`1552`)
        * Update Advanced Custom Primitives page to be Jupyter notebook and use Woodwork integration (:pr:`1587`)
        * Update Deployment page to use Woodwork integration (:pr:`1588`)
        * Update Using Dask EntitySets page to be Jupyter notebook and use Woodwork integration (:pr:`1590`)
        * Update Specifying Primitive Options page to be Jupyter notebook and use Woodwork integration (:pr:`1593`)
        * Update API Reference to match Featuretools 1.0 API (:pr:`1600`)
        * Update Index page to be Jupyter notebook and use Woodwork integration (:pr:`1602`)
        * Update Feature Descriptions page to be Jupyter notebook and use Woodwork integration (:pr:`1603`)
        * Update Using Koalas EntitySets page to be Jupyter notebook and use Woodwork integration (:pr:`1604`)
        * Update Glossary to use Woodwork integration (:pr:`1608`)
        * Update Tuning DFS page to be Jupyter notebook and use Woodwork integration (:pr:`1610`)
        * Fix small formatting issues in Documentation (:pr:`1607`)
        * Remove Variables page and more references to variables (:pr:`1629`)
        * Update Feature Selection page to use Woodwork integration (:pr:`1618`)
        * Update Improving Performance page to be Jupyter notebook and use Woodwork integration (:pr:`1591`)
        * Fix typos in transition guide (:pr:`1672`)
        * Update installation instructions for 1.0.0rc1 announcement in docs (:pr:`1707`, :pr:`1708`, :pr:`1713`, :pr:`1716`)
        * Fixed broken link for Demo notebook in README.md (:pr:`1728`)
        * Update ``contributing.md`` to improve instructions for external contributors (:pr:`1723`)
        * Manually revert changes made by :pr:`1677` and :pr:`1679`.  The related bug in pandas still exists. (:pr:`1731`)
    * Testing Changes
        * Remove entity tests (:pr:`1521`)
        * Fix broken ``EntitySet`` tests (:pr:`1548`)
        * Fix broken primitive tests (:pr:`1568`)
        * Added Jupyter notebook cleaner to the linters (:pr:`1719`)
        * Update reviewers for minimum and latest dependency checkers (:pr:`1715`)
        * Full coverage for EntitySet.__eq__ method (:pr:`1725`)
        * Add tests to verify all primitives can be initialized without parameter values (:pr:`1726`)

    Thanks to the following people for contributing to this release:
    :user:`bchen1116`, :user:`gsheni`, :user:`HenryRocha`, :user:`jeff-hernandez`, :user:`rwedge`, :user:`tamargrey`, :user:`thehomebrewnerd`, :user:`VaishnaviNandakumar`

Breaking Changes
++++++++++++++++

* ``Entity.add_interesting_values`` has been removed. To add interesting values for a single
  entity, call ``EntitySet.add_interesting_values`` and pass the name of the dataframe for
  which to add interesting values in the ``dataframe_name`` parameter (:pr:`1405`, :pr:`1370`).
* ``Entity.set_secondary_time_index`` has been removed and replaced by ``EntitySet.set_secondary_time_index``
  with an added ``dataframe_name`` parameter to specify the dataframe on which to set the secondary time index (:pr:`1405`, :pr:`1370`).
* ``Relationship`` initialization has been updated to accept four name values for the parent dataframe,
  parent column, child dataframe and child column instead of accepting two ``Variable`` objects  (:pr:`1405`, :pr:`1370`).
* ``EntitySet.add_relationship`` has been updated to accept dataframe and column name values or a
  ``Relationship`` object. Adding a relationship from a ``Relationship`` object now requires passing
  the relationship as a keyword argument  (:pr:`1405`, :pr:`1370`).
* ``Entity.update_data`` has been removed. To update the dataframe, call ``EntitySet.replace_dataframe`` and use the ``dataframe_name`` parameter (:pr:`1630`, :pr:`1522`).
* The data in an ``EntitySet`` is no longer stored in ``Entity`` objects. Instead, dataframes
  with Woodwork typing information are used. Accordingly, most language referring to “entities”
  will now refer to “dataframes”, references to “variables” will now refer to “columns”, and
  “variable types” will use the Woodwork type system’s “logical types” and “semantic tags” (:pr:`1405`).
* The dictionary of tuples passed to ``EntitySet.__init__`` has replaced the ``variable_types`` element
  with separate ``logical_types`` and ``semantic_tags`` dictionaries (:pr:`1405`).
* ``EntitySet.entity_from_dataframe`` no longer exists. To add new tables to an entityset, use``EntitySet.add_dataframe`` (:pr:`1405`).
* ``EntitySet.normalize_entity`` has been renamed to ``EntitySet.normalize_dataframe`` (:pr:`1405`).
* Instead of raising an error at ``EntitySet.add_relationship`` when the dtypes of parent and child columns
  do not match, Featuretools will now check whether the Woodwork logical type of the parent and child columns
  match. If they do not match, there will now be a warning raised, and Featuretools will attempt to update
  the logical type of the child column to match the parent’s (:pr:`1405`).
* If no index is specified at ``EntitySet.add_dataframe``, the first column will only be used as index if
  Woodwork has not been initialized on the DataFrame. When adding a dataframe that already has Woodwork
  initialized, if there is no index set, an error will be raised (:pr:`1405`).
* Featuretools will no longer re-order columns in DataFrames so that the index column is the first column of the DataFrame (:pr:`1405`).
* Type inference can now be performed on Dask and Koalas dataframes, though a warning will be issued
  indicating that this may be computationally intensive (:pr:`1405`).
* EntitySet.time_type is no longer stored as Variable objects. Instead, Woodwork typing is used, and a
  numeric time type will be indicated by the ``'numeric'`` semantic tag string, and a datetime time type
  will be indicated by the ``Datetime`` logical type (:pr:`1405`).
* ``last_time_index``, ``secondary_time_index``, and ``interesting_values`` are no longer attributes
  of an entityset’s tables that can be accessed directly. Now they must be accessed through the metadata
  of the Woodwork DataFrame, which is a dictionary (:pr:`1405`).
* The helper function ``list_variable_types`` will be removed in a future release and replaced by ``list_logical_types``.
  In the meantime, ``list_variable_types`` will return the same output as ``list_logical_types`` (:pr:`1447`).

What's New in this Release
++++++++++++++++++++++++++

**Adding Interesting Values**

To add interesting values for a single entity, call ``EntitySet.add_interesting_values`` passing the
id of the dataframe for which interesting values should be added.

.. code-block:: python

    >>> es.add_interesting_values(dataframe_name='log')

**Setting a Secondary Time Index**

To set a secondary time index for a specific dataframe, call ``EntitySet.set_secondary_time_index`` passing
the dataframe name for which to set the secondary time index along with the dictionary mapping the secondary time
index column to the for which the secondary time index applies.

.. code-block:: python

    >>> customers_secondary_time_index = {'cancel_date': ['cancel_reason']}
    >>> es.set_secondary_time_index(dataframe_name='customers', customers_secondary_time_index)

**Creating a Relationship and Adding to an EntitySet**

Relationships are now created by passing parameters identifying the entityset along with four string values
specifying the parent dataframe, parent column, child dataframe and child column. Specifying parameter names
is optional.

.. code-block:: python

    >>> new_relationship = Relationship(
    ...     entityset=es,
    ...     parent_dataframe_name='customers',
    ...     parent_column_name='id',
    ...     child_dataframe_name='sessions',
    ...     child_column_name='customer_id'
    ... )

Relationships can now be added to EntitySets in one of two ways. The first approach is to pass in
name values for the parent dataframe, parent column, child dataframe and child column. Specifying
parameter names is optional with this approach.

.. code-block:: python

    >>> es.add_relationship(
    ...     parent_dataframe_name='customers',
    ...     parent_column_name='id',
    ...     child_dataframe_name='sessions',
    ...     child_column_name='customer_id'
    ... )

Relationships can also be added by passing in a previously created ``Relationship`` object. When using
this approach the ``relationship`` parameter name must be included.

.. code-block:: python

    >>> es.add_relationship(relationship=new_relationship)

**Replace DataFrame**

To replace a dataframe in an EntitySet with a new dataframe, call ``EntitySet.replace_dataframe`` and pass in the name of the dataframe to replace along with the new data.

.. code-block:: python

    >>> es.replace_dataframe(dataframe_name='log', df=df)

**List Logical Types and Semantic Tags**

Logical types and semantic tags have replaced variable types to parse and interpret columns. You can list all the available logical types by calling ``featuretools.list_logical_types``.

.. code-block:: python

    >>> ft.list_logical_types()

You can list all the available semantic tags by calling ``featuretools.list_semantic_tags``.

.. code-block:: python

    >>> ft.list_semantic_tags()

v0.27.1 Sep 2, 2021
===================
    * Documentation Changes
        * Add banner to docs about upcoming Featuretools 1.0 release (:pr:`1669`)

    Thanks to the following people for contributing to this release:
    :user:`thehomebrewnerd`

v0.27.0 Aug 31, 2021
====================
    * Changes
        * Remove autonormalize, tsfresh, nlp_primitives, sklearn_transformer, caegorical_encoding as an add-on libraries (will be added back later) (:pr:`1644`)
        * Emit a warning message when a ``featuretools_primitives`` entrypoint
          throws an exception (:pr:`1662`)
        * Throw a ``RuntimeError`` when two primitives with the same name are
          encountered during ``featuretools_primitives`` entrypoint handling
          (:pr:`1662`)
        * Prevent the ``featuretools_primitives`` entrypoint loader from
          loading non-class objects as well as the ``AggregationPrimitive`` and
          ``TransformPrimitive`` base classes (:pr:`1662`)
    * Testing Changes
        * Update latest dependency checker with proper install command (:pr:`1652`)
        * Update isort dependency (:pr:`1654`)

    Thanks to the following people for contributing to this release:
    :user:`davesque`, :user:`gsheni`, :user:`jeff-hernandez`, :user:`rwedge`

v0.26.2 Aug 17, 2021
====================
    * Documentation Changes
        * Specify conda channel and Windows exe in graphviz installation instructions (:pr:`1611`)
        * Remove GA token from the layout html (:pr:`1622`)
    * Testing Changes
        * Add additional reviewers to minimum and latest dependency checkers (:pr:`1558`, :pr:`1562`, :pr:`1564`, :pr:`1567`)

    Thanks to the following people for contributing to this release:
    :user:`gsheni`, :user:`simha104`

v0.26.1 Jul 23, 2021
====================
    * Fixes
        * Set ``name`` attribute for ``EmailAddressToDomain`` primitive (:pr:`1543`)
    * Documentation Changes
        * Remove and ignore unnecessary graph files (:pr:`1544`)

    Thanks to the following people for contributing to this release:
    :user:`davesque`, :user:`rwedge`

v0.26.0 Jul 15, 2021
====================
    * Enhancements
        * Add ``replace_inf_values`` utility function for replacing ``inf`` values in a feature matrix (:pr:`1505`)
        * Add URLToProtocol, URLToDomain, URLToTLD, EmailAddressToDomain, IsFreeEmailDomain as transform primitives (:pr:`1508`, :pr:`1531`)
    * Fixes
        * ``include_entities`` correctly overrides ``exclude_entities`` in ``primitive_options`` (:pr:`1518`)
    * Documentation Changes
        * Prevent logging on build (:pr:`1498`)
    * Testing Changes
        * Test featuretools on pandas 1.3.0 release candidate and make fixes (:pr:`1492`)

    Thanks to the following people for contributing to this release:
    :user:`frances-h`, :user:`gsheni`, :user:`rwedge`, :user:`tamargrey`, :user:`thehomebrewnerd`, :user:`tuethan1999`

v0.25.0 Jun 11, 2021
====================
    * Enhancements
       * Add ``get_valid_primitives`` function (:pr:`1462`)
       * Add ``EntitySet.dataframe_type`` attribute (:pr:`1473`)
    * Changes
        * Upgrade minimum alteryx open source update checker to 2.0.0 (:pr:`1460`)
    * Testing Changes
        * Upgrade minimum pip requirement for testing to 21.1.2 (:pr:`1475`)

    Thanks to the following people for contributing to this release:
    :user:`gsheni`, :user:`rwedge`

v0.24.1 May 26, 2021
====================
    * Fixes
        * Update minimum pyyaml requirement to 5.4 (:pr:`1433`)
        * Update minimum psutil requirement to 5.6.6 (:pr:`1438`)
    * Documentation Changes
        * Update nbsphinx version to fix docs build issue (:pr:`1436`)
    * Testing Changes
        * Create separate worksflows for each CI job (:pr:`1422`)
        * Add minimum dependency checker to generate minimum requirement files (:pr:`1428`)
        * Add unit tests against minimum dependencies for python 3.7 on PRs and main (:pr:`1432`, :pr:`1445`)
        * Update minimum urllib3 requirement to 1.26.5 (:pr:`1457`)

    Thanks to the following people for contributing to this release:
    :user:`gsheni`, :user:`jeff-hernandez`, :user:`rwedge`, :user:`thehomebrewnerd`

v0.24.0 Apr 30, 2021
====================
    * Changes
        * Add auto assign bot on GitHub (:pr:`1380`)
        * Reduce DFS max_depth to 1 if single entity in entityset (:pr:`1412`)
        * Drop Python 3.6 support (:pr:`1413`)
    * Documentation Changes
        * Improve formatting of release notes (:pr:`1396`)
    * Testing Changes
        * Update Dask/Koalas test fixtures (:pr:`1382`)
        * Update Spark config in test fixtures and docs (:pr:`1387`, :pr:`1389`)
        * Don't cancel other CI jobs if one fails (:pr:`1386`)
        * Update boto3 and urllib3 version requirements (:pr:`1394`)
        * Update token for dependency checker PR creation (:pr:`1402`, :pr:`1407`, :pr:`1409`)

    Thanks to the following people for contributing to this release:
    :user:`gsheni`, :user:`jeff-hernandez`, :user:`rwedge`, :user:`tamargrey`, :user:`thehomebrewnerd`

v0.23.3 Mar 31, 2021
====================
    .. warning::
        The next non-bugfix release of Featuretools will not support Python 3.6

    * Changes
        * Minor updates to work with Koalas version 1.7.0 (:pr:`1351`)
        * Explicitly mention Python 3.8 support in setup.py classifiers (:pr:`1371`)
        * Fix issue with smart-open version 5.0.0 (:pr:`1372`, :pr:`1376`)
    * Testing Changes
        * Make release notes updated check separate from unit tests (:pr:`1347`)
        * Performance tests now specify which commit to check (:pr:`1354`)

    Thanks to the following people for contributing to this release:
    :user:`gsheni`, :user:`rwedge`, :user:`thehomebrewnerd`

v0.23.2 Feb 26, 2021
====================
    .. warning::
        The next non-bugfix release of Featuretools will not support Python 3.6

    * Enhancements
        * The ``list_primitives`` function returns valid input types and the return type (:pr:`1341`)
    * Fixes
        * Restrict numpy version when installing koalas (:pr:`1329`)
    * Changes
        * Warn python 3.6 users support will be dropped in future release (:pr:`1344`)
    * Documentation Changes
        * Update docs for defining custom primitives (:pr:`1332`)
        * Update featuretools release instructions (:pr:`1345`)

    Thanks to the following people for contributing to this release:
    :user:`gsheni`, :user:`jeff-hernandez`, :user:`rwedge`

v0.23.1 Jan 29, 2021
====================
    * Fixes
        * Calculate direct features uses default value if parent missing (:pr:`1312`)
        * Fix bug and improve tests for ``EntitySet.__eq__`` and ``Entity.__eq__`` (:pr:`1323`)
    * Documentation Changes
        * Update Twitter link to documentation toolbar (:pr:`1322`)
    * Testing Changes
        * Unpin python-graphviz package on Windows (:pr:`1296`)
        * Reorganize and clean up tests (:pr:`1294`, :pr:`1303`, :pr:`1306`)
        * Trigger tests on pull request events (:pr:`1304`, :pr:`1315`)
        * Remove unnecessary test skips on Windows (:pr:`1320`)

    Thanks to the following people for contributing to this release:
    :user:`gsheni`, :user:`jeff-hernandez`, :user:`rwedge`, :user:`seriallazer`, :user:`thehomebrewnerd`

v0.23.0 Dec 31, 2020
====================
    * Fixes
        * Fix logic for inferring variable type from unusual dtype (:pr:`1273`)
        * Allow passing entities without relationships to ``calculate_feature_matrix`` (:pr:`1290`)
    * Changes
        * Move ``query_by_values`` method from ``Entity`` to ``EntitySet`` (:pr:`1251`)
        * Move ``_handle_time`` method from ``Entity`` to ``EntitySet`` (:pr:`1276`)
        * Remove usage of ``ravel`` to resolve unexpected warning with pandas 1.2.0 (:pr:`1286`)
    * Documentation Changes
        * Fix installation command for Add-ons (:pr:`1279`)
        * Fix various broken links in documentation (:pr:`1313`)
    * Testing Changes
        * Use repository-scoped token for dependency check (:pr:`1245`:, :pr:`1248`)
        * Fix install error during docs CI test (:pr:`1250`)

    Thanks to the following people for contributing to this release:
    :user:`gsheni`, :user:`jeff-hernandez`, :user:`rwedge`, :user:`thehomebrewnerd`

Breaking Changes
++++++++++++++++

* ``Entity.query_by_values`` has been removed and replaced by ``EntitySet.query_by_values`` with an
  added ``entity_id`` parameter to specify which entity in the entityset should be used for the query.

v0.22.0 Nov 30, 2020
====================
    * Enhancements
        * Allow variable descriptions to be set directly on variable (:pr:`1207`)
        * Add ability to add feature description captions to feature lineage graphs (:pr:`1212`)
        * Add support for local tar file in read_entityset (:pr:`1228`)
    * Fixes
        * Updates to fix unit test errors from koalas 1.4 (:pr:`1230`, :pr:`1232`)
    * Documentation Changes
        * Removed link to unused feedback board (:pr:`1220`)
        * Update footer with Alteryx Innovation Labs (:pr:`1221`)
        * Update links to repo in documentation to use alteryx org url (:pr:`1224`)
    * Testing Changes
        * Update release notes check to use new repo url (:pr:`1222`)
        * Use new version of pull request Github Action (:pr:`1234`)
        * Upgrade pip during featuretools[complete] test (:pr:`1236`)
        * Migrated CI tests to github actions (:pr:`1226`, :pr:`1237`, :pr:`1239`)

    Thanks to the following people for contributing to this release:
    :user:`frances-h`, :user:`gsheni`, :user:`jeff-hernandez`, :user:`kmax12`, :user:`rwedge`, :user:`thehomebrewnerd`

v0.21.0 Oct 30, 2020
====================
    * Enhancements
        * Add ``describe_feature`` to generate an English language feature description for a given feature (:pr:`1201`)
    * Fixes
        * Update ``EntitySet.add_last_time_indexes`` to work with Koalas 1.3.0 (:pr:`1192`, :pr:`1202`)
    * Changes
        * Keep koalas requirements in separate file (:pr:`1195`)
    * Documentation Changes
        * Added footer to the documentation (:pr:`1189`)
        * Add guide for feature selection functions (:pr:`1184`)
        * Fix README.md badge with correct link (:pr:`1200`)
    * Testing Changes
        * Add ``pyspark`` and ``koalas`` to automated dependency checks (:pr:`1191`)
        * Add DockerHub credentials to CI testing environment (:pr:`1204`)
        * Update premium primitives job name on CI (:pr:`1205`)

    Thanks to the following people for contributing to this release:
    :user:`frances-h`, :user:`gsheni`, :user:`jeff-hernandez`, :user:`rwedge`, :user:`tamargrey`, :user:`thehomebrewnerd`

v0.20.0 Sep 30, 2020
====================
    .. warning::
        The Text variable type has been deprecated and been replaced with the NaturalLanguage variable type. The Text variable type will be removed in a future release.

    * Fixes
        * Allow FeatureOutputSlice features to be serialized (:pr:`1150`)
        * Fix duplicate label column generation when labels are passed in cutoff times and approximate is being used (:pr:`1160`)
        * Determine calculate_feature_matrix behavior with approximate and a cutoff df that is a subclass of a pandas DataFrame (:pr:`1166`)
    * Changes
        * Text variable type has been replaced with NaturalLanguage (:pr:`1159`)
    * Documentation Changes
        * Update release doc for clarity and to add Future Release template (:pr:`1151`)
        * Use the PyData Sphinx theme (:pr:`1169`)
    * Testing Changes
        * Stop requiring single-threaded dask scheduler in tests (:pr:`1163`, :pr:`1170`)

    Thanks to the following people for contributing to this release:
    :user:`gsheni`, :user:`rwedge`, :user:`tamargrey`, :user:`tuethan1999`

v0.19.0 Sep 8, 2020
===================
    * Enhancements
        * Support use of Koalas DataFrames in entitysets (:pr:`1031`)
        * Add feature selection functions for null, correlated, and single value features (:pr:`1126`)
    * Fixes
        * Fix ``encode_features`` converting excluded feature columns to a numeric dtype (:pr:`1123`)
        * Improve performance of unused primitive check in dfs (:pr:`1140`)
    * Changes
        * Remove the ability to stack transform primitives (:pr:`1119`, :pr:`1145`)
        * Sort primitives passed to ``dfs`` to get consistent ordering of features\* (:pr:`1119`)
    * Documentation Changes
        * Added return values to dfs and calculate_feature_matrix (:pr:`1125`)
    * Testing Changes
        * Better test case for normalizing from no time index to time index (:pr:`1113`)

    \* When passing multiple instances of a primitive built with ``make_trans_primitive``
    or ``maxe_agg_primitive``, those instances must have the same relative order when passed
    to ``dfs`` to ensure a consistent ordering of features.

    Thanks to the following people for contributing to this release:
    :user:`frances-h`, :user:`gsheni`, :user:`rwedge`, :user:`tamargrey`, :user:`thehomebrewnerd`, :user:`tuethan1999`


Breaking Changes
++++++++++++++++

* ``ft.dfs`` will no longer build features from Transform primitives where one
  of the inputs is a Transform feature, a GroupByTransform feature,
  or a Direct Feature of a Transform / GroupByTransform feature. This will make some
  features that would previously be generated by ``ft.dfs`` only possible if
  explicitly specified in ``seed_features``.

v0.18.1 Aug 12, 2020
====================
    * Fixes
        * Fix ``EntitySet.plot()`` when given a dask entityset (:pr:`1086`)
    * Changes
        * Use ``nlp-primitives[complete]`` install for ``nlp_primitives`` extra in ``setup.py`` (:pr:`1103`)
    * Documentation Changes
        * Fix broken downloads badge in README.md (:pr:`1107`)
    * Testing Changes
        * Use CircleCI matrix jobs in config to trigger multiple runs of same job with different parameters (:pr:`1105`)

    Thanks to the following people for contributing to this release:
    :user:`gsheni`, :user:`systemshift`, :user:`thehomebrewnerd`

v0.18.0 Jul 31, 2020
====================
    * Enhancements
        * Warn user if supplied primitives are not used during dfs (:pr:`1073`)
    * Fixes
        * Use more consistent and uniform warnings (:pr:`1040`)
        * Fix issue with missing instance ids and categorical entity index (:pr:`1050`)
        * Remove warnings.simplefilter in feature_set_calculator to un-silence warnings (:pr:`1053`)
        * Fix feature visualization for features with '>' or '<' in name (:pr:`1055`)
        * Fix boolean dtype mismatch between encode_features and dfs and calculate_feature_matrix (:pr:`1082`)
        * Update primitive options to check reversed inputs if primitive is commutative (:pr:`1085`)
        * Fix inconsistent ordering of features between kernel restarts (:pr:`1088`)
    * Changes
        * Make DFS match ``TimeSince`` primitive with all ``Datetime`` types (:pr:`1048`)
        * Change default branch to ``main`` (:pr:`1038`)
        * Raise TypeError if improper input is supplied to ``Entity.delete_variables()`` (:pr:`1064`)
        * Updates for compatibility with pandas 1.1.0 (:pr:`1079`, :pr:`1089`)
        * Set pandas version to pandas>=0.24.1,<2.0.0. Filter pandas deprecation warning in Week primitive. (:pr:`1094`)
    * Documentation Changes
        * Remove benchmarks folder (:pr:`1049`)
        * Add custom variables types section to variables page (:pr:`1066`)
    * Testing Changes
        * Add fixture for ``ft.demo.load_mock_customer`` (:pr:`1036`)
        * Refactor Dask test units (:pr:`1052`)
        * Implement automated process for checking critical dependencies (:pr:`1045`, :pr:`1054`, :pr:`1081`)
        * Don't run changelog check for release PRs or automated dependency PRs (:pr:`1057`)
        * Fix non-deterministic behavior in Dask test causing codecov issues (:pr:`1070`)

    Thanks to the following people for contributing to this release:
    :user:`frances-h`, :user:`gsheni`, :user:`monti-python`, :user:`rwedge`,
    :user:`systemshift`,  :user:`tamargrey`, :user:`thehomebrewnerd`, :user:`wsankey`

v0.17.0 Jun 30, 2020
====================
    * Enhancements
        * Add ``list_variable_types`` and ``graph_variable_types`` for Variable Types (:pr:`1013`)
        * Add ``graph_feature`` to generate a feature lineage graph for a given feature (:pr:`1032`)
    * Fixes
        * Improve warnings when using a Dask dataframe for cutoff times (:pr:`1026`)
        * Error if attempting to add entityset relationship where child variable is also child index (:pr:`1034`)
    * Changes
        * Remove ``Feature.get_names`` (:pr:`1021`)
        * Remove unnecessary ``pd.Series`` and ``pd.DatetimeIndex`` calls from primitives (:pr:`1020`, :pr:`1024`)
        * Improve cutoff time handling when a single value or no value is passed (:pr:`1028`)
        * Moved ``find_variable_types`` to Variable utils (:pr:`1013`)
    * Documentation Changes
        * Add page on Variable Types to describe some Variable Types, and util functions (:pr:`1013`)
        * Remove featuretools enterprise from documentation (:pr:`1022`)
        * Add development install instructions to contributing.md (:pr:`1030`)
    * Testing Changes
        * Add ``required`` flag to CircleCI codecov upload command (:pr:`1035`)

    Thanks to the following people for contributing to this release:
    :user:`frances-h`, :user:`gsheni`, :user:`kmax12`, :user:`rwedge`,
    :user:`thehomebrewnerd`, :user:`tuethan1999`

Breaking Changes
++++++++++++++++

* Removed ``Feature.get_names``, ``Feature.get_feature_names`` should be used instead

v0.16.0 Jun 5, 2020
===================
    * Enhancements
        * Support use of Dask DataFrames in entitysets (:pr:`783`)
        * Add ``make_index`` when initializing an EntitySet by passing in an ``entities`` dictionary (:pr:`1010`)
        * Add ability to use primitive classes and instances as keys in primitive_options dictionary (:pr:`993`)
    * Fixes
        * Cleanly close tqdm instance (:pr:`1018`)
        * Resolve issue with ``NaN`` values in ``LatLong`` columns (:pr:`1007`)
    * Testing Changes
        * Update tests for numpy v1.19.0 compatability (:pr:`1016`)

    Thanks to the following people for contributing to this release:
    :user:`Alex-Monahan`, :user:`frances-h`, :user:`gsheni`, :user:`rwedge`, :user:`thehomebrewnerd`

v0.15.0 May 29, 2020
====================
    * Enhancements
        * Add ``get_default_aggregation_primitives`` and ``get_default_transform_primitives`` (:pr:`945`)
        * Allow cutoff time dataframe columns to be in any order (:pr:`969`, :pr:`995`)
        * Add Age primitive, and make it a default transform primitive for DFS (:pr:`987`)
        * Add ``include_cutoff_time`` arg - control whether data at cutoff times are included in feature calculations (:pr:`959`)
        * Allow ``variables_types`` to be referenced by their ``type_string``
          for the ``entity_from_dataframe`` function (:pr:`988`)
    * Fixes
        * Fix errors with Equals and NotEquals primitives when comparing categoricals or different dtypes (:pr:`968`)
        * Normalized type_strings of ``Variable`` classes so that the ``find_variable_types`` function produces a
          dictionary with a clear key to name transition (:pr:`982`, :pr:`996`)
        * Remove pandas.datetime in test_calculate_feature_matrix due to deprecation (:pr:`998`)
    * Documentation Changes
        * Add python 3.8 support for docs (:pr:`983`)
        * Adds consistent Entityset Docstrings (:pr:`986`)
    * Testing Changes
        * Add automated tests for python 3.8 environment (:pr:`847`)
        * Update testing dependencies (:pr:`976`)

    Thanks to the following people for contributing to this release:
    :user:`ctduffy`, :user:`frances-h`, :user:`gsheni`, :user:`jeff-hernandez`, :user:`rightx2`, :user:`rwedge`, :user:`sebrahimi1988`, :user:`thehomebrewnerd`,  :user:`tuethan1999`

Breaking Changes
++++++++++++++++

* Calls to ``featuretools.dfs`` or ``featuretools.calculate_feature_matrix`` that use a cutoff time
  dataframe, but do not label the time column with either the target entity time index variable name or
  as ``time``, will now result in an ``AttributeError``. Previously, the time column was selected to be the first
  column that was not the instance id column. With this update, the position of the column in the dataframe is
  no longer used to determine the time column. Now, both instance id columns and time columns in a cutoff time
  dataframe can be in any order as long as they are named properly.

* The ``type_string`` attributes of all ``Variable`` subclasses are now a snake case conversion of their class names. This
  changes the ``type_string`` of the ``Unknown``, ``IPAddress``, ``EmailAddress``, ``SubRegionCode``, ``FilePath``, ``LatLong``, and ``ZIPcode`` classes.
  Old saved entitysets that used these variables may load incorrectly.

v0.14.0 Apr 30, 2020
====================
    * Enhancements
        * ft.encode_features - use less memory for one-hot encoded columns (:pr:`876`)
    * Fixes
        * Use logger.warning to fix deprecated logger.warn (:pr:`871`)
        * Add dtype to interesting_values to fix deprecated empty Series with no dtype (:pr:`933`)
        * Remove overlap in training windows (:pr:`930`)
        * Fix progress bar in notebook (:pr:`932`)
    * Changes
        * Change premium primitives CI test to Python 3.6 (:pr:`916`)
        * Remove Python 3.5 support (:pr:`917`)
    * Documentation Changes
        * Fix README links to docs (:pr:`872`)
        * Fix Github links with correct organizations (:pr:`908`)
        * Fix hyperlinks in docs and docstrings with updated address (:pr:`910`)
        * Remove unused script for uploading docs to AWS (:pr:`911`)

    Thanks to the following people for contributing to this release:
    :user:`frances-h`, :user:`gsheni`, :user:`jeff-hernandez`, :user:`rwedge`

Breaking Changes
++++++++++++++++

* Using training windows in feature calculations can result in different values than previous versions.
  This was done to prevent consecutive training windows from overlapping by excluding data at the oldest point in time.
  For example, if we use a cutoff time at the first minute of the hour with a one hour training window,
  the first minute of the previous hour will no longer be included in the feature calculation.

v0.13.4 Mar 27, 2020
====================
    .. warning::
        The next non-bugfix release of Featuretools will not support Python 3.5

    * Fixes
        * Fix ft.show_info() not displaying in Jupyter notebooks (:pr:`863`)
    * Changes
        * Added Plugin Warnings at Entry Point (:pr:`850`, :pr:`869`)
    * Documentation Changes
        * Add links to primitives.featurelabs.com (:pr:`860`)
        * Add source code links to API reference (:pr:`862`)
        * Update links for testing Dask/Spark integrations (:pr:`867`)
        * Update release documentation for featuretools (:pr:`868`)
    * Testing Changes
        * Miscellaneous changes (:pr:`861`)

    Thanks to the following people for contributing to this release:
    :user:`frances-h`, :user:`FreshLeaf8865`, :user:`jeff-hernandez`, :user:`rwedge`, :user:`thehomebrewnerd`

v0.13.3 Feb 28, 2020
====================
    * Fixes
        * Fix a connection closed error when using n_jobs (:pr:`853`)
    * Changes
        * Pin msgpack dependency for Python 3.5; remove dataframe from Dask dependency (:pr:`851`)
    * Documentation Changes
        * Update link to help documentation page in Github issue template (:pr:`855`)

    Thanks to the following people for contributing to this release:
    :user:`frances-h`, :user:`rwedge`

v0.13.2 Jan 31, 2020
====================
    * Enhancements
        * Support for Pandas 1.0.0 (:pr:`844`)
    * Changes
        * Remove dependency on s3fs library for anonymous downloads from S3 (:pr:`825`)
    * Testing Changes
        * Added GitHub Action to automatically run performance tests (:pr:`840`)

    Thanks to the following people for contributing to this release:
    :user:`frances-h`, :user:`rwedge`

v0.13.1 Dec 28, 2019
====================
    * Fixes
        * Raise error when given wrong input for ignore_variables (:pr:`826`)
        * Fix multi-output features not created when there is no child data (:pr:`834`)
        * Removing type casting in Equals and NotEquals primitives (:pr:`504`)
    * Changes
        * Replace pd.timedelta time units that were deprecated (:pr:`822`)
        * Move sklearn wrapper to separate library (:pr:`835`, :pr:`837`)
    * Testing Changes
        * Run unit tests in windows environment (:pr:`790`)
        * Update boto3 version requirement for tests (:pr:`838`)

    Thanks to the following people for contributing to this release:
    :user:`jeffzi`, :user:`kmax12`, :user:`rwedge`, :user:`systemshift`

v0.13.0 Nov 30, 2019
====================
    * Enhancements
        * Added GitHub Action to auto upload releases to PyPI (:pr:`816`)
    * Fixes
        * Fix issue where some primitive options would not be applied (:pr:`807`)
        * Fix issue with converting to pickle or parquet after adding interesting features (:pr:`798`, :pr:`823`)
        * Diff primitive now calculates using all available data (:pr:`824`)
        * Prevent DFS from creating Identity Features of globally ignored variables (:pr:`819`)
    * Changes
        * Remove python 2.7 support from serialize.py (:pr:`812`)
        * Make smart_open, boto3, and s3fs optional dependencies (:pr:`827`)
    * Documentation Changes
        * remove python 2.7 support and add 3.7 in install.rst (:pr:`805`)
        * Fix import error in docs (:pr:`803`)
        * Fix release title formatting in changelog (:pr:`806`)
    * Testing Changes
        * Use multiple CPUS to run tests on CI (:pr:`811`)
        * Refactor test entityset creation to avoid saving to disk (:pr:`813`, :pr:`821`)
        * Remove get_values() from test_es.py to remove warnings (:pr:`820`)

    Thanks to the following people for contributing to this release:
    :user:`frances-h`, :user:`jeff-hernandez`, :user:`rwedge`, :user:`systemshift`

Breaking Changes
++++++++++++++++

* The libraries used for downloading or uploading from S3 or URLs are now
  optional and will no longer be installed by default.  To use this
  functionality they will need to be installed separately.
* The fix to how the Diff primitive is calculated may slow down the overall
  calculation time of feature lists that use this primitive.

v0.12.0 Oct 31, 2019
====================
    * Enhancements
        * Added First primitive (:pr:`770`)
        * Added Entropy aggregation primitive (:pr:`779`)
        * Allow custom naming for multi-output primitives (:pr:`780`)
    * Fixes
        * Prevents user from removing base entity time index using additional_variables (:pr:`768`)
        * Fixes error when a multioutput primitive was supplied to dfs as a groupby trans primitive (:pr:`786`)
    * Changes
        * Drop Python 2 support (:pr:`759`)
        * Add unit parameter to AvgTimeBetween (:pr:`771`)
        * Require Pandas 0.24.1 or higher (:pr:`787`)
    * Documentation Changes
        * Update featuretools slack link (:pr:`765`)
        * Set up repo to use Read the Docs (:pr:`776`)
        * Add First primitive to API reference docs (:pr:`782`)
    * Testing Changes
        * CircleCI fixes (:pr:`774`)
        * Disable PIP progress bars (:pr:`775`)

    Thanks to the following people for contributing to this release:
    :user:`ablacke-ayx`, :user:`BoopBoopBeepBoop`, :user:`jeffzi`,
    :user:`kmax12`, :user:`rwedge`, :user:`thehomebrewnerd`, :user:`twdobson`

v0.11.0 Sep 30, 2019
====================
    .. warning::
        The next non-bugfix release of Featuretools will not support Python 2

    * Enhancements
        * Improve how files are copied and written (:pr:`721`)
        * Add number of rows to graph in entityset.plot (:pr:`727`)
        * Added support for pandas DateOffsets in DFS and Timedelta (:pr:`732`)
        * Enable feature-specific top_n value using a dictionary in encode_features (:pr:`735`)
        * Added progress_callback parameter to dfs() and calculate_feature_matrix() (:pr:`739`, :pr:`745`)
        * Enable specifying primitives on a per column or per entity basis (:pr:`748`)
    * Fixes
        * Fixed entity set deserialization (:pr:`720`)
        * Added error message when DateTimeIndex is a variable but not set as the time_index (:pr:`723`)
        * Fixed CumCount and other group-by transform primitives that take ID as input (:pr:`733`, :pr:`754`)
        * Fix progress bar undercounting (:pr:`743`)
        * Updated training_window error assertion to only check against observations (:pr:`728`)
        * Don't delete the whole destination folder while saving entityset (:pr:`717`)
    * Changes
        * Raise warning and not error on schema version mismatch (:pr:`718`)
        * Change feature calculation to return in order of instance ids provided (:pr:`676`)
        * Removed time remaining from displayed progress bar in dfs() and calculate_feature_matrix() (:pr:`739`)
        * Raise warning in normalize_entity() when time_index of base_entity has an invalid type (:pr:`749`)
        * Remove toolz as a direct dependency (:pr:`755`)
        * Allow boolean variable types to be used in the Multiply primitive (:pr:`756`)
    * Documentation Changes
        * Updated URL for Compose (:pr:`716`)
    * Testing Changes
        * Update dependencies (:pr:`738`, :pr:`741`, :pr:`747`)

    Thanks to the following people for contributing to this release:
    :user:`angela97lin`, :user:`chidauri`, :user:`christopherbunn`,
    :user:`frances-h`, :user:`jeff-hernandez`, :user:`kmax12`,
    :user:`MarcoGorelli`, :user:`rwedge`, :user:`thehomebrewnerd`

Breaking Changes
++++++++++++++++

* Feature calculations will return in the order of instance ids provided instead of the order of time points instances are calculated at.

v0.10.1 Aug 25, 2019
====================
    * Fixes
        * Fix serialized LatLong data being loaded as strings (:pr:`712`)
    * Documentation Changes
        * Fixed FAQ cell output (:pr:`710`)

    Thanks to the following people for contributing to this release:
    :user:`gsheni`, :user:`rwedge`


v0.10.0 Aug 19, 2019
====================
    .. warning::
        The next non-bugfix release of Featuretools will not support Python 2


    * Enhancements
        * Give more frequent progress bar updates and update chunk size behavior (:pr:`631`, :pr:`696`)
        * Added drop_first as param in encode_features (:pr:`647`)
        * Added support for stacking multi-output primitives (:pr:`679`)
        * Generate transform features of direct features (:pr:`623`)
        * Added serializing and deserializing from S3 and deserializing from URLs (:pr:`685`)
        * Added nlp_primitives as an add-on library (:pr:`704`)
        * Added AutoNormalize to Featuretools plugins (:pr:`699`)
        * Added functionality for relative units (month/year) in Timedelta (:pr:`692`)
        * Added categorical-encoding as an add-on library (:pr:`700`)
    * Fixes
        * Fix performance regression in DFS (:pr:`637`)
        * Fix deserialization of feature relationship path (:pr:`665`)
        * Set index after adding ancestor relationship variables (:pr:`668`)
        * Fix user-supplied variable_types modification in Entity init (:pr:`675`)
        * Don't calculate dependencies of unnecessary features (:pr:`667`)
        * Prevent normalize entity's new entity having same index as base entity (:pr:`681`)
        * Update variable type inference to better check for string values (:pr:`683`)
    * Changes
        * Moved dask, distributed imports (:pr:`634`)
    * Documentation Changes
        * Miscellaneous changes (:pr:`641`, :pr:`658`)
        * Modified doc_string of top_n in encoding (:pr:`648`)
        * Hyperlinked ComposeML (:pr:`653`)
        * Added FAQ (:pr:`620`, :pr:`677`)
        * Fixed FAQ question with multiple question marks (:pr:`673`)
    * Testing Changes
        * Add master, and release tests for premium primitives (:pr:`660`, :pr:`669`)
        * Miscellaneous changes (:pr:`672`, :pr:`674`)

    Thanks to the following people for contributing to this release:
    :user:`alexjwang`, :user:`allisonportis`, :user:`ayushpatidar`,
    :user:`CJStadler`, :user:`ctduffy`, :user:`gsheni`, :user:`jeff-hernandez`,
    :user:`jeremyliweishih`, :user:`kmax12`, :user:`rwedge`, :user:`zhxt95`,

v0.9.1 Jul 3, 2019
====================
    * Enhancements
        * Speedup groupby transform calculations (:pr:`609`)
        * Generate features along all paths when there are multiple paths between entities (:pr:`600`, :pr:`608`)
    * Fixes
        * Select columns of dataframe using a list (:pr:`615`)
        * Change type of features calculated on Index features to Categorical (:pr:`602`)
        * Filter dataframes through forward relationships (:pr:`625`)
        * Specify Dask version in requirements for python 2 (:pr:`627`)
        * Keep dataframe sorted by time during feature calculation (:pr:`626`)
        * Fix bug in encode_features that created duplicate columns of
          features with multiple outputs (:pr:`622`)
    * Changes
        * Remove unused variance_selection.py file (:pr:`613`)
        * Remove Timedelta data param (:pr:`619`)
        * Remove DaysSince primitive (:pr:`628`)
    * Documentation Changes
        * Add installation instructions for add-on libraries (:pr:`617`)
        * Clarification of Multi Output Feature Creation (:pr:`638`)
        * Miscellaneous changes (:pr:`632`, :pr:`639`)
    * Testing Changes
        * Miscellaneous changes (:pr:`595`, :pr:`612`)

    Thanks to the following people for contributing to this release:
    :user:`CJStadler`, :user:`kmax12`, :user:`rwedge`, :user:`gsheni`, :user:`kkleidal`, :user:`ctduffy`

v0.9.0 Jun 19, 2019
===================
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

v0.8.0 May 17, 2019
===================
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

Breaking Changes
++++++++++++++++

* ``NUnique`` has been renamed to ``NumUnique``.

    Previous behavior

    .. code-block:: python

        from featuretools.primitives import NUnique

    New behavior

    .. code-block:: python

        from featuretools.primitives import NumUnique

v0.7.1 Apr 24, 2019
===================
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

v0.7.0 Mar 29, 2019
===================
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

Breaking Changes
++++++++++++++++

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


v0.6.1 Feb 15, 2019
===================
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

v0.6.0 Jan 30, 2018
===================
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

v0.5.1 Dec 17, 2018
===================
    * Add missing dependencies (:pr:`353`)
    * Move comment to note in documentation (:pr:`352`)

v0.5.0 Dec 17, 2018
===================
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

v0.4.1 Nov 29, 2018
===================
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

v0.4.0 Oct 31, 2018
===================
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

v0.3.1 Sep 28, 2018
===================
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

v0.3.0 Aug 27, 2018
===================
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

v0.2.2 Aug 20, 2018
===================
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

v0.2.1 Jul 2, 2018
==================
    * Cpu count fix (:pr:`176`)
    * Update flight (:pr:`175`)
    * Move feature matrix calculation helper functions to separate file (:pr:`177`)

v0.2.0 Jun 22, 2018
===================
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


v0.1.21 May 30, 2018
====================
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

v0.1.20 Apr 13, 2018
====================
    * Primitives as strings in DFS parameters (:pr:`129`)
    * Integer time index bugfixes (:pr:`128`)
    * Add make_temporal_cutoffs utility function (:pr:`126`)
    * Show all entities, switch shape display to row/col (:pr:`124`)
    * Improved chunking when calculating feature matrices  (:pr:`121`)
    * fixed num characters nan fix (:pr:`118`)
    * modify ignore_variables docstring (:pr:`117`)

v0.1.19 Mar 21, 2018
====================
    * More descriptive DFS progress bar (:pr:`69`)
    * Convert text variable to string before NumWords (:pr:`106`)
    * EntitySet.concat() reindexes relationships (:pr:`96`)
    * Keep non-feature columns when encoding feature matrix (:pr:`111`)
    * Uses full entity update for dependencies of uses_full_entity features (:pr:`110`)
    * Update column names in retail demo (:pr:`104`)
    * Handle Transform features that need access to all values of entity (:pr:`91`)

v0.1.18 Feb 27, 2018
====================
    * fixes related instances bug (:pr:`97`)
    * Adding non-feature columns to calculated feature matrix (:pr:`78`)
    * Relax numpy version req (:pr:`82`)
    * Remove `entity_from_csv`, tests, and lint (:pr:`71`)

v0.1.17 Jan 18, 2018
====================
    * LatLong type (:pr:`57`)
    * Last time index fixes (:pr:`70`)
    * Make median agg primitives ignore nans by default (:pr:`61`)
    * Remove Python 3.4 support (:pr:`64`)
    * Change `normalize_entity` to update `secondary_time_index` (:pr:`59`)
    * Unpin requirements (:pr:`53`)
    * associative -> commutative (:pr:`56`)
    * Add Words and Chars primitives (:pr:`51`)

v0.1.16 Dec 19, 2017
====================
    * fix EntitySet.combine_variables and standardize encode_features (:pr:`47`)
    * Python 3 compatibility (:pr:`16`)

v0.1.15 Dec 18, 2017
====================
    * Fix variable type in demo data (:pr:`37`)
    * Custom primitive kwarg fix (:pr:`38`)
    * Changed order and text of arguments in make_trans_primitive docstring (:pr:`42`)

v0.1.14 Nov 20, 2017
====================
    * Last time index (:pr:`33`)
    * Update Scipy version to 1.0.0 (:pr:`31`)


v0.1.13 Nov 1, 2017
===================
    * Add MANIFEST.in (:pr:`26`)

v0.1.11 Oct 31, 2017
====================
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

v0.1.10 Oct 12, 2017
====================
    * NumTrue primitive added and docstring of other primitives updated (:pr:`11`)
    * fixed hash issue with same base features (:pr:`8`)
    * Head fix (:pr:`9`)
    * Fix training window (:pr:`10`)
    * Add associative attribute to primitives (:pr:`3`)
    * Add status badges, fix license in setup.py (:pr:`1`)
    * fixed head printout and flight demo index (:pr:`2`)

v0.1.9 Sep 8, 2017
==================
    * Documentation improvements
    * New ``featuretools.demo.load_mock_customer`` function

v0.1.8 Sep 1, 2017
==================
    * Bug fixes
    * Added ``Percentile`` transform primitive

v0.1.7 Aug 17, 2017
===================
    * Performance improvements for approximate in ``calculate_feature_matrix`` and ``dfs``
    * Added ``Week`` transform primitive

v0.1.6 Jul 26, 2017
===================
    * Added ``load_features`` and ``save_features`` to persist and reload features
    * Added save_progress argument to ``calculate_feature_matrix``
    * Added approximate parameter to ``calculate_feature_matrix`` and ``dfs``
    * Added ``load_flight`` to ft.demo

v0.1.5 Jul 11, 2017
===================
    * Windows support

v0.1.3 Jul 10, 2017
===================
    * Renamed feature submodule to primitives
    * Renamed prediction_entity arguments to target_entity
    * Added training_window parameter to ``calculate_feature_matrix``

v0.1.2 Jul 3rd, 2017
====================
    * Initial release

.. command
.. git log --pretty=oneline --abbrev-commit
