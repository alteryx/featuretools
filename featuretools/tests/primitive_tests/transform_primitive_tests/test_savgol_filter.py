from math import floor

import numpy as np
import pandas as pd
from pytest import raises

from featuretools.primitives import SavgolFilter
from featuretools.tests.primitive_tests.utils import (
    PrimitiveTestBase,
    find_applicable_primitives,
    valid_dfs,
)


class TestSavgolFilter(PrimitiveTestBase):
    primitive = SavgolFilter
    data = pd.Series(
        [
            0,
            1,
            1,
            2,
            3,
            4,
            5,
            7,
            8,
            7,
            9,
            9,
            12,
            11,
            12,
            14,
            15,
            17,
            17,
            17,
            20,
            21,
            20,
            20,
            22,
            21,
            25,
            25,
            26,
            29,
            30,
            30,
            28,
            26,
            34,
            35,
            33,
            31,
            38,
            34,
            39,
            37,
            42,
            35,
            36,
            44,
            46,
            43,
            39,
            39,
            44,
            49,
            45,
            44,
            44,
            52,
            50,
            47,
            58,
            59,
            60,
            55,
            57,
            63,
            61,
            65,
            66,
            57,
            65,
            61,
            60,
            71,
            64,
            62,
            70,
            65,
            67,
            77,
            68,
            75,
            72,
            69,
            82,
            66,
            84,
            80,
            76,
            87,
            77,
            73,
            90,
            91,
            92,
            93,
            78,
            76,
            82,
            96,
            91,
            94,
        ],
    )
    expected_output = pd.Series(
        [
            -0.24600037643516087,
            0.6354225484660259,
            1.518717742974036,
            2.405318302343475,
            3.296657321828948,
            4.1941678966850615,
            5.099283122166421,
            6.0134360935276305,
            6.938059906023296,
            7.874587654908025,
            8.824452435436303,
            9.786858450473883,
            10.923177508989724,
            12.025171624713803,
            13.009153318077633,
            14.08041843739766,
            14.900621118012227,
            15.796338672768673,
            16.77084014383764,
            17.662961752206375,
            18.472703497874882,
            19.451454723765682,
            20.530565544295253,
            21.849950964367157,
            22.478260869564927,
            23.15233736515171,
            24.12356979405003,
            25.23962079110788,
            26.000980712650854,
            27.082379862699877,
            27.787839163124843,
            28.879045439685797,
            29.762994442627924,
            31.067342268714864,
            32.11147433801854,
            32.666557698593884,
            33.06864988558309,
            34.00098071265075,
            35.134030728995945,
            36.15135665250035,
            36.945733899966825,
            37.56227525335028,
            38.55769859431137,
            39.3975155279498,
            39.87054593004198,
            40.304347826086435,
            41.11670480549146,
            42.00948022229432,
            41.982674076495044,
            42.62798300098016,
            43.15887544949274,
            44.53481529911678,
            45.680614579927486,
            46.93886891140834,
            47.98300098071202,
            48.80549199084604,
            50.28244524354299,
            52.66851912389601,
            54.28604118993064,
            55.81529911735788,
            57.10297482837455,
            57.82641386073805,
            59.45276234063342,
            60.77280156913945,
            61.23667865315383,
            61.81660673422607,
            62.60281137626594,
            62.54004576658957,
            62.78653154625613,
            63.23046747302958,
            64.09087937234307,
            65.25661981039471,
            65.19385420071833,
            66.34161490683144,
            66.65021248774022,
            67.38280483818154,
            68.8126838836212,
            69.79470415168265,
            70.943772474664,
            72.74076495586698,
            73.04020921869797,
            73.3586139261187,
            74.67734553775647,
            75.71559333115299,
            77.51814318404607,
            79.62471395880902,
            80.60150375939745,
            80.61163779012645,
            81.89342922523593,
            82.41124550506593,
            83.19293292519846,
            83.97174920172642,
            84.7620599588564,
            85.57823082079385,
            86.4346274117442,
            87.34561535591293,
            88.32556027750543,
            89.38882780072717,
            90.54978354978357,
            91.82279314888011,
        ],
    )

    def test_error(self):
        window_length = 1
        polyorder = 3
        mode = "incorrect"
        error_text = "polyorder must be less than window_length."
        with raises(ValueError, match=error_text):
            self.primitive(window_length, polyorder)

        error_text = (
            "Both window_length and polyorder must be defined if you define one."
        )

        with raises(ValueError, match=error_text):
            self.primitive(window_length=window_length)
        with raises(ValueError, match=error_text):
            self.primitive(polyorder=polyorder)
        error_text = "mode must be 'mirror', 'constant', 'nearest', 'wrap' or 'interp'."
        with raises(ValueError, match=error_text):
            self.primitive(
                window_length=window_length,
                polyorder=polyorder,
                mode=mode,
            )

    def test_less_window_size(self):
        primitive_func = self.primitive().get_function()
        for i in range(20):
            data = pd.Series(list(range(i)), dtype="float64")
            assert data.equals(primitive_func(data))

    def test_regular(self):
        window_length = floor(len(self.data) / 10) * 2 + 1
        polyorder = 3
        primitive_func = self.primitive(window_length, polyorder).get_function()
        output = list(primitive_func(self.data))
        for a, b in zip(self.expected_output, output):
            assert np.isclose(a, b)

    def test_nans(self):
        primitive_func = self.primitive().get_function()
        data_nans = self.data.copy()
        data_nans = pd.concat([data_nans, pd.Series([np.nan] * 5, dtype="float64")])
        # more than 5 nans due to window
        assert sum(np.isnan(primitive_func(data_nans))) == 15

    def test_with_featuretools(self, es):
        transform, aggregation = find_applicable_primitives(self.primitive)
        primitive_instantiate = self.primitive()
        transform.append(primitive_instantiate)
        valid_dfs(es, aggregation, transform, self.primitive)
