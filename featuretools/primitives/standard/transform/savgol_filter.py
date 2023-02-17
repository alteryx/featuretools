from math import floor

import numpy as np
from scipy.signal import savgol_coeffs, savgol_filter
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Double

from featuretools.primitives.base import TransformPrimitive


class SavgolFilter(TransformPrimitive):
    """Applies a Savitzky-Golay filter to a list of values.

    Description:
        Given a list of values, return a smoothed list which increases
        the signal to noise ratio without greatly distoring the
        signal. Uses the `Savitzky–Golay filter` method.

        If the input list has less than 20 values, it will be returned
        as is.

        See the following page for more info:
        https://docs.scipy.org/doc/scipy-0.16.0/reference/generated/scipy.signal.savgol_filter.html

    Args:
        window_length (int):  The length of the filter window (i.e. the number
            of coefficients). `window_length` must be a positive odd integer.

        polyorder (int): The order of the polynomial used to fit the samples.
            `polyorder` must be less than `window_length`.

        deriv (int): Optional. The order of the derivative to compute.  This
            must be a nonnegative integer.  The default is 0, which means to
            filter the data without differentiating.

        delta (float): Optional. The spacing of the samples to which the filter
            will be applied. This is only used if deriv > 0.  Default is 1.0.

        mode (str): Optional. Must be 'mirror', 'constant', 'nearest', 'wrap'
            or 'interp'.  This determines the type of extension to use for the
            padded signal to which the filter is applied.  When `mode` is
            'constant', the padding value is given by `cval`.  See the Notes
            for more details on 'mirror', 'constant', 'wrap', and 'nearest'.

            When the 'interp' mode is selected (the default), no extension
            is used.  Instead, a degree `polyorder` polynomial is fit to the
            last `window_length` values of the edges, and this polynomial is
            used to evaluate the last `window_length // 2` output values.

        cval (scalar): Optional. Value to fill past the edges of the input
            if `mode` is 'constant'. Default is 0.0.

    Examples:
        >>> savgol_filter = SavgolFilter()
        >>> data = [0, 1, 1, 2, 3, 4, 5, 7, 8, 7, 9, 9, 12, 11, 12, 14, 15, 17, 17, 17, 20]
        >>> [round(x, 4) for x in savgol_filter(data).tolist()[:3]]
        [0.0429, 0.8286, 1.2571]

        We can control `window_length` and `polyorder` of the filter.

        >>> savgol_filter = SavgolFilter(window_length=13, polyorder=3)
        >>> [round(x, 4) for x in savgol_filter(data).tolist()[:3]]
        [-0.0962, 0.6484, 1.4451]

        We can also control the `deriv` and `delta` parameters.

        >>> savgol_filter = SavgolFilter(deriv=1, delta=1.5)
        >>> [round(x, 4) for x in savgol_filter(data).tolist()[:3]]
        [0.754, 0.3492, 0.2778]

        Finally, we can use `mode` to control how edge values are handled.

        >>> savgol_filter = SavgolFilter(mode='constant', cval=5)
        >>> [round(x, 4) for x in savgol_filter(data).tolist()[:3]]
        [1.5429, 0.2286, 1.2571]
    """

    name = "savgol_filter"
    input_types = [ColumnSchema(semantic_tags={"numeric"})]
    return_type = ColumnSchema(logical_type=Double, semantic_tags={"numeric"})

    def __init__(
        self,
        window_length=None,
        polyorder=None,
        deriv=0,
        delta=1.0,
        mode="interp",
        cval=0.0,
    ):
        if window_length is not None and polyorder is not None:
            try:
                if mode not in ["mirror", "constant", "nearest", "interp", "wrap"]:
                    raise ValueError(
                        "mode must be 'mirror', 'constant', "
                        "'nearest', 'wrap' or 'interp'.",
                    )
                savgol_coeffs(window_length, polyorder, deriv=deriv, delta=delta)
            except Exception:
                raise
        elif (window_length is None and polyorder is not None) or (
            window_length is not None and polyorder is None
        ):
            error_text = (
                "Both window_length and polyorder must be defined if you define one."
            )
            raise ValueError(error_text)

        self.window_length = window_length
        self.polyorder = polyorder
        self.deriv = deriv
        self.delta = delta
        self.mode = mode
        self.cval = cval

    def get_function(self):
        def smooth(x):
            if x.shape[0] < 20:
                return x
            if np.isnan(np.min(x)):
                # interpolate the nan values, works for edges & middle nans
                mask = np.isnan(x)
                x[mask] = np.interp(
                    np.flatnonzero(mask),
                    np.flatnonzero(~mask),
                    x[~mask],
                )
            window_length = self.window_length
            polyorder = self.polyorder
            if window_length is None and polyorder is None:
                window_length = floor(len(x) / 10) * 2 + 1
                polyorder = 3
            return savgol_filter(
                x,
                window_length=window_length,
                polyorder=polyorder,
                deriv=self.deriv,
                delta=self.delta,
                mode=self.mode,
                cval=self.cval,
            )

        return smooth
