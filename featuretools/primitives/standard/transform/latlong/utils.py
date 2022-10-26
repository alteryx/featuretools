import numpy as np


def _haversine_calculate(lat_1s, lon_1s, lat_2s, lon_2s, unit):
    # https://stackoverflow.com/a/29546836/2512385
    lon1, lat1, lon2, lat2 = map(np.radians, [lon_1s, lat_1s, lon_2s, lat_2s])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    radius_earth = 3958.7613
    if unit == "kilometers":
        radius_earth = 6371.0088
    distances = radius_earth * 2 * np.arcsin(np.sqrt(a))
    return distances
