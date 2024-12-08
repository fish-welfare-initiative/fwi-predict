import datetime
from typing import Literal
from typing import Union

import ee
from pandas import Timestamp

def get_sentinel2_l2a() -> ee.ImageCollection: 
    """Returns Sentinel-2 image collection."""
    return ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")


def get_era5_land_hourly() -> ee.ImageCollection:
    """Returns ERA5-LAND hourly climate reanalysis image collection."""
    return ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY")


def get_era5_land_daily() -> ee.ImageCollection:
    """Returns ERA5-LAND historical daily climate aggregates."""
    return ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")


def get_gfs() -> ee.ImageCollection:
    """Returns Global Forecast System 384-hour forecast data."""
    return ee.ImageCollection("NOAA/GFS0P25")