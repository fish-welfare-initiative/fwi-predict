"""Utilities for Google Earth Engine."""

import datetime
import pandas as pd
from typing import Any, Dict, Literal, Union

import ee

SENTINEL2_SCL_MAP = {
 	1: "Saturated/defective",
	2: "Dark area pixels",
	3: "Cloud shadows",
	4: "Vegetation",
	5: "Bare soils",
	6: "Water",
	7: "Low probability clouds / unclassified",
	8: "Medium probability clouds",
	9: "High probability clouds",
	10: "Cirrus",
	11: "Snow/ice"
}

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


def scale_sentinel2_l2a(image: ee.Image) -> ee.Image :
	"""Scales Sentinel-2 L2A bands to reflectance values."""
	scale_factors = {
		'B1': 0.0001, 'B2': 0.0001, 'B3': 0.0001, 'B4': 0.0001, 'B5': 0.0001,
		'B6': 0.0001, 'B7': 0.0001, 'B8': 0.0001, 'B8A': 0.0001, 'B9': 0.0001,
		'B11': 0.0001, 'B12': 0.0001, 'AOT': 0.001, 'WVP': 0.001, 'MSK_CLDPRB': 0.01, 'MSK_SNWPRB': 0.01
	}

	def scale_band(band_name: ee.String):
		return image.select(band_name).multiply(scale_factors[band_name]).rename(band_name)
	
	scaled_bands = ee.Image([scale_band(band) for band in scale_factors.keys()])

	return image.addBands(
		scaled_bands, overwrite=True
	)


def get_time_distance(
	image: ee.Image,
	reference_date: Union[datetime.datetime, pd.Timestamp, ee.Date],
	property_name: str = "hours_from_date",
	unit: Literal['second', 'minute', 'hour', 'day', 'month', 'year'] = 'hour',
) -> ee.Image:
    """Calculate time distance between EE image date property and a reference date.
    
    Args:
			image: an EE image.
			reference date: the reference date to calculate distance to.
			date_property: the image date property to calculate distnance from.
			property_name: the name of the calculated time distance property.
			units: the units to calculate time distance in.
			absolute: whether to return absolute value.
        
    Returns:
			the EE image with the time distance set.
    """
    ref_date = reference_date if isinstance(reference_date, ee.Date) else ee.Date(reference_date)
    
    return image.set(
			property_name,
			image.date().difference(ref_date, unit=unit)
    )


def get_nearest_sentinel2_image(feature: ee.Feature, forward_days: int = 0, back_days: int = 10):
	"""Add docstring.
     
  Can generalize to be non sentinel later. 
	"""
	# Set the date range (e.g., 30 days before and after the sample date)
	date = ee.Date(feature.get('sample_dt'))
	start_date = date.advance(-back_days, 'day') # Can later add datetime feature name argument
	end_date = date.advance(forward_days, 'day')
	
	# Get Sentinel-2 image collection
	collection = get_sentinel2_l2a() \
		.filterBounds(feature.geometry()) \
		.filterDate(start_date, end_date) \
		.map(lambda img: get_time_distance(img, date, property_name='hours_from_date')) \
		.map(lambda img: img.set({'hours_from_date': ee.Number(img.get('hours_from_date')).abs()})) \
		.sort('hours_from_date')
			
	# Return the nearest image if available.
	return ee.Image(collection.first())