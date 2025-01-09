"""Utilities for Google Earth Engine."""
import time
from pathlib import Path
from typing import List, Literal, Union

import datetime
import ee
import geopandas as gpd
import pandas as pd
from geemap import gdf_to_ee


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



def monitor_task(task: ee.batch.Task, check_interval: int = 60) -> bool:
	"""Check for Earth Engine export completion.
	
	Args:
		task: an Earth Engine export task.
		check_interval: check status interval (in seconds).

	Returns:
		True if task completed successfully, False otherwise.
	"""
	while True:
		status = task.status()
		state = status['state']
		if state == 'COMPLETED':
			print('Data export completed successfully.')
			return True
		elif state == 'FAILED':
			print('Data export failed.')
			return False
		elif state in ['CANCELLED', 'CANCEL_REQUESTED']:
			print('Data export was cancelled.')
			return False
		else:
			time.sleep(check_interval)


def get_sample_gfs_forecast(sample: ee.Feature,
							              forecast_times: List,
							              gfs: ee.ImageCollection = None) -> ee.FeatureCollection:
	"""Add docstring."""
	if gfs is None:
		gfs = get_gfs()


	# Get times for which we want forecasts.
	sample_idx = sample.get('sample_idx') # Get sample index
	sample_dt = ee.Date(sample.get('sample_dt'))
	day_prior = sample_dt \
    .advance(5.5, 'hour') \
    .advance(-1, 'day') \
    .update(hour=0, minute=0, second=0) #Have to adjust for Asia/Kolkata timezone before finding previous day.
  
	forecast_time_list = ee.List(forecast_times).map(
    lambda hours: day_prior.advance(hours, 'hour').advance(-6, 'hour').millis() # Again adjusting for timezone so forecasts don't overlap with sample time
  )

  # Pre-filter GFS to reduce computation
	forecast_subset = gfs.filterDate( 
    ee.Date(forecast_time_list.sort().getNumber(0)).advance(-1, 'day'), # Earliest forecast initialization time we are interested in 
    sample_dt.advance(-1, 'day') # Want forecasts initialized one day before sample was taken.
  )

  # Get latest forecast for each forecast (that is at least one day older than sample time)
	def get_latest_forecast_for_time(forecast_time: ee.Number) -> ee.Image:
		"""Get most recent forecast for a given forecast time."""
    # Get frecast for specific time of interest
		subset = forecast_subset \
      .filter(ee.Filter.lt('creation_time', forecast_time)) \
      .filter(ee.Filter.eq('forecast_time', forecast_time)) # Less than in first so we get all datapoints.
   
    # Then get most recent forecast
		latest_init_time = subset.aggregate_array('creation_time').sort().get(-1)
	
		return subset.filter(ee.Filter.eq('creation_time', latest_init_time)).first()
  
  
  # Extract forecast values
	forecasts_for_times = ee.ImageCollection(
    forecast_time_list.map(get_latest_forecast_for_time)
  )

  # Assign metadata to forecast values and cumulative values
	forecast_values = forecasts_for_times \
    .map(lambda img: img.sample(sample.geometry())) \
    .flatten() \
    .map(lambda f: f # Set metadata
      .set('forecast_creation_dt', f.id().slice(0, 10)) # Same as below
      .set('forecast_hour', f.id().slice(11, 14)) # Would be good to make this less hacky
      .set('sample_idx', sample_idx)
    )
	
	# Map each element of forecast_time_list to each feature of forecast_values
	forecast_values_list = forecast_values.toList(forecast_values.size())
	forecast_values = ee.FeatureCollection(
		forecast_values.map(lambda f: f.set('forecast_time', 
			ee.List(forecast_times).get(forecast_values_list.indexOf(f))))
	)

    # Get forecast at time of sample
	sample_dt_rounded = sample_dt \
    .millis() \
    .divide(1000 * 60 * 60) \
    .round() \
    .multiply(1000 * 60 * 60) # Round sample time to nearest hour
	sample_time_forecast = ee.Image(get_latest_forecast_for_time(sample_dt_rounded))

	id = sample_time_forecast.getString('system:id').split("/").getString(2)
	sample_time_forecast = sample_time_forecast \
    .sample(sample.geometry()) \
    .first()
  
	sample_time_forecast = sample_time_forecast \
    .set('forecast_creation_dt', id.slice(0, 10)) \
		.set('forecast_hour', id.slice(11, 14)) \
    .set('forecast_time', 'sample') \
    .set('sample_idx', sample_idx)

  # Get cumulative values for the week prior to the sample time
  # 9 AM UTC is 3:30 PM IST
	def get_cumulative_history(lookback_days: ee.Number) -> ee.FeatureCollection:
		"""Get cumulative history for a given number of days."""
		cum_days = ee.List.sequence(0, ee.Number(lookback_days).multiply(-1), step=-1)
		gfs_subset = gfs.filterDate(
      day_prior.advance(ee.Number(cum_days.sort().get(0)).subtract(1), 'day'),
      sample_dt
    )

    # Ought to check that you are summing correct number of days for each
		global_history = ee.ImageCollection(
      cum_days
      .map(lambda day: day_prior.advance(day, 'day').update(hour=9).millis())
      .map(lambda f_time: gfs_subset.filter(ee.Filter.eq('forecast_time', f_time)).sort('creation_time', False).first())
    )
		global_aggregate = global_history.reduce(ee.Reducer.sum())
    
		cum_values = ee.Image(global_aggregate)
		cum_values = cum_values \
      .rename(cum_values.bandNames().map(lambda name: ee.String(name).slice(0, -4))) \
      .sample(sample.geometry()) \
      .first() # Remove sum from end of band names
		
		return cum_values
  
	three_day_history = get_cumulative_history(3)
	week_history = get_cumulative_history(7)

	three_day_history = three_day_history \
    .set('sample_idx', sample_idx) \
    .set('forecast_time', 'three_day_cum')
  
	week_history = week_history \
    .set('sample_idx', sample_idx) \
    .set('forecast_time', 'seven_day_cum')

  # Merge and return
	forecast_values = forecast_values.merge(ee.FeatureCollection([sample_time_forecast, three_day_history, week_history]))
  
	return forecast_values


def export_forecasts_for_samples(samples: gpd.GeoDataFrame,
																 forecast_times: List[int],
																 filepath: Union[str, Path],
																 description: str = None,
																 bucket: str = 'fwi-predict',
																 project: str = 'fwi-water-quality-sensing') -> ee.batch.Task:
	"""Export GFS forecasts for samples."""
	ee.Authenticate()
	ee.Initialize(project=project)

	# Export GFS data
	samples_ee = gdf_to_ee(samples)
	samples_ee = samples_ee.map(lambda f: f.set('sample_dt', ee.Date(f.get('sample_dt'))))

	# Get GFS forecast
	forecast_coll = samples_ee \
		.map(lambda f: get_sample_gfs_forecast(f, forecast_times)) \
		.flatten()
	
	# Remove file extension if present.
	fp = Path(filepath)
	fp = fp.parent / fp.stem
	fp = fp.as_posix()

	# Export GFS forecast
	task = ee.batch.Export.table.toCloudStorage(
		collection=forecast_coll,
		description=description,
		bucket=bucket,
		fileNamePrefix=fp,
		fileFormat='CSV'
	)
	task.start()
	print(f"Exporting GFS forecast data to {bucket}/{fp}.\nVisit https://code.earthengine.google.com/tasks to monitor the export.")
	
	return task
