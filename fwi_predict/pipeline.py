import ee
import os
from typing import List

import geopandas as gpd
import pandas as pd

from .constants import FORECAST_TIMES
from .geo.ee import export_forecasts_for_samples, monitor_task
from .gcs import download_files


def clean_gfs(raw_gfs: pd.DataFrame) -> pd.DataFrame:
	"""Cleans GFS data download."""
	# Really ought to get correct time zones for forecasts again.
	
	gfs = raw_gfs.copy()
	gfs = gfs.drop(columns=['system:index', '.geo'])

	# Check data correctness
	observations_per_measurement = gfs.groupby('measurement_idx').size()
	assert (observations_per_measurement.eq(observations_per_measurement.iloc[0]).all()), (
		"Number of observations per measurement varies."
	)

	# Format date columns. Right now we drop many of these but may keep later.
	gfs['forecast_creation_dt'] = pd.to_datetime(gfs['forecast_creation_dt'].astype(str), format='%Y%m%d%H')
	fixed_time_forecast = gfs['forecast_time'].str.isnumeric()
	gfs.loc[fixed_time_forecast, 'forecast_dt'] = (
		gfs.loc[fixed_time_forecast, 'forecast_creation_dt'] + 
		pd.Series(pd.to_timedelta(gfs.loc[fixed_time_forecast, 'forecast_time'], unit='hour'), index=gfs[fixed_time_forecast].index)
	)
	gfs.loc[~fixed_time_forecast, 'forecast_dt'] = pd.NaT
	gfs['forecast_date'] = gfs['forecast_dt'].dt.date
	gfs['forecast_hour_of_day'] = gfs['forecast_dt'].dt.hour

	# Reorder columns and rows
	front_cols = ['measurement_idx', 'forecast_time', 'forecast_dt', 'forecast_date', 'forecast_hour_of_day', 'forecast_creation_dt'] 
	gfs = gfs[front_cols + [col for col in gfs.columns if col not in front_cols]]
	gfs = gfs.sort_values(['measurement_idx', 'forecast_dt'])

	# Pivot wide so one observation per measurement
	gfs_wide = gfs \
		.drop(columns=['forecast_creation_dt', 'forecast_hour_of_day', 'forecast_date', 'forecast_dt']) \
		.pivot(index='measurement_idx', columns='forecast_time')
	gfs_wide.columns = gfs_wide.columns.map('{0[0]}_{0[1]}'.format)

	return gfs_wide


def merge_samples_and_gfs(samples: pd.DataFrame,
					  			 				gfs_data: pd.DataFrame) -> pd.DataFrame:
	"""Create Dataframe for prediction from samples and GFS data."""
	predict_df = samples.set_index('measurement_idx').join(gfs_data)

	# Add time categoricals
	predict_df['month'] = predict_df['sample_dt'].dt.month
	predict_df['week_of_month'] = predict_df['sample_dt'].dt.isocalendar().week
	predict_df['day_of_week'] = predict_df['sample_dt'].dt.dayofweek

	# Line below will likely have to chagne
	predict_df['morning'] = predict_df['sample_dt'].dt.hour < 12

	return predict_df


def create_standard_dataset(samples: gpd.GeoDataFrame,
														filepath: str,
														download_dir: str,
														description: str,
														gcs_bucket: str = 'fwi-predict',
														gee_project: str = 'fwi-water-quality-sensing') -> pd.DataFrame:
	"""Create standard modeling dataset for a set of samples."""
	ee.Authenticate()
	ee.Initialize(project=gee_project)
	
	# Creat export and wait until it resolves.
	task = export_forecasts_for_samples(samples,
									 										FORECAST_TIMES,
																			filepath,
																			description=description,
																			bucket=gcs_bucket,
																			project=gee_project)
	task_success = monitor_task(task)

	if not task_success:
		print("Data export failed. Please consult GEE task manager for information.")
		return None
	
	# Download exported data from GCS
	## Consider using gsutil command line instead as it can multithread downloads
	download_files(bucket=gcs_bucket,
								 file_glob=filepath,
								 download_dir=download_dir,
								 project='fwi-predict')

	# Clean GFS data
	gfs = pd.read_csv(os.path.join(download_dir, filepath))
	gfs_clean = clean_gfs(gfs)

	# Create prediction dataframe
	predict_df = merge_samples_and_gfs(samples, gfs_clean)

	return predict_df