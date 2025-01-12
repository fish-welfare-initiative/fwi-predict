import ee
from pathlib import Path
from typing import Union

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
	observations_per_measurement = gfs.groupby('sample_idx').size()
	assert (observations_per_measurement.eq(observations_per_measurement.iloc[0]).all()), (
		"Number of observations per measurement varies."
	)

	# Reorder columns and rows
	front_cols = ['sample_idx', 'forecast_time', 'forecast_creation_dt', 'forecast_hour'] 
	gfs = gfs[front_cols + [col for col in gfs.columns if col not in front_cols]]
	gfs = gfs.sort_values(['sample_idx', 'forecast_time'])

	# Pivot wide to one observation per measurement
	value_cols = gfs.columns[~gfs.columns.isin(front_cols)].tolist()
	gfs_wide = gfs.pivot(index='sample_idx', columns='forecast_time', values=value_cols)
	gfs_wide.columns = gfs_wide.columns.map('{0[0]}_{0[1]}'.format)

	return gfs_wide


def create_standard_dataset(samples: gpd.GeoDataFrame,
														gfs_gcs_filepath: Union[str, Path],
														gfs_download_dir: str,
														description: str,
														gcs_bucket: str = 'fwi-predict',
														gee_project: str = 'fwi-water-quality-sensing') -> pd.DataFrame:
	"""Create standard modeling dataset for a set of samples."""
	ee.Authenticate()
	ee.Initialize(project=gee_project)
	
	# Creat export and wait until it resolves.
	task = export_forecasts_for_samples(samples,
									 										FORECAST_TIMES,
																			gfs_gcs_filepath,
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
								 file_glob=gfs_gcs_filepath,
								 download_dir=gfs_download_dir,
								 project=gee_project)

	# Clean GFS data
	gfs_path = Path(gfs_download_dir) / gfs_gcs_filepath
	gfs = pd.read_csv(gfs_path)
	gfs_clean = clean_gfs(gfs)

	# Create prediction dataframe
	predict_df = samples.set_index('sample_idx').join(gfs_clean)

	# Add time categoricals
	predict_df['morning'] = predict_df['sample_dt'].dt.hour < 12
	predict_df['month'] = predict_df['sample_dt'].dt.month
	predict_df['week_of_month'] = predict_df['sample_dt'].dt.isocalendar().week
	predict_df['day_of_week'] = predict_df['sample_dt'].dt.dayofweek

	return predict_df