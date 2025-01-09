import os
from datetime import datetime
from typing import List, Union

import click
import geopandas as gpd
import pandas as pd
from pytz import timezone
from timezonefinder import TimezoneFinder

from fwi_predict.pipeline import create_standard_dataset


def prep_daily_sample(pond_metadata: gpd.GeoDataFrame,
					  					target_date: Union[int, str] = 'tomorrow',
					  					times_of_day: List[str] = ['09:00:00', '16:00:00']) -> gpd.GeoDataFrame:
	"""Get dataframe of samples to predict for a given day and times of day."""

	keep_cols = ['pond_id', 'farmer', 'village', 'geometry']
	ponds = pond_metadata[keep_cols].copy()

	# Add timezone information
	ponds['timezones'] = ponds['geometry'].apply(
		lambda g: TimezoneFinder().timezone_at(lng=g.x, lat=g.y)
	)
	
	predict_samples = ( # Check that this code is correct.
		ponds.loc[ponds.index.repeat(len(times_of_day))]
		.assign(
			time_of_day=list(times_of_day) * len(ponds)
		)
	)

	# Convert to datetime in the respective timezone
	def localize_time(row):
		tz = timezone(row['timezones'])
		return tz.localize(datetime.combine(target_date, pd.to_datetime(row['time_of_day']).time()))

	predict_samples['sample_dt'] = predict_samples.apply(localize_time, axis=1)
	predict_samples.reset_index(drop=True, inplace=True)

	return predict_samples


def run_daily_inference(pond_metadata: gpd.GeoDataFrame,
												models_filepath: str, # Maybe will actually be dataset, which then indexes models.
												target_date: Union[int, str] = 'tomorrow',
												times_of_day: List[str] = ['09:00:00', '16:00:00'],
												download_dir: str = 'data/gcs',
												bucket: str = 'fwi-predict',
												project: str = 'fwi-water-quality-sensing') -> gpd.GeoDataFrame:
	"""Run daily inference for a given day and times of day."""
	# Get prediction samples
	if target_date is 'tomorrow':
		target_date = datetime.today() + datetime.timedelta(days=1)
		target_date = target_date.strftime('%Y-%m-%d')

	predict_samples = prep_daily_sample(pond_metadata, target_date, times_of_day)
	filename = os.path.splitext(os.path.basename(models_filepath))[0] + '_' + target_date + '.csv'
	description = f'daily_inference_{target_date}'

	# Creat pre
	predict_df = create_standard_dataset(predict_samples,
																			 filename,
																			 download_dir,
																			 description,
																			 bucket=bucket,
																			 gee_project=project)
	
	# After this you should save the final data, load the model, and make predictions.
	# Then write the streamlit app to display the results.

