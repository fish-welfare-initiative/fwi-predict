import os
import pickle
from datetime import datetime, timedelta
from pathlib import Path
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

	keep_cols = ['pond_id', 'farmer', 'village', 'geometry', 'pond_depth_meters']
	ponds = pond_metadata[keep_cols].copy()
	ponds['winkler'] = True

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
		date = datetime.strptime(target_date, '%Y-%m-%d')
		return tz.localize(datetime.combine(date, pd.to_datetime(row['time_of_day']).time()))

	predict_samples['sample_dt'] = predict_samples.apply(localize_time, axis=1)
	predict_samples.reset_index(drop=True, inplace=True)
	predict_samples['sample_idx'] = pd.Series(range(len(predict_samples)))
	predict_samples = gpd.GeoDataFrame(predict_samples)

	print(predict_samples.head())

	return predict_samples


def run_daily_inference(pond_metadata: gpd.GeoDataFrame,
												target_date: Union[int, str] = 'tomorrow',
												times_of_day: List[str] = ['09:00:00', '16:00:00'],
												download_dir: str = 'data/gcs',
												bucket: str = 'fwi-predict',
												project: str = 'fwi-water-quality-sensing') -> gpd.GeoDataFrame:
	"""Run daily inference for a given day and times of day."""
	# Get prediction samples
	if target_date == 'tomorrow':
		target_date = datetime.today() + timedelta(days=1)
		target_date = target_date.strftime('%Y-%m-%d')

	predict_df_path = Path(f"./data/predict_dfs/daily/{target_date}.csv")
	predict_samples = prep_daily_sample(pond_metadata, target_date, times_of_day)


	if predict_df_path.exists(): 
		predict_df = pd.read_csv(predict_df_path, parse_dates=['sample_dt'], index_col=0)
	else:
		gcs_fp = f"daily_inference/{target_date}.csv"
		description = f'daily_inference_{target_date}'
		predict_df = create_standard_dataset(predict_samples,
																				 gcs_fp,
																				 download_dir,
																				 description,
																				 gcs_bucket=bucket,
																				 gee_project=project)
		# Save predict df
		predict_df.to_csv(predict_df_path)

	num_sum_cols = predict_df.columns[predict_df.columns.str.contains('num_sum')].tolist()
	drop_cols = ['sample_idx', 'pond_id', 'geometry'] + num_sum_cols
	predict_df = predict_df.drop(columns=drop_cols)

	# Get time parameters
	predict_df['morning'] = predict_df['hour'] < 12
	predict_df['half_hour'] = (predict_df['sample_dt'].dt.hour * 2 + (predict_df['sample_dt'].dt.minute >= 30).astype(int))

	# Load prediction model
	model_root = Path("./models/jun_21_dec_24_w_metadata").resolve()
	model_name = 'XGBoost'
	target = 'do_in_range'

	target_root = model_root / target

	with open(target_root / "encoder.pkl", 'rb') as f:
			encoder = pickle.load(f)

	with open(target_root / f"{model_name}.pkl", 'rb') as f:
			model = pickle.load(f)

	X = predict_df[model.feature_names_in_]
	probs = model.predict_proba(X)
	preds = encoder.inverse_transform(model.predict(X))

	# Add prediction probabilities and results to dataframe
	predict_samples['prediction'] = preds
	predict_samples['prob'] = [probs[i, pred] for i, pred in enumerate(encoder.transform(preds))]
	
	predict_samples.to_csv(f"./output/daily/{target_date}")


if __name__ == "__main__":

	ponds = gpd.read_file("./data/clean/pond_metadata_clean.geojson")
	ponds = ponds[ponds['geometry'].is_valid].head(50)

	run_daily_inference(ponds)