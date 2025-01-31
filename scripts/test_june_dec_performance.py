# Test performance of models on measurements from June to December 2024.
import pickle
from pathlib import Path

import click
import geopandas as gpd
import pandas as pd

from fwi_predict.constants import TIMEZONE
from fwi_predict.pipeline import create_standard_dataset

@click.command()
@click.option('--re-export', is_flag=True, help='Re-export GFS data even if it already exists.')
def main(re_export):
  """Test performance of models on measurements from June to December 2024."""
  # Clean measurements to get sample and pond ID
  measurements = pd.read_excel("./data/raw/Testing Data Jun-Dec 2024_ID,Date,Time only.xls")
  measurements['sample_dt'] = pd.to_datetime(
    measurements['Date of data collection'].dt.strftime('%Y-%m-%d') + ' ' + 
    measurements['Time of data collection'].astype(str)
  )
  measurements['sample_dt'] = measurements['sample_dt'].dt.tz_localize(TIMEZONE)
  measurements['sample_idx'] = pd.Series(range(len(measurements)))

  measurements = measurements \
    .drop(columns=['Date of data collection', 'Time of data collection', 'Sr. No']) \
    .rename(columns={'Pond ID': 'pond_id'})

  # Merge ponds with pond metadata so you get the right variables.
  # Also fix the time one hot encoding now.
  ponds = gpd.read_file("./data/clean/pond_metadata_clean.geojson")

  keep_cols = ['pond_id', 'property_area_acres', 'pond_area_acres',
               'pond_depth_meters', 'geometry']
  
  ponds = ponds[keep_cols]
  samples = gpd.GeoDataFrame(
    measurements.merge(ponds, on='pond_id', how='left', validate='many_to_one'),
    crs=ponds.crs
  )

  # Record samples that lack a geometry
  # Make sample_idx index if sklearn doesn't process it.
  no_geom_samples = samples[samples['geometry'].isna()].copy()
  print(f"Ponds without locations: {no_geom_samples['pond_id'].unique().tolist()}")

  samples = samples[samples['geometry'].notna()]
  assert(samples['geometry'].isna().sum() == 0)

  # Create feature df if it doesn't exist or reexport flag is True
  predict_df_path = Path("./data/predict_dfs/trial/testing_data_jun_dec.csv")
  
  if not predict_df_path.exists() or re_export:
    print("Creating feature data for measurements.")
    gfs_gcs_filepath = "trial/gfs/testing_data_jun_dec.csv"
    gfs_download_root = Path("./data/gcs").resolve()
    description = Path(gfs_gcs_filepath).stem

    predict_df = create_standard_dataset(samples,
                                         gfs_gcs_filepath,
                                         gfs_download_root,
                                         description=description)
    
    predict_df_path.parent.mkdir(parents=True, exist_ok=True)
    predict_df.to_csv(predict_df_path)
  else:
    print("Loading existing feature data.")
    predict_df = pd.read_csv(predict_df_path)

  # Create X frame
  sample_idx = predict_df['sample_idx'].copy() # Keep sample idx for later merge.
  X = predict_df.drop(columns=['sample_dt', 'pond_id', 'geometry', 'sample_idx'])

  # Predict
  targets = ['do_in_range', 'ph_in_range', 'ammonia_in_range', 'turbidity_in_range']
  results_df = measurements.copy()

  for target in targets:
    model_dir = Path("./models").resolve() / "measurements_with_metadata_simple" / target

    # Load model
    for model_name in['Random Forest', 'Gradient Boosting']:
      with open(model_dir / f"{model_name}.pkl", 'rb') as f:
        model = pickle.load(f)

      with open(model_dir / "encoder.pkl", 'rb') as f:
        encoder = pickle.load(f)
      
      # Check that X has all required features in correct order
      model_features = model.feature_names_in_
      missing_features = set(model_features) - set(X.columns)
      if missing_features:
          raise ValueError(f"Missing required features: {missing_features}")
      extra_features = set(X.columns) - set(model_features) 
      if extra_features:
          print(f"Warning: Extra features will be ignored: {extra_features}")

      # Predict and store
      X_temp = X[model_features]  # Reorder columns to match model's expected order
      preds = pd.Series(encoder.inverse_transform(model.predict(X_temp)), index=sample_idx)
      var_name = f"{target}_pred_{model_name.lower().replace(' ', '_')}"
      results_df[var_name] = results_df['sample_idx'].map(preds)

  outpath = Path("./output").resolve() / "trial" / "testing_data_jun_dec_results.csv"
  outpath.parent.mkdir(parents=True, exist_ok=True)
  results_df.to_csv(outpath)


if __name__ == "__main__":
  main()
