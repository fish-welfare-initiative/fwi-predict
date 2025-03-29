from pathlib import Path

import click
import geopandas as gpd

from fwi_predict.gcs import upload_files
from fwi_predict.pipeline import create_standard_dataset

@click.command()
@click.argument('samples_path', type=click.Path(exists=True))
@click.option('--outdir', type=click.Path(), default="./data/predict_dfs/train", help='Relative path to save file.')
@click.option('--gfs_download_root', type=str, default='./data/gcs', help='Root directory in which to save file.')
@click.option('--gcs_bucket', type=str, default='fwi-predict', help='GCS bucket to save file to.')
@click.option('--gee_project', type=str, default='fwi-water-quality-sensing', help='GEE project to use for export.')
def create_dataset(samples_path, outdir, gfs_download_root, gcs_bucket, gee_project):
	"""Create standard training dataset."""
	filename = Path(samples_path).stem
	gcs_filepath = Path("train") / "gfs" / f"{filename}.csv"
	gfs_download_root = Path(gfs_download_root).resolve()

	samples = gpd.read_file(samples_path)
	ds = create_standard_dataset(samples, gcs_filepath, gfs_download_root,
															 filename, gcs_bucket, gee_project)
		
	if ds is not None:
		# Save locally
		outdir = Path(outdir).resolve()
		outdir.mkdir(parents=True, exist_ok=True)
		outpath = outdir / f"{filename}_predict_df.csv"
		ds.to_csv(outpath)
		print(f"Training data created.\nSaved to {outpath}.")

		# Upload to GCS
		gcs_dest = Path("train") / "predict_dfs" / f"{filename}_predict_df.csv"
		upload_files(outpath, gcs_dest, gcs_bucket, project=gee_project)
		print(f"Uploaded to gs://{gcs_bucket}/{gcs_dest}")
	else:
		print("Training data creation failed.")


if __name__ == '__main__':
	create_dataset()