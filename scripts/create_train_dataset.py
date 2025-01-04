import os

import click
import geopandas as gpd

from fwi_predict.pipeline import create_standard_dataset

@click.command()
@click.argument('samples_path', type=click.Path(exists=True))
@click.option('--outdir', type=click.Path(), default="./data/predict_dfs/train", help='Relative path to save file.')
@click.option('--gfs_download_root', type=str, default='./data/gcs', help='Root directory in which to save file.')
@click.option('--gcs_bucket', type=str, default='fwi-predict', help='GCS bucket to save file to.')
@click.option('--gee_project', type=str, default='fwi-water-quality-sensing', help='GEE project to use for export.')
def create_dataset(samples_path, outdir, gfs_download_root, gcs_bucket, gee_project):
	"""Create standard training dataset."""
	filename = os.path.splitext(os.path.basename(samples_path))[0]
	gcs_filepath = os.path.join("train", "gfs", filename + '.csv')

	samples = gpd.read_file(samples_path)
	ds = create_standard_dataset(samples, gcs_filepath, gfs_download_root,
								 							 filename, gcs_bucket, gee_project)
		
	if ds is not None:
		os.makedirs(outdir, exist_ok=True)
		outpath = os.path.join(outdir, filename + '_predict_df.csv')
		ds.to_csv(outpath)
		print(f"Dataset created and saved to {outpath}.")


if __name__ == '__main__':
	create_dataset()