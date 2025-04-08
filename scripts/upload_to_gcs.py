from pathlib import Path

import click

from fwi_predict.gcs import upload_files

@click.command()
@click.argument('source_path', type=click.Path(exists=True))
@click.argument('destination_path', type=str)
@click.option('--bucket', type=str, default='fwi-predict', help='GCS bucket to upload to.')
@click.option('--project', type=str, default='fwi-water-quality-sensing', help='GCP project ID.')
@click.option('--recursive/--no-recursive', default=True, help='Whether to recursively upload folders.')
def upload_to_gcs(source_path, destination_path, bucket, project, recursive):
    """Upload files or folders to Google Cloud Storage.
    
    Args:
        source_path: Local file/folder path to upload. Can be a glob pattern.
        destination_path: Destination path in GCS bucket.
    """
    upload_files(source_path, destination_path, bucket, project, recursive)
    print(f"Uploaded {source_path} to gs://{bucket}/{destination_path}")

if __name__ == '__main__':
    upload_to_gcs()
