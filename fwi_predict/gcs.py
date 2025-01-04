import os

from google.cloud import storage


def download_files(bucket: str,
                   file_glob: str,
				   				 download_dir: str,
                   project: str = 'fwi-predict') -> None:
	"""Download files from GCS bucket."""
	storage_client = storage.Client(project=project)
	bucket = storage_client.bucket(bucket)
	file_blobs = bucket.list_blobs(match_glob=file_glob)

	for blob in file_blobs:
		fp = os.path.join(download_dir, blob.name)
		os.makedirs(os.path.dirname(fp), exist_ok=True)
		blob.download_to_filename(fp)

