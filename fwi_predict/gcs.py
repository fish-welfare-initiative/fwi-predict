from pathlib import Path
from typing import Union

from google.cloud import storage


def download_files(bucket: str,
                   file_glob: Union[str, Path],
				   				 download_dir: str,
                   project: str = 'fwi-water-quality-sensing') -> None:
	"""Download files from GCS bucket."""
	client = storage.Client(project=project)
	bucket = client.bucket(bucket)
	glob = Path(file_glob).as_posix()
	file_blobs = bucket.list_blobs(match_glob=glob)

	for blob in file_blobs:
		fp = Path(download_dir) / blob.name
		fp.parent.mkdir(parents=True, exist_ok=True)
		blob.download_to_filename(fp)
