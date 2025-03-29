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


def upload_files(source_path: Union[str, Path],
                 destination_path: Union[str, Path],
                 bucket: str,
                 project: str = 'fwi-water-quality-sensing',
                 recursive: bool = True) -> None:
    """Upload files or folders to Google Cloud Storage.
    
    Args:
        source_path: Local file/folder path to upload. Can be a glob pattern.
        destination_path: Destination path in GCS bucket.
        bucket: Name of the GCS bucket.
        project: GCP project ID. Defaults to 'fwi-water-quality-sensing'.
        recursive: Whether to recursively upload folders. Defaults to True.
    """
    client = storage.Client(project=project)
    bucket = client.bucket(bucket)
    
    source = Path(source_path)
    dest = Path(destination_path)
    
    # Handle both file and folder uploads
    if source.is_file():
        files = [source]
    else:
        # Use rglob for recursive search, glob for non-recursive
        glob_func = source.rglob if recursive else source.glob
        files = list(glob_func('*'))
        
    for file_path in files:
        if file_path.is_file():
            # Calculate relative path to maintain folder structure
            rel_path = file_path.relative_to(source.parent if source.is_file() else source)
            blob_path = dest / rel_path
            
            # Create blob and upload
            blob = bucket.blob(blob_path.as_posix())
            blob.upload_from_filename(file_path)
