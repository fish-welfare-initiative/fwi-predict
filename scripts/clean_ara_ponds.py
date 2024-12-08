import re

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from shapely.geometry import Point
from urllib.parse import unquote

column_map = {'Sr. No': 'pond_serial_no',
              'Date of data collection': 'pond_data_date',
              'Pond ID': 'pond_id',
              'Farmer': 'farmer',
              'Enrollment mechanism': 'enrollment_mechanism',
              'Culture change': 'culture_change',
              'Existing practices': 'existing_practices',
              'Notes (existing practices)': 'notes_existing_practices',
              'Fertilizers used': 'fertilizers_used',
              'Fish source (e.g. hatchery name)': 'fish_source',
              'Notes': 'notes',
              'Location': 'location',
              'Village': 'village',
              'Added by': 'added_by',
              'Property area in acres': 'property_area_acres',
              'Pond area in acres': 'pond_area_acres',
              'Depth in meters': 'pond_depth_meters',
              'Status': 'treatment_group', # Use treatment group from measurements since that hopefully reflects historical status
              'Measurements': 'measurements',
              'Equipment': 'equipment',
              'Feed type': 'feed_type',
              'Feed source': 'feed_source',
              'Feed brand or name': 'feed_brand'}


def dms_to_decimal(dms_string):
    """
    Parses a DMS string and converts it to decimal degrees.
    Automatically detects degrees, minutes, and seconds.
    """
    # Regex to match the DMS pattern
    pattern = r"(\d+)°(\d+)'([\d.]+)\"?([NSEW])"
    match = re.match(pattern, dms_string)

    if not match:
        raise ValueError(f"Invalid DMS format: {dms_string}")

    degrees, minutes, seconds, direction = match.groups()
    decimal_degrees = (
        float(degrees) + float(minutes) / 60 + float(seconds) / 3600
    )
    if direction in "SW":
        decimal_degrees = -decimal_degrees
    return decimal_degrees


def is_coordinate_string(s: str):
  """Chceck whether string expresses geographic coordinates."""
  pattern = r"^\d+\.\d+,\s\d+\.\d+$"
  return bool(re.match(pattern, s))


def is_gmaps_bitly(s: str):
  """Check whether is a short Google Maps URL."""
  if isinstance(s, str):
    return s.startswith("https://goo.gl/maps/")
  else:
    return False


def get_coords_from_gmaps_bitly(short_url: str) -> tuple:
  """Extract coordinates from short Google Maps URL."""
  # Expand the short URL
  response = requests.get(short_url, allow_redirects=True)
  full_url = response.url

  # Parse the URL to extract coordinates
  if '/place' in full_url:
    # Coordinates are in the path, e.g., @lat,lng
    dms = full_url.split('/place/')[1].split('/')[0]
    dms = unquote(unquote(dms))
    return tuple((dms_to_decimal(coord) for coord in dms.split('+')))
  elif '/search' in full_url:
    parsed_url = unquote(full_url)
    str_coords = parsed_url.split('/search/')[1].split('?')[0]
    return tuple((float(coord) for coord in str_coords.split(",+")))
  else:
    return "Coordinates not found in URL."


def load_coords(x: str) -> tuple:
  """Load coords as tuple depending on data type."""
  if not isinstance(x, str):
    return x

  elif is_coordinate_string(x):
    return tuple(float(coord) for coord in x.split(", "))
  
  elif is_gmaps_bitly(x):
    return get_coords_from_gmaps_bitly(x)
  
  else:
    return x


if __name__ == "__main__":

	ponds = pd.read_csv("data/raw/ara_exports/2024_11_27/ponds.csv",
											parse_dates=['Date of data collection'])
	
	ponds = ponds.rename(columns=column_map)
	assert(ponds.columns.isin(column_map.values()).all()) # Assert columns renamed

	ponds['culture_change'] = ponds['culture_change'].str.capitalize().str.replace('N0', 'No')

	# Not completely clean
	ponds['fish_source'] = ponds['fish_source'] \
		.str.capitalize() \
		.str.replace('farmers', 'farmer') \
		.str.replace('nurseries', 'nursery') \
		.str.replace('nursary', 'nursery') \
		.str.replace('nursury', 'nursery') \
		.str.replace('pond.', 'pond') \
		.str.replace('ponds', 'pond') \
		.str.replace('pond', '')
	
	# Get lat/lons as tuples
	ponds['location_parsed'] = ponds['location'].apply(load_coords)
   
 # Make sure we got coords for all gmaps links
	gmaps_parsed = ponds.loc[
  	ponds['location'].apply(is_gmaps_bitly), 'location_parsed'
	].apply(lambda x: isinstance(x, tuple)).all()
	assert(gmaps_parsed)

  # Load into gedf
	points = [Point(tup[1], tup[0])
						if isinstance(tup, tuple) else np.nan
						for tup in ponds['location_parsed'].tolist()]
	ponds = gpd.GeoDataFrame(ponds, geometry=points, crs=4326)
	ponds = ponds.drop(columns=['location', 'location_parsed'])

	ponds.to_file("data/clean/pond_metadata_clean.geojson")
