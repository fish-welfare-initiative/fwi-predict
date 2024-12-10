"""Utilities for visualization."""
from typing import Any
from typing import Dict
from typing import Optional

import folium


basemaps = {
	"Google Satellite": folium.TileLayer(
		tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
		attr="Google",
		name="Google Satellite",
		overlay=True,
		control=True,
	),
	"Esri Satellite": folium.TileLayer(
		tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
		attr="Esri",
		name="Esri Satellite",
		overlay=True,
		control=True,
	),
}


def create_map(
	lat: float,
	lon: float,
	basemap: str = "Google Satellite",
	map_kwargs: Optional[Dict[str, Any]] = None,
) -> folium.Map:
	"""Generate map centered on latitude longitude pair.

	Args:
		lat: latitude
		lon: longitude
		layer_control: whether to add layer control to map
		basemap: basemap to use
		map_kwargs: additional kwargs to pass to folium.Map

	Returns:
		Folium Map object
	"""
	if not map_kwargs:
		map_kwargs = {}

	map = folium.Map(location=(lat, lon), **map_kwargs)

	basemaps[basemap].add_to(map)

	return map
