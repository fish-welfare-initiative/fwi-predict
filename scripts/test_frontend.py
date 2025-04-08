import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from streamlit_folium import folium_static
from folium.plugins import MarkerCluster
import plotly.express as px

# Load or simulate GeoDataFrame
def load_data():
    data = {
        "farmer_name": ["Alice", "Bob", "Charlie", "David", "Eve"] * 5,
        "pond_id": range(1, 26),
        "prediction": ["below", "within", "above", "below", "above"] * 5,
        "model_confidence": [0.95, 0.7, 0.9, 0.85, 0.92] * 5,
        "geometry": [
            (10 + i * 0.01, 20 + i * 0.01) for i in range(25)
        ],  # Replace with actual Point geometries
    }
    gdf = gpd.GeoDataFrame(data)
    gdf["geometry"] = gdf["geometry"].apply(lambda x: gpd.points_from_xy([x[0]], [x[1]])[0])
    return gdf

gdf = load_data()

# Summary statistics
st.title("Pond Water Quality Predictions")
summary = gdf["prediction"].value_counts()
st.table(pd.DataFrame({"Category": ["Within", "Above", "Below", "Total"], "Count": [summary.get("within", 0), summary.get("above", 0), summary.get("below", 0), len(gdf)]}))

gdf_sorted = gdf.sort_values(by=["model_confidence"], ascending=False)

def display_table():
    st.dataframe(
        gdf_sorted.style.applymap(
            lambda x: "background-color: #add8e6" if x == "below" else 
                      "background-color: #90ee90" if x == "within" else 
                      "background-color: #f08080",
            subset=["prediction"]
        ), height=400
    )

target_farmers = st.multiselect("Filter by Farmer", gdf["farmer_name"].unique())
if target_farmers:
    gdf_sorted = gdf_sorted[gdf_sorted["farmer_name"].isin(target_farmers)]

prediction_filter = st.multiselect("Filter by Prediction", ["below", "within", "above"], default=["below", "above"])
gdf_sorted = gdf_sorted[gdf_sorted["prediction"].isin(prediction_filter)]

# Create color mapping
color_map = {"below": "blue", "within": "green", "above": "red"}

# Create Folium Map
m = folium.Map(location=[10, 20], zoom_start=6)
marker_cluster = MarkerCluster().add_to(m)

for _, row in gdf.iterrows():
    folium.Marker(
        location=[row.geometry.y, row.geometry.x],
        popup=f"Pond ID: {row.pond_id}\nConfidence: {row.model_confidence:.2f}",
        icon=folium.Icon(color=color_map[row.prediction]),
    ).add_to(marker_cluster)

# Layout: side-by-side map and dataframe
col1, col2 = st.columns([1, 1])
with col1:
    folium_static(m)
with col2:
    display_table()
