import streamlit as st
import pandas as pd
import plotly.express as px
import geopandas as gpd
import folium
from streamlit_folium import folium_static
from folium.plugins import MarkerCluster
import os
#from dotenv import load_dotenv

# Load environment variables
#load_dotenv()
GOOGLE_API_KEY = "AIzaSyDKOf6xvq1qKe0_mTms2WyAlNJTgUPRNj0" #os.getenv("GOOGLE_API_KEY")

# ---- Load Data ----
@st.cache_data
def load_data():
    # Mock data (replace with actual CSV later)
    data = {
        "pond_name": ["Pond A", "Pond B", "Pond C", "Pond D", "Pond E"] * 4,
        "pond_id": list(range(1, 21)),
        "prediction": ["above", "within", "within", "below", "above"] * 4,
        "model_confidence": [0.95, 0.7, 0.9, 0.85, 0.92] * 4,
        "latitude": [22.57 + i * 0.01 for i in range(20)],  # Mock latitudes in Eastern India
        "longitude": [88.36 + i * 0.01 for i in range(20)],  # Mock longitudes in Eastern India
    }
    df = pd.DataFrame(data)
    df["prediction"] = df["prediction"].str.lower()
    return df

df = load_data()

# ---- Streamlit UI ----
st.title("Pond Water Quality Predictions")

# ---- Filters ----
selected_ponds = st.multiselect("Filter by Pond", df["pond_name"].unique(), default=df["pond_name"].unique())
prediction_filter = st.multiselect("Filter by Prediction", ["above", "below", "within"], default=["above", "below", "within"])
confidence_threshold = st.slider("Filter by Model Confidence", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

# Apply Filters
filtered_df = df[
    (df["pond_name"].isin(selected_ponds)) &
    (df["prediction"].isin(prediction_filter)) &
    (df["model_confidence"] >= confidence_threshold)
]

# ---- Table Display ----
st.subheader("Dissolved Oxygen Data Table")

def color_predictions(val):
    color_map = {"above": "background-color: #f08080", "below": "background-color: #add8e6", "within": "background-color: #90ee90"}  # Red for above, green for within, blue for below
    return color_map.get(val, "")

styled_df = filtered_df.style.applymap(color_predictions, subset=["prediction"])
st.dataframe(styled_df, height=400)

# ---- Summary Count Below Table ----
st.subheader("Summary: Dissolved Oxygen Predictions")
summary_counts = filtered_df["prediction"].value_counts().to_dict()
st.write(f"**Above Threshold:** {summary_counts.get('above', 0)} ponds")
st.write(f"**Within Threshold:** {summary_counts.get('within', 0)} ponds")
st.write(f"**Below Threshold:** {summary_counts.get('below', 0)} ponds")

# ---- Geo Visualization ----
st.subheader("Pond Locations on Map")

# Create Folium Map with Google Satellite Basemap
m = folium.Map(location=[df["latitude"].mean(), df["longitude"].mean()], zoom_start=10)

# Add Google Satellite Layer if API Key Exists
if GOOGLE_API_KEY:
    folium.TileLayer(
        tiles=f"https://mt1.google.com/vt/lyrs=s&x={{x}}&y={{y}}&z={{z}}&key={GOOGLE_API_KEY}",
        attr="Google Maps",
        name="Google Satellite",
    ).add_to(m)
else:
    folium.TileLayer("Stamen Terrain", attr="Stamen", name="Stamen Terrain").add_to(m)

marker_cluster = MarkerCluster().add_to(m)
color_map = {"above": "red", "below": "blue", "within": "lightblue"}

for _, row in filtered_df.iterrows():
    folium.Marker(
        location=[row.latitude, row.longitude],
        popup=f"Pond: {row.pond_name}\nPrediction: {row.prediction}\nConfidence: {row.model_confidence:.2f}",
        icon=folium.Icon(color=color_map[row.prediction]),
    ).add_to(marker_cluster)

folium_static(m)

# ---- Stretch Goal: Interactive Time-Series Predictions ----
st.subheader("Time-Series Predictions")

# Mock time-series data
mock_timeseries_data = {
    "date": pd.date_range(start="2024-02-20", periods=10, freq="D").tolist() * 2,
    "time": ["4 PM"] * 10 + ["8 AM"] * 10,
    "pond_name": ["Pond A"] * 20,
    "dissolved_oxygen": [5.2, 5.5, 5.1, 5.3, 5.4, 5.2, 5.6, 5.7, 5.5, 5.3] + [4.8, 4.9, 5.0, 4.7, 5.1, 4.9, 4.8, 4.6, 5.0, 4.9]
}
df_timeseries = pd.DataFrame(mock_timeseries_data)

# Pond Selection
selected_pond = st.selectbox("Select a Pond", df_timeseries["pond_name"].unique())

# Days Filter
max_days = df_timeseries["date"].nunique()
num_days = st.slider("Select Number of Days", min_value=1, max_value=max_days, value=max_days)

# Apply Filters
filtered_timeseries = df_timeseries[df_timeseries["pond_name"] == selected_pond]
filtered_timeseries = filtered_timeseries.sort_values("date").groupby("date").head(num_days).reset_index()

# Interactive Time-Series Plot
fig = px.line(
    filtered_timeseries,
    x="date",
    y="dissolved_oxygen",
    color="time",
    markers=True,
    title=f"Dissolved Oxygen Levels for {selected_pond}",
)
st.plotly_chart(fig)

# ---- End ----
st.write("Fish Welfare Initiative")
