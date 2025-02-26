import streamlit as st
import pandas as pd
import plotly.express as px
import geopandas as gpd
import folium
from streamlit_folium import folium_static
from folium.plugins import MarkerCluster

# ---- Load Data ----
@st.cache_data
def load_data():
    # Mock data (replace with actual CSV later)
    data = {
        "pond_name": ["Pond A", "Pond B", "Pond C", "Pond D", "Pond E"] * 4,
        "pond_id": list(range(1, 21)),
        "prediction": ["above", "below", "above", "below", "above"] * 4,
        "model_confidence": [0.95, 0.7, 0.9, 0.85, 0.92] * 4,
        "latitude": [34.05 + i * 0.01 for i in range(20)],  # Mock latitudes
        "longitude": [-118.25 - i * 0.01 for i in range(20)],  # Mock longitudes
    }
    df = pd.DataFrame(data)
    df["prediction"] = df["prediction"].str.lower()
    return df

# Load Data
df = load_data()

# ---- Streamlit UI ----
st.title("Pond Water Quality Predictions")

# ---- Filters ----
selected_ponds = st.multiselect("Filter by Pond", df["pond_name"].unique(), default=df["pond_name"].unique())
prediction_filter = st.multiselect("Filter by Prediction", ["above", "below"], default=["above", "below"])
confidence_threshold = st.slider("Filter by Model Confidence", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

# Apply Filters
filtered_df = df[
    (df["pond_name"].isin(selected_ponds)) &
    (df["prediction"].isin(prediction_filter)) &
    (df["model_confidence"] >= confidence_threshold)
]

# ---- Table Display ----
st.subheader("Pond Data Table")

def color_predictions(val):
    color_map = {"above": "background-color: #f08080", "below": "background-color: #90ee90"}  # Red for above, green for below
    return color_map.get(val, "")

styled_df = filtered_df.style.applymap(color_predictions, subset=["prediction"])
st.dataframe(styled_df, height=400)

# ---- Geo Visualization ----
st.subheader("Pond Locations on Map")

# Create Folium Map
m = folium.Map(location=[df["latitude"].mean(), df["longitude"].mean()], zoom_start=6)
marker_cluster = MarkerCluster().add_to(m)

# Define color mapping
color_map = {"above": "red", "below": "blue"}

# Add markers
for _, row in filtered_df.iterrows():
    folium.Marker(
        location=[row.latitude, row.longitude],
        popup=f"Pond: {row.pond_name}\nPrediction: {row.prediction}\nConfidence: {row.model_confidence:.2f}",
        icon=folium.Icon(color=color_map[row.prediction]),
    ).add_to(marker_cluster)

# Display Map
folium_static(m)

# ---- Stretch Goal: Time-Series Predictions ----
st.subheader("Future Feature: Time-Series Predictions (Mock Example)")

mock_timeseries_data = {
    "date": pd.date_range(start="2024-02-20", periods=10, freq="D").tolist() * 2,
    "time": ["8 AM"] * 10 + ["4 PM"] * 10,
    "pond_name": ["Pond A"] * 10 + ["Pond B"] * 10,
    "dissolved_oxygen": [5.2, 5.5, 5.1, 5.3, 5.4, 5.2, 5.6, 5.7, 5.5, 5.3] + [4.8, 4.9, 5.0, 4.7, 5.1, 4.9, 4.8, 4.6, 5.0, 4.9]
}
df_timeseries = pd.DataFrame(mock_timeseries_data)

fig = px.line(df_timeseries, x="date", y="dissolved_oxygen", color="pond_name", markers=True, line_dash="time")
st.plotly_chart(fig)

# ---- End ----
st.write("Fish Welfare Initiative")
