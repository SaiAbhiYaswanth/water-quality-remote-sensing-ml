import folium
from streamlit_folium import st_folium
import streamlit as st
import ee
import joblib
from utils import check_water

# Initialize Earth Engine
ee.Initialize(project='my-idp-project-472621')

st.set_page_config(page_title="Water Quality Prediction", layout="wide")

st.title("🌊 Remote Sensing Based Water Quality Monitoring System")
st.subheader("Select Location on Map")

# Default center (India)
m = folium.Map(location=[20.5, 78.9], zoom_start=5)

# Map click
map_data = st_folium(m, width=700, height=400)

if map_data and map_data["last_clicked"]:
    lat = map_data["last_clicked"]["lat"]
    lon = map_data["last_clicked"]["lng"]
    st.success(f"Selected Coordinates: {lat:.5f}, {lon:.5f}")

# Load models
chl_model = joblib.load("models/chl_model.pkl")
do_model = joblib.load("models/do_model.pkl")
nh3_model = joblib.load("models/nh3_model.pkl")

# Sidebar input
st.sidebar.header("Input Coordinates")

with st.sidebar.form("input_form"):
    lat = st.number_input("Latitude", format="%.6f")
    lon = st.number_input("Longitude", format="%.6f")
    submit = st.form_submit_button("Predict Water Quality")

# Reflectance fetch
def get_reflectance(lat, lon):

    point = ee.Geometry.Point(lon, lat)

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(point)
        .filterDate('2023-01-01', '2023-12-31')
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
    )

    image = collection.median()

    bands = image.select(['B2','B3','B4','B5','B6','B7','B8','B8A'])

    values = bands.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=point,
        scale=10
    ).getInfo()

    scaled = [values[b] / 10000 for b in ['B2','B3','B4','B5','B6','B7','B8','B8A']]
    return scaled


def get_satellite_image(lat, lon):

    point = ee.Geometry.Point(lon, lat)

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(point)
        .filterDate('2023-01-01', '2023-12-31')
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
    )

    size = collection.size().getInfo()

    if size == 0:
        return None

    image = collection.median().divide(10000)

    vis_params = {
        'bands': ['B4', 'B3', 'B2'],
        'min': 0.02,
        'max': 0.3,
        'gamma': 1.4
    }

    url = image.getThumbURL({
        'region': point.buffer(800),
        'dimensions': 600,
        'format': 'png',
        **vis_params
    })

    return url


def get_ndwi_map(lat, lon):

    point = ee.Geometry.Point(lon, lat)

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(point)
        .filterDate('2023-01-01', '2023-12-31')
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
    )

    size = collection.size().getInfo()

    if size == 0:
        return None

    image = collection.median()
    ndwi = image.normalizedDifference(['B3', 'B8'])

    vis_params = {
        'min': -0.5,
        'max': 0.5,
        'palette': ['brown', 'yellow', 'green', 'cyan', 'blue']
    }

    url = ndwi.getThumbURL({
        'region': point.buffer(800),
        'dimensions': 600,
        'format': 'png',
        **vis_params
    })

    return url


# Water classification
def classify_water_quality(chl, do, nh3):

    if chl < 10:
        chl_status = "Good"
    elif chl < 25:
        chl_status = "Moderate"
    else:
        chl_status = "Poor"

    if do > 7:
        do_status = "Good"
    elif do > 5:
        do_status = "Moderate"
    else:
        do_status = "Poor"

    if nh3 < 0.5:
        nh3_status = "Good"
    elif nh3 < 1.5:
        nh3_status = "Moderate"
    else:
        nh3_status = "Poor"

    if "Poor" in [chl_status, do_status, nh3_status]:
        overall = "Poor"
    elif chl_status == "Good" and do_status == "Good" and nh3_status == "Good":
        overall = "Good"
    else:
        overall = "Moderate"

    return overall, chl_status, do_status, nh3_status


# Prediction
if submit:

    st.subheader("Processing Location...")

    if lat == 0 and lon == 0:
        st.warning("Please enter valid coordinates")
        st.stop()

    st.info("Checking for water body...")
    is_water = check_water(lat, lon)

    if not is_water:
        st.error("❌ No water body detected at this location")
        st.stop()

    st.success("✅ Water body detected!")

    st.info("Fetching Sentinel-2 reflectance...")
    features = get_reflectance(lat, lon)

    st.subheader("Remote Sensing Dashboard")

    col1, col2, col3 = st.columns(3)

    # Satellite
    with col1:
        st.markdown("### 🛰️ Satellite Image")
        image_url = get_satellite_image(lat, lon)
        if image_url:
            st.image(image_url, use_container_width=True)
        else:
            st.warning("Satellite image unavailable")

    # NDWI
    with col2:
        st.markdown("### 🌊 NDWI Water Map")
        ndwi_url = get_ndwi_map(lat, lon)
        if ndwi_url:
            st.image(ndwi_url, use_container_width=True)
        else:
            st.warning("NDWI map unavailable")

    # Predictions
    with col3:
        st.markdown("### 📊 Predictions")

        chl = chl_model.predict([features])[0]
        do = do_model.predict([features])[0]
        nh3 = nh3_model.predict([features])[0]

        overall, chl_status, do_status, nh3_status = classify_water_quality(chl, do, nh3)

        st.metric("Chlorophyll-a", f"{chl:.2f}")
        st.metric("Dissolved Oxygen", f"{do:.2f}")
        st.metric("NH3-N", f"{nh3:.2f}")

        st.markdown("### Water Quality")

        if overall == "Good":
            st.success("GOOD")
        elif overall == "Moderate":
            st.warning("MODERATE")
        else:
            st.error("POOR")

    st.write("### Parameter-wise Status")
    st.write(f"Chlorophyll-a: {chl_status}")
    st.write(f"Dissolved Oxygen: {do_status}")
    st.write(f"NH3-N: {nh3_status}")
