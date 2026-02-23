import ee

# Initialize Earth Engine
ee.Initialize(project='my-idp-project-472621')


# Function to fetch Sentinel image
def get_sentinel_image(lat, lon):

    point = ee.Geometry.Point(lon, lat)

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(point)
        .filterDate('2023-01-01', '2023-12-31')
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
    )

    size = collection.size().getInfo()

    # If no image found
    if size == 0:
        return None, point

    image = collection.median()
    return image, point


# Compute NDWI, MNDWI, NDVI
def compute_indices(image):

    ndwi = image.normalizedDifference(['B3', 'B8'])
    mndwi = image.normalizedDifference(['B3', 'B11'])
    ndvi = image.normalizedDifference(['B8', 'B4'])

    return ndwi, mndwi, ndvi


# Check if pixel is water
def check_water(lat, lon):

    image, point = get_sentinel_image(lat, lon)

    # If no satellite data available
    if image is None:
        print("No Sentinel data available for this location.")
        return False

    ndwi, mndwi, ndvi = compute_indices(image)

    # Use 3x3 pixel window instead of single pixel
    region = point.buffer(30)  # 30 meters around point

    ndwi_val = ndwi.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=region,
        scale=10
    ).getInfo()['nd']

    mndwi_val = mndwi.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=region,
        scale=10
    ).getInfo()['nd']

    ndvi_val = ndvi.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=region,
        scale=10
    ).getInfo()['nd']

    # NIR reflectance (water absorbs NIR strongly)
    nir_val = image.select('B8').reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=region,
        scale=10
    ).getInfo()['B8']

    # Normalize NIR
    nir_val = nir_val / 10000 if nir_val else 0

    print("\nNDWI:", ndwi_val)
    print("MNDWI:", mndwi_val)
    print("NDVI:", ndvi_val)
    print("NIR:", nir_val)

    # Probability scoring
    score = 0

    if ndwi_val and ndwi_val > 0:
        score += 0.3

    if mndwi_val and mndwi_val > 0.1:
        score += 0.3

    if ndvi_val and ndvi_val < 0.2:
        score += 0.2

    if nir_val < 0.15:
        score += 0.2

    print("Water probability score:", score)

    # Final decision
    if score >= 0.6:
        return True
    else:
        return False
