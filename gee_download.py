import ee
import pandas as pd

# Initialize Earth Engine
ee.Initialize(project='my-idp-project-472621')

print("Earth Engine initialized successfully!")

# Global waterbody sample points
LOCATIONS = [
    (16.5, 80.6),
    (17.2, 78.4),
    (13.0, 77.5),
    (25.3, 83.0),
    (22.5, 88.3),
    (9.9, 76.2),
    (19.1, 72.9),
    (11.0, 79.8)
]

# Sentinel bands
BANDS = ['B2','B3','B4','B5','B6','B7','B8','B8A']

def mask_and_select(image):
    return image.select(BANDS)

def get_images_for_location(lat, lon):
    point = ee.Geometry.Point(lon, lat)

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(point)
        .filterDate('2022-01-01', '2023-12-31')
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
        .map(mask_and_select)
    )

    return collection.toList(collection.size()), point

def extract_values(image, point):
    values = image.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=point,
        scale=10
    ).getInfo()
    return values

def build_large_dataset():
    rows = []

    for lat, lon in LOCATIONS:
        print(f"\nProcessing location: {lat}, {lon}")

        images_list, point = get_images_for_location(lat, lon)
        n = images_list.size().getInfo()

        print(f"Total images found: {n}")

        for i in range(n):
            try:
                img = ee.Image(images_list.get(i))
                vals = extract_values(img, point)

                if vals:
                    vals['lat'] = lat
                    vals['lon'] = lon
                    rows.append(vals)

            except Exception as e:
                print("Skipping image due to error:", e)

    df = pd.DataFrame(rows)
    df.to_csv("data/reflectance_dataset.csv", index=False)

    print("\nLarge dataset saved to data/reflectance_dataset.csv")
    print("Total samples:", len(df))


if __name__ == "__main__":
    build_large_dataset()
