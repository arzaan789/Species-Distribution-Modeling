import pandas as pd
from pyproj import Transformer
import rasterio
from rasterio.windows import Window
import numpy as np


# Load and filter data more efficiently
def load_filter_data(filepath):
    df = pd.read_csv(filepath, delimiter='\t', nrows=5)
    df = df[df["occurrenceStatus"] == "PRESENT"]
    return df[['species', 'decimalLatitude', 'decimalLongitude']]


# Load all data at once
files = ['hedgehog.csv', 'squirrel.csv', 'fox.csv']
dfs = [load_filter_data(f) for f in files]
df = pd.concat(dfs, ignore_index=True)

# Land cover mapping
land_cover_map = {
    1: "Deciduous woodland",
    2: "Coniferous woodland",
    3: "Arable",
    4: "Improved grassland",
    5: "Neutral grassland",
    6: "Calcareous grassland",
    7: "Acid grassland",
    8: "Fen",
    9: "Heather",
    10: "Heather grassland",
    11: "Bog",
    12: "Inland rock",
    13: "Saltwater",
    14: "Freshwater",
    15: "Supralittoral rock",
    16: "Supralittoral sediment",
    17: "Littoral rock",
    18: "Littoral sediment",
    19: "Saltmarsh",
    20: "Urban",
    21: "Suburban"
}

# Batch coordinate transformation
transformer_ni = Transformer.from_crs("EPSG:4326", "EPSG:29903", always_xy=True)
transformer_gb = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)

coords = list(zip(df['decimalLongitude'], df['decimalLatitude']))
df['easting_ni'], df['northing_ni'] = zip(*transformer_ni.itransform(coords))
df['easting_gb'], df['northing_gb'] = zip(*transformer_gb.itransform(coords))

# Raster processing optimization
gb_raster = 'gblcm2023_10m.tif'
n_ireland_raster = 'nilcm2023_10m.tif'


def get_land_cover_class(row):
    try:
        # Try GB raster first
        with rasterio.open(gb_raster) as src:
            row_idx, col_idx = src.index(row['easting_gb'], row['northing_gb'])
            # Read a small window around the point for better performance
            window = Window(col_idx, row_idx, 1, 1)
            land_cover_class = src.read(1, window=window)[0, 0]

            if land_cover_class == 0:  # Check NI raster if GB is 0
                with rasterio.open(n_ireland_raster) as src_ni:
                    row_idx, col_idx = src_ni.index(row['easting_ni'], row['northing_ni'])
                    window = Window(col_idx, row_idx, 1, 1)
                    land_cover_class = src_ni.read(1, window=window)[0, 0]

        return land_cover_map.get(land_cover_class, "Unknown")
    except Exception as e:
        print(f"Error processing row: {e}")
        return "Unknown"


# Vectorized operation (faster than iterrows)
df['Land_cover'] = df.apply(get_land_cover_class, axis=1)

print(df['Land_cover'])