#%%
import rasterio
import geopandas as gpd
import numpy as np

from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, confusion_matrix

# Read points from shapefile
shp_file = "../data/tree_spec_val/tree_spec_val_shp.shp"
spec_file = "../data/spec_map/high-res/ena_spec_recls_encBotDec_7780_km.tif"

pts = gpd.read_file(shp_file)

gt = pts["cls"].values
coords = pts["geometry"]
coords = [(x, y) for x, y in zip(coords.x, coords.y)]

# # Open the raster and store metadata
src = rasterio.open(spec_file)

preds = [x for x in src.sample(coords)]
preds = np.ravel(preds)

print(f"OA: {accuracy_score(gt, preds)}")
# print(f"f1: {f1_score(gt, preds)}")
print(f"kappa: {cohen_kappa_score(gt, preds)}")

cf = confusion_matrix(gt, preds)


# %%
