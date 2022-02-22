#%%
import rasterio
import geopandas as gpd
import numpy as np

from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, confusion_matrix

def point_val(shp_file, spec_file):
    
    #Read coords and truth value from shapefile to DF
    pts = gpd.read_file(shp_file)

    gt = pts["cls"].values
    coords = pts["geometry"]
    coords = [(x, y) for x, y in zip(coords.x, coords.y)]

    # Read species map
    src = rasterio.open(spec_file)

    # Sample the image by points' coords
    preds = [x for x in src.sample(coords)]
    preds = np.ravel(preds)

    # Calculate the metrics
    print(f"OA: {accuracy_score(gt, preds)}")
    print(f"f1: {f1_score(gt, preds, average='weighted')}")
    print(f"kappa: {cohen_kappa_score(gt, preds)}")

    cf = confusion_matrix(gt, preds)

if __name__ == '__main__':
    
    shp_file = "../data/tree_spec_val/tree_spec_val_shp.shp"
    spec_file = "../data/spec_map/high-res/ena_spec_recls_encBotDec_7780_km.tif"

    point_val(shp_file, spec_file)

# %%
