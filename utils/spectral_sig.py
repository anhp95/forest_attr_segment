#%%
import rasterio
import geopandas as gpd
import numpy as np
import os
import glob

def get_coords(shp_file, cls_val):

    pts = gpd.read_file(shp_file)
    pts = pts.loc[pts['cls'] == cls_val]
    coords = pts["geometry"]
    coords = [(x, y) for x, y in zip(coords.x, coords.y)]

    return coords

def extract_sig(bands, coords):

    sig_list = []
    for band in bands:
        src = rasterio.open(band)

        sig = [x for x in src.sample(coords)]
        sig_list.append(np.mean(np.ravel(sig)))
    
    return sig_list

def plot():
    pass

shp_file = "../data/tree_spec_val/tree_spec_val_shp.shp"
img_dir = r"D:\co2_data\DL\large_img\sentinel\s2_ena_recls\l2"

bands = ["b2", "b3", "b4", "b5", "b6", "b7", "b8", "b8a", "b11", "b12"]

imgs = [os.path.join(img_dir,f"{band}.tif") for band in bands]

sugi_list = extract_sig(imgs, get_coords(shp_file, cls_val=1))
bf_list = extract_sig(imgs, get_coords(shp_file, cls_val=2))
cf_list = extract_sig(imgs, get_coords(shp_file, cls_val=3))
hinoki_list = extract_sig(imgs, get_coords(shp_file, cls_val=4))


# %%
