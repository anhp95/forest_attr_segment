#%%
import rasterio
import pandas as pd
import geopandas as gpd
import numpy as np
import os
import seaborn as sns

from tif import read_tif


WAVE_LENGTH = {
    "b2": 496.6,
    "b3": 560,
    "b4": 664.5,
    "b5": 703.9,
    "b6": 740.2,
    "b7": 782.5,
    "b8": 835.1,
    "b8a": 864.8,
    "b11": 1613.7,
    "b12": 2202.4,
}


def get_coords_gt(shp_file, cls_val):

    pts = gpd.read_file(shp_file)
    pts = pts.loc[pts["cls"] == cls_val]
    coords = pts["geometry"]
    coords = [(x, y) for x, y in zip(coords.x, coords.y)]

    return coords


def get_coords_spec(spec_file, cls_val):

    img, _ = read_tif(spec_file)
    img = np.ravel(img)

    return np.where(img == cls_val)[0]


def sample_img_rasterio(img_path, coords):

    src = rasterio.open(img_path)
    sig = [x for x in src.sample(coords)]

    return np.ravel(sig)


def sample_img_gdal(img_path, coords):

    img, _ = read_tif(img_path)
    img = np.ravel(img)

    return img[coords]


def extract_sig(img_dir, wave_length, coords, spec, data="gt"):

    bands = wave_length.keys()

    sigs = []
    wls = []
    specs = []
    data_types = []
    for band in bands:
        img_path = os.path.join(img_dir, f"{band}.tif")
        if data == "gt":
            sig = sample_img_rasterio(img_path, coords)
        else:
            sig = sample_img_gdal(img_path, coords)
        sigs.append(sig)
        wls.append([wave_length[band] for _ in sig])
        specs.append([spec for _ in sig])
        data_types.append([data for _ in sig])

    data = {
        "sig": np.ravel(sigs),
        "wl": np.ravel(wls),
        "spec": np.ravel(specs),
        "data_type": np.ravel(data_types),
    }
    df = pd.DataFrame(data)

    return df


def agg_sig_gt(shp_file, img_dir):

    df_sugi = extract_sig(img_dir, WAVE_LENGTH, get_coords_gt(shp_file, 1), "sugi")
    df_bf = extract_sig(img_dir, WAVE_LENGTH, get_coords_gt(shp_file, 2), "bf")
    df_cf = extract_sig(img_dir, WAVE_LENGTH, get_coords_gt(shp_file, 3), "cf")
    df_hinoki = extract_sig(img_dir, WAVE_LENGTH, get_coords_gt(shp_file, 4), "hinoki")

    df_spec = (
        df_sugi.append(df_bf, ignore_index=True)
        .append(df_cf, ignore_index=True)
        .append(df_hinoki, ignore_index=True)
    )
    return df_spec


def agg_sig_spec(spec_file, img_dir):
    df_sugi = extract_sig(
        img_dir, WAVE_LENGTH, get_coords_spec(spec_file, 1), "sugi", "spec"
    )
    df_bf = extract_sig(
        img_dir, WAVE_LENGTH, get_coords_spec(spec_file, 2), "bf", "spec"
    )
    df_cf = extract_sig(
        img_dir, WAVE_LENGTH, get_coords_spec(spec_file, 3), "cf", "spec"
    )
    df_hinoki = extract_sig(
        img_dir, WAVE_LENGTH, get_coords_spec(spec_file, 4), "hinoki", "spec"
    )

    df_spec = (
        df_sugi.append(df_bf, ignore_index=True)
        .append(df_cf, ignore_index=True)
        .append(df_hinoki, ignore_index=True)
    )
    return df_spec


def plot(df):
    sns.set_theme(style="darkgrid")
    sns.lineplot(x="wave l", y="sig", hue="spec", style="data_type", data=df)


def main():
    shp_file = "../data/tree_spec_val/tree_spec_val_shp.shp"

    spec_file = f"../data/spec_map/high-res/toki_spec_3d_adj_emd_acb_7749_km10.tif"

    img_dir = r"D:\co2_data\DL\large_img\sentinel\s2_ena_recls\l2"

    sig_gt = agg_sig_gt(shp_file, img_dir)
    # sig_spec = agg_sig_spec(spec_file, img_dir)

    # sig_plot = sig_gt.append(sig_spec, ignore_index=True)

    plot(sig_gt)


# %%
