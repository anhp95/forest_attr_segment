#%%
import os
import glob

BB_DIR = r"D:\co2_data\DL\bb_box\tono\individual_bb"

SMALL_DIR = r"D:\co2_data\DL\small_img"
LARGE_DIR = r"D:\co2_data\DL\large_img"

bb_shp_list = glob.glob(os.path.join(BB_DIR, "*.shp"))

# list_sub_folders = ["age", "spec", "timber"]
# list_sub_folders = ["age", "sentinel", "spec", "timber"]
list_sub_folders = ["sentinel"]
region = r"tono_469"
resolution = "10m"
dict = {}

for folder in list_sub_folders:

    list_large_img = glob.glob(
        os.path.join(LARGE_DIR, folder, "preprocessing", region, "*.tif")
    )

    small_img_folder = os.path.join(SMALL_DIR, folder, region)
    print(small_img_folder)

    for large_img in list_large_img:
        print(large_img)
        for bb_shp in bb_shp_list:

            tif_name = large_img.split("\\")[-1][:-4]
            bb_shp_name = bb_shp.split("\\")[-1][:-4]
            name = f"{tif_name}_{bb_shp_name}.tif"
            out_tif_path = os.path.join(small_img_folder, name)
            if not os.path.exists(out_tif_path):
                gdal_cmd = f"gdalwarp -srcnodata -9999 -dstnodata -9999 -cutline {bb_shp} -crop_to_cutline -dstalpha {large_img} {out_tif_path}"
                os.system(gdal_cmd)
