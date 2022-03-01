#%%

import os
import glob

# roi_shp = r"D:\co2_data\ENA\ena_boundary\Ena_City.shp"
# gifu_img_dir = r"D:\co2_data\DL\large_img\sentinel\preprocessing\10m"
# tono_recls_dir = r"D:\co2_data\DL\large_img\sentinel\s2_tono_recls\l2"

shp_dir = r"D:\SHP\tono\city"

# cities = ["tajimi", "toki", "mizunami", "nakat"]

# gifu_img_list = glob.glob(os.path.join(tono_recls_dir, "*.tif"))
# for city in cities:
#     print(city)
#     roi_shp = os.path.join(shp_dir, f"{city}.shp")
#     for img in gifu_img_list:
#         # out_tif_path = img.replace("10m", city)
#         out_tif_path = img.replace("tono", city)
#         if not os.path.exists(out_tif_path):
#             gdal_cmd = f"gdalwarp -srcnodata -9999 -dstnodata -9999 -cutline \
#                         {roi_shp} -crop_to_cutline {img} {out_tif_path}"
#             os.system(gdal_cmd)
#             print(img.split("\\")[-1][:-4])

# %%
tono_forest = r"D:\co2-seq\forest_attr_segment\data\forest_map\tono_forest_map.tif"
cities = ["tajimi", "toki", "mizunami"]
for city in cities:
    print(city)
    roi_shp = os.path.join(shp_dir, f"{city}.shp")
    out_tif_path = tono_forest.replace("tono", city)
    gdal_cmd = f"gdalwarp -srcnodata -9999 -dstnodata -9999 -cutline \
                        {roi_shp} -crop_to_cutline {tono_forest} {out_tif_path}"
    os.system(gdal_cmd)

# %%
