#%%

import os
import glob

roi_shp = r"D:\co2_data\ENA\ena_boundary\Ena_City.shp"
gifu_img_dir = r"D:\co2_data\DL\large_img\sentinel\preprocessing\10m"


gifu_img_list = glob.glob(os.path.join(gifu_img_dir, "*.tif"))

for img in gifu_img_list:
    out_tif_path = img.replace("10m", "nakat")
    if not os.path.exists(out_tif_path):
        gdal_cmd = f"gdalwarp -srcnodata -9999 -dstnodata -9999 -cutline \
                    {roi_shp} -crop_to_cutline -dstalpha {img} {out_tif_path}"
        os.system(gdal_cmd)
        print(img.split("\\")[-1][:-4])
