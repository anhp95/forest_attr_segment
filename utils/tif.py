from osgeo import gdal
import glob
import os
import numpy as np


def read_tif(path_):
    ds = gdal.Open(path_)
    nodata_value = 65535
    ds.GetRasterBand(1).SetNoDataValue(nodata_value)
    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray()
    [rows, cols] = arr.shape

    metadata = {
        "transform": ds.GetGeoTransform(),
        "prj": ds.GetProjection(),
        "rows": rows,
        "cols": cols,
    }

    return (arr, metadata)


def write_tif(out_arr, metadata, filename, nodata_value=-9999):
    driver = gdal.GetDriverByName("GTiff")

    transfrom = metadata["transform"]
    prj = metadata["prj"]
    rows = metadata["rows"]
    cols = metadata["cols"]

    number_of_bands = 1
    band_type = gdal.GDT_Float32

    out_data = driver.Create(filename, cols, rows, number_of_bands, band_type)
    out_data.SetGeoTransform(transfrom)
    out_data.SetProjection(prj)
    out_data.GetRasterBand(1).WriteArray(out_arr)
    out_data.GetRasterBand(1).SetNoDataValue(nodata_value)
    out_data.FlushCache()

    out_data = None


def resample(ref_file, input_file, output_file):
    # open reference file and get resolution
    reference = gdal.Open(ref_file, 0)  # this opens the file in only reading mode
    referenceTrans = reference.GetGeoTransform()
    x_res = referenceTrans[1]
    y_res = -referenceTrans[5]  # make sure this value is positive

    # call gdal Warp
    kwargs = {"format": "GTiff", "xRes": x_res, "yRes": y_res}
    ds = gdal.Warp(output_file, input_file, **kwargs)
    return ds


def main_resample(ref_file, input_dir):

    # ref_file = r"D:\co2_data\DL\large_img\sentinel\S2_RAW\s2l2\3\GRANULE\L2A_T53SQV_A015701_20180625T013653\IMG_DATA\R10m\T53SQV_20180625T013651_B02_10m.jp2"
    # input_dir = r"D:\co2_data\DL\large_img\sentinel\S2_RAW\s2l2\2\GRANULE\L2A_T53SPV_A015701_20180625T013653\IMG_DATA\R20m"
    input_list = glob.glob(os.path.join(input_dir, "*.jp2"))

    for input_file in input_list:
        output_file = input_file.replace("20m.jp2", "10m.tif")

        resample(ref_file, input_file, output_file)
        print(output_file)


def merge():
    base_dir = r"D:\co2_data\DL\large_img\sentinel\S2_RAW\preprocessed_s2"
    list_1 = glob.glob(os.path.join(base_dir, "2", "*.tif"))
    list_2 = glob.glob(os.path.join(base_dir, "3", "*.tif"))

    for tif_1, tif_2 in zip(list_1, list_2):
        filename = tif_1[-11:]
        outfile = os.path.join(base_dir, "merged", filename)

        vrt = gdal.BuildVRT("merged.vrt", [tif_1, tif_2])
        gdal.Translate(outfile, vrt, xRes=10, yRes=-10)
        print(outfile)
        vrt = None


def clip():
    tono_shp = r"D:\SHP\tono\Tono.shp"
    base_dir = r"D:\co2_data\DL\large_img\sentinel\S2_RAW\preprocessed_s2"
    list_tif = glob.glob(os.path.join(base_dir, "merged", "*.tif"))
    for tif in list_tif:
        outfile = tif.replace("merged", "clip")
        gdal_cmd = f"gdalwarp -srcnodata -9999 -dstnodata -9999 -cutline {tono_shp} -crop_to_cutline -dstalpha {tif} {outfile}"
        os.system(gdal_cmd)
        print(outfile)


def dn_to_ref_norm():
    # dn = 1e4 * reflectance
    base_dir = r"D:\co2_data\DL\large_img\sentinel\S2_RAW\preprocessed_s2"
    list_tif = glob.glob(os.path.join(base_dir, "clip", "*.tif"))

    nodata = 0
    for tif in list_tif:
        arr, metadata = read_tif(tif)
        rows, cols = arr.shape
        ref_arr = arr.reshape(-1) / 1e4

        non_zeor_index = np.where(ref_arr > 0)[0]
        non_zeor_arr = ref_arr[ref_arr > nodata]
        norm_arr = (non_zeor_arr - np.min(non_zeor_arr)) / (
            np.max(non_zeor_arr) - np.min(non_zeor_arr)
        )
        ref_arr[non_zeor_index] = norm_arr
        write_tif(ref_arr.reshape(rows, cols), metadata, tif.replace("clip", "norm"))
        print("tif")


def norm_l1_img(list_tif):
    nodata = 0
    nodata_rgb = 66356
    nodata_l2a = 2147483647

    for tif in list_tif:
        preprocessed_file_path = tif.replace(r"clip", r"preprocessed_clip")
        if not os.path.exists(preprocessed_file_path):
            print(tif)
            arr, metadata = read_tif(tif)

            arr = np.nan_to_num(arr, nan=nodata)
            _max = arr[arr != nodata].max()
            _min = arr[arr != nodata].min()
            arr = (arr - _min) / (_max - _min)

            write_tif(arr, metadata, preprocessed_file_path)
