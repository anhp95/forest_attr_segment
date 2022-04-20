import glob
import os


def get_path_train(img_shape, forest_attr="spec", backbone="3d"):

    train_dir = "data/data_train"
    if "3d" in backbone:
        folder = f"data_{forest_attr}_{img_shape[0]}b_{img_shape[1]}d_{img_shape[2]}x{img_shape[2]}"
    elif "2d" in backbone:
        folder = f"data_{forest_attr}_{img_shape[0]}d_{img_shape[1]}x{img_shape[2]}"

    # Configure your path to training set here
    train_img_dir = f"{train_dir}/{folder}/train/image/"
    train_mask_dir = f"{train_dir}/{folder}/train/mask/"
    val_img_dir = f"{train_dir}/{folder}/val/image/"
    val_mask_dir = f"{train_dir}/{folder}/val/mask/"

    return train_img_dir, train_mask_dir, val_img_dir, val_mask_dir


def get_path_infer(infer_obj, args):
    region = infer_obj.region
    forest_attr = infer_obj.forest_attr
    acc = infer_obj.acc
    backbone = infer_obj.backbone
    n_clusters = args.n_clusters

    # Configure your path to reclassion dataset here
    s2_img_dir = f"D:/co2_data/DL/large_img/sentinel/s2_{region}_recls/l2"

    map_dir = f"data/{forest_attr}_map"

    low_res_dir = os.path.join(map_dir, region, "low_res")
    high_res_dir = os.path.join(map_dir, region, "high_res")

    if not os.path.isdir(low_res_dir):
        os.makedirs(low_res_dir)

    if not os.path.isdir(high_res_dir):
        os.makedirs(high_res_dir)

    low_res_name = f"{region}_{forest_attr}_{backbone}_{acc}"

    low_res_tif = os.path.join(low_res_dir, f"{low_res_name}.tif")
    high_res_tif = os.path.join(high_res_dir, f"{low_res_name}_km{n_clusters}.tif")

    input_s2_tifs = glob.glob(os.path.join(s2_img_dir, "*.tif"))

    return low_res_tif, input_s2_tifs, high_res_tif
