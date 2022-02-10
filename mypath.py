def get_path(img_shape, forest_attr="spec", backbone="3d"):

    train_dir = "data/data_train"
    if "3d" in backbone:
        folder = f"data_{forest_attr}_{img_shape[0]}b_{img_shape[1]}d_{img_shape[2]}x{img_shape[2]}"
    elif "2d" in backbone:
        folder = f"data_{forest_attr}_{img_shape[0]}d_{img_shape[1]}x{img_shape[2]}"

    train_img_dir = f"{train_dir}/{folder}/train/image/"
    train_mask_dir = f"{train_dir}/{folder}/train/mask/"
    val_img_dir = f"{train_dir}/{folder}/val/image/"
    val_mask_dir = f"{train_dir}/{folder}/val/mask/"

    return train_img_dir, train_mask_dir, val_img_dir, val_mask_dir
