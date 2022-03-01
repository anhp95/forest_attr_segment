#%%

import numpy as np
import math
import glob
import os

from utils.tif import read_tif, write_tif
from collections import Counter

ORG_IMG_DIR = r"D:\co2_data\DL\large_img\sentinel\preprocessing\10m"
LARGE_TIFS = glob.glob(os.path.join(ORG_IMG_DIR, "*.tif"))


class Forest:

    _S1S2_YEAR_BAND = [tif_.split("\\")[-1][:-4] for tif_ in LARGE_TIFS]
    _SIZE = 32

    _TIME_SERIES = [
        "brown",
        "green",
        "yellow",
    ]

    _S1_BANDS = ["VV", "VH"]
    _S2_BANDS = [
        "B2",
        "B3",
        "B4",
        "B5",
        "B6",
        "B7",
        "B8",
        "BR4",
        "B11",
        "B12",
        "NDVI",
    ]

    def __init__(self, sentinel, spec, age, timber):

        self.sentinel = sentinel
        self.spec = spec
        self.age = age
        self.timber = timber

        self.major_spec = 0
        self.major_age = 0
        self.major_timber = 0

        self._relabel_spec()
        # self._relabel_age()
        self._set_main_label()

    def _relabel_spec(self):
        """
        beech -> broadleaf, birch -> broadleaf, larch -> conifer, fir -> conifer
        1-> 3, 8 -> 3, 4 -> 5, 6 -> 5
        2->0, sugi
        3->1, broadleaf
        5->2, conifer
        7->3, cypress
        """
        self.spec[self.spec == 1] = 3
        self.spec[self.spec == 8] = 3

        self.spec[self.spec == 4] = 5
        self.spec[self.spec == 6] = 5

        self.spec[self.spec == 2] = 0
        self.spec[self.spec == 3] = 1
        self.spec[self.spec == 5] = 2
        self.spec[self.spec == 7] = 3

    def _relabel_age(self):
        """
        0 - young age, 1 - mature age, 2 - harvesting age
        """
        self.age[self.age == 65535] = 0
        self.age[self.age <= 20] = 0
        self.age[(self.age > 20) & (self.age < 50)] = 1
        self.age[self.age >= 50] = 2

    def _set_main_label(self):

        # set_main_spec
        (unique, counts) = np.unique(self.spec, return_counts=True)
        spec_count = {c: u for u, c in zip(unique, counts)}
        max_count = np.sort(counts)[-1]
        self.major_spec = spec_count[max_count]

        # set_main_age
        (unique, counts) = np.unique(self.age, return_counts=True)
        age_count = {c: u for u, c in zip(unique, counts)}
        max_count = np.sort(counts)[-1]
        self.major_age = age_count[max_count]

    @staticmethod
    def fill_missing_data(arr, missing_value_):
        _WINDOW_SIZE = 9
        _OFFSET = 1

        (org_row, org_col) = arr.shape

        (missing_index_rows, missing_index_cols) = np.where(arr == missing_value_)
        if len(missing_index_rows) > 0:
            for row, col in zip(missing_index_rows, missing_index_cols):
                r_w = max(row - _OFFSET, 0)
                r_w_n = min(row + _WINDOW_SIZE, org_row)

                c_w = max(col - _OFFSET, 0)
                c_w_n = min(col + _WINDOW_SIZE, org_col)

                window = arr[r_w:r_w_n, c_w:c_w_n]
                avg_window = window[window != missing_value_]
                if len(avg_window) > 0:
                    arr[row, col] = window[window != missing_value_].mean()
                else:
                    arr[row, col] = 0

        return arr

    @staticmethod
    def create_mask_from_specs(arr):
        arr[arr == 255] = 0
        arr[arr > 1] = 1

        return arr

    @classmethod
    def crop_index_training(cls, arr):
        (rows, cols) = arr.shape
        nrows = int(rows / cls._SIZE)
        ncols = int(cols / cls._SIZE)

        list_row = [(r * cls._SIZE, (r + 1) * cls._SIZE) for r in range(nrows)]
        list_col = [(c * cls._SIZE, (c + 1) * cls._SIZE) for c in range(ncols)]

        list_index = []
        for (r, r_n) in list_row:
            for (c, c_n) in list_col:
                small_img = Forest.create_mask_from_specs(arr[r:r_n, c:c_n])
                count = np.count_nonzero(small_img)
                if count == math.pow(cls._SIZE, 2):
                    list_index.append(((r, r_n), (c, c_n)))

        return list_index

    @classmethod
    def crop_index_infer(cls, tif):
        arr, _ = read_tif(tif)

        (rows, cols) = arr.shape
        nrows = int(rows / cls._SIZE)
        ncols = int(cols / cls._SIZE)

        list_row = [(r * cls._SIZE, (r + 1) * cls._SIZE) for r in range(nrows)]
        list_col = [(c * cls._SIZE, (c + 1) * cls._SIZE) for c in range(ncols)]

        last_row = (nrows - cls._SIZE, nrows)
        last_col = (ncols - cls._SIZE, ncols)

        list_row.append(last_row)
        list_col.append(last_col)

        list_index = []
        for (r, r_n) in list_row:
            for (c, c_n) in list_col:
                list_index.append(((r, r_n), (c, c_n)))
        return list_index

    @staticmethod
    def crop_image(l_tif, l_index, fill_missing=True, crop_mode="training"):

        _NODATA = -9999

        if crop_mode == "infer":
            # iterate the l_index, using same index for all the images
            l_index = [l_index for _ in range(0, len(l_tif))]
        elif crop_mode == "training":
            # just to clarify each training image has its own index
            l_index = l_index

        crop_array = []
        for tif, index_per_tif in zip(l_tif, l_index):
            arr = read_tif(tif)[0]
            for ((r, r_n), (c, c_n)) in index_per_tif:
                splitted = arr[r:r_n, c:c_n]
                if fill_missing:
                    splitted = Forest.fill_missing_data(
                        splitted, missing_value_=_NODATA
                    )
                crop_array.append(splitted)
        return crop_array

    @classmethod
    def stich_image(cls, region, npy_file, arr_band=0):

        _FOREST_MAP = f"data/forest_map/{region}_forest_map.tif"

        l_index = Forest.crop_index_infer(_FOREST_MAP)
        forest_mask, _ = read_tif(_FOREST_MAP)
        spec_arr = np.load(npy_file)

        (rows, cols) = forest_mask.shape
        zero_arr = np.zeros((rows, cols))

        for ((r, r_n), (c, c_n)), spec_arr_idx in zip(l_index, spec_arr):
            zero_arr[r:r_n, c:c_n] = spec_arr_idx[:, arr_band].reshape(
                cls._SIZE, cls._SIZE
            )
        final_arr = forest_mask * (zero_arr + 1)
        return final_arr

    @staticmethod
    def agg_infered_stitch(region, out_tif, *preds, forest_attr="spec"):

        _FOREST_MAP = f"data/forest_map/{region}_forest_map.tif"

        forest_mask, metadata = read_tif(_FOREST_MAP)
        (rows, cols) = forest_mask.shape
        stacked_ts_arr = np.stack(preds, axis=-1).reshape(rows * cols, -1)
        final_arr = [Counter(ts_arr).most_common(1)[0][0] for ts_arr in stacked_ts_arr]
        final_arr = np.array(final_arr).reshape(rows, cols)

        if forest_attr == "spec":

            print("area of sugi:", np.count_nonzero(final_arr[final_arr == 1]) / 100)
            print("area of BF:", np.count_nonzero(final_arr[final_arr == 2]) / 100)
            print("area of C:", np.count_nonzero(final_arr[final_arr == 3]) / 100)
            print("area of cypress:", np.count_nonzero(final_arr[final_arr == 4]) / 100)
        elif forest_attr == "age":
            print(
                "area of young forest:",
                np.count_nonzero(final_arr[final_arr == 1]) / 100,
            )
            print(
                "area of mature forest:",
                np.count_nonzero(final_arr[final_arr == 2]) / 100,
            )
            print(
                "area of harvesting age:",
                np.count_nonzero(final_arr[final_arr == 3]) / 100,
            )

        write_tif(
            final_arr,
            metadata=metadata,
            filename=out_tif,
        )

    @classmethod
    def stack_crop_sentinel(
        cls, s1s2_tif, l_index, cnn_mode="3d", crop_mode="training"
    ):

        valid_s12yb_arr = []
        if cnn_mode == "2d":
            for s12yb in cls._S1S2_YEAR_BAND:
                print(s12yb)
                s12yb_tif = [s1s2 for s1s2 in s1s2_tif if s12yb in s1s2]
                valid_s12yb_arr.append(
                    Forest.crop_image(
                        s12yb_tif, l_index, fill_missing=True, crop_mode=crop_mode
                    )
                )
            s1s2 = np.stack(valid_s12yb_arr, axis=3)
        elif cnn_mode == "3d":

            s1_sat = "s1"
            s2_sat = "s2"
            dem = "elevation"
            time_series = cls._TIME_SERIES

            s1_bands = cls._S1_BANDS
            s2_bands = cls._S2_BANDS

            bands_time_series = []
            for bands1 in s1_bands:
                bands_time_series.append(
                    [f"{s1_sat}_{time}_{bands1}" for time in time_series]
                )
            for bands2 in s2_bands:
                bands_time_series.append(
                    [f"{s2_sat}_{time}_{bands2}" for time in time_series]
                )
            arr = []
            for band in bands_time_series:
                band_ts = []
                for ts in band:
                    tifs = [ts_tif for ts_tif in s1s2_tif if ts in ts_tif]
                    band_ts.append(
                        Forest.crop_image(
                            tifs, l_index, fill_missing=True, crop_mode=crop_mode
                        )
                    )
                dem_tifs = [dem_tif for dem_tif in s1s2_tif if dem in dem_tif]
                band_ts.append(
                    Forest.crop_image(
                        dem_tifs, l_index, fill_missing=True, crop_mode=crop_mode
                    )
                )
                arr.append(np.stack(band_ts, axis=1))
                print(band)

            s1s2 = np.stack(arr, axis=1)
            print(s1s2.shape)

        return s1s2

    @classmethod
    def crop_label_img(cls, l_spec_tif, l_age_tif, l_timber_tif, l_index):

        spec = Forest.crop_image(l_spec_tif, l_index, fill_missing=True)
        age = Forest.crop_image(l_age_tif, l_index, fill_missing=True)
        timber = Forest.crop_image(l_timber_tif, l_index, fill_missing=True)

        return spec, age, timber

    @classmethod
    def gen_training_obj_from_tif(
        cls, s1s2_tif, l_spec_tif, l_age_tif, l_timber_tif, cnn_mode="2d"
    ):
        l_index = [Forest.crop_index_training(read_tif(tif)[0]) for tif in l_spec_tif]

        s1s2 = Forest.stack_crop_sentinel(
            s1s2_tif, l_index, cnn_mode=cnn_mode, crop_mode="training"
        )
        spec, age, timber = Forest.crop_label_img(
            l_spec_tif, l_age_tif, l_timber_tif, l_index
        )
        return [Forest(s, sp, ag, ti) for s, sp, ag, ti in zip(s1s2, spec, age, timber)]

    @classmethod
    def gen_infer_obj_from_tif(cls, s1s2_tif, cnn_mode="3d"):

        l_index = Forest.crop_index_infer(s1s2_tif[0])
        s1s2 = Forest.stack_crop_sentinel(
            s1s2_tif, l_index, cnn_mode=cnn_mode, crop_mode="infer"
        )

        return s1s2

    def stack_s1s2_spec(self):

        return np.dstack((self.sentinel, self.spec.reshape(self._SIZE, self._SIZE, 1)))

    def stack_s1s2_age(self):
        return np.dstack(
            (
                self.sentinel,
                self.spec.reshape(self._SIZE, self._SIZE, 1),
                self.age.reshape(self._SIZE, self._SIZE, 1),
            )
        )


class TreeSpecies(Forest):

    _TRAIN_DIR = "../data/data_train/data_spec_40d_32x32/"
    _IMG_FOLDER = "image/"
    _MASK_FOLDER = "mask/"
    _TRAIN_FOLDER = "train/"
    _VAL_FOLDER = "val/"

    _VAL_RATIO = 0.05  #

    def __init__(self, list_forest_obj):

        self.train = None
        self.val = None

        self._split_train_test(list_forest_obj)
        # self._save_to_npy()

    def _split_train_test(self, list_forest_obj):

        unique_specs = np.unique([fr_obj.major_spec for fr_obj in list_forest_obj])

        object_by_spec = {
            unique_spec: (
                [
                    forest_obj
                    for forest_obj in list_forest_obj
                    if forest_obj.major_spec == unique_spec
                ]
            )
            for unique_spec in unique_specs
        }
        val_set = [
            object_by_spec[key][
                0 : int(round(self._VAL_RATIO * len(object_by_spec[key])))
            ]
            for key in object_by_spec.keys()
        ]
        train_set = [
            object_by_spec[key][
                int(round(self._VAL_RATIO * len(object_by_spec[key]))) :
            ]
            for key in object_by_spec.keys()
        ]

        self.train = [obj for set_ in train_set for obj in set_]
        self.val = [obj for set_ in val_set for obj in set_]

    def _save_to_npy(self):
        def _save(type_):

            print(type_)
            if type_ == self._TRAIN_FOLDER:
                list_fr_obj = self.train
            else:
                list_fr_obj = self.val

            image_dir = os.path.join(self._TRAIN_DIR, type_, self._IMG_FOLDER)
            mask_dir = os.path.join(self._TRAIN_DIR, type_, self._MASK_FOLDER)

            if not os.path.isdir(image_dir):
                os.makedirs(image_dir)
            if not os.path.isdir(mask_dir):
                os.makedirs(mask_dir)

            for idx, forest_obj in enumerate(list_fr_obj):

                np.save(
                    os.path.join(
                        image_dir,
                        f"{idx}.npy",
                    ),
                    forest_obj.sentinel,
                    # forest_obj.stack_s1s2_spec
                    # forest_obj.stack_s1s2_age(),
                )
                np.save(
                    os.path.join(
                        mask_dir,
                        f"{idx}.npy",
                    ),
                    forest_obj.spec,
                    # forest_obj.age
                    # forest_obj.timber,
                )

        _save(type_=self._TRAIN_FOLDER)
        _save(type_=self._VAL_FOLDER)


class TreeAge(Forest):
    _TRAIN_DIR = "../data/data_train/data_age_14d_32x32/"
    _IMG_FOLDER = "image/"
    _MASK_FOLDER = "mask/"
    _TRAIN_FOLDER = "train/"
    _VAL_FOLDER = "val/"

    _VAL_RATIO = 0.05  #

    _VAL_RATIO_2 = 0.02
    _TRAIN_SAMPLE_2 = 4000

    def __init__(self, list_forest_obj):

        self.train = None
        self.val = None

        self._split_train_test(list_forest_obj)
        # self._save_to_npy()

    def _split_train_test(self, list_forest_obj):

        unique_ages = np.unique([fr_obj.major_age for fr_obj in list_forest_obj])

        object_by_age = {
            unique_age: (
                [
                    forest_obj
                    for forest_obj in list_forest_obj
                    if forest_obj.major_age == unique_age
                ]
            )
            for unique_age in unique_ages
        }
        val_set = []
        train_set = []
        for key in object_by_age.keys():
            if key != 2:
                val_set.append(
                    object_by_age[key][
                        0 : int(round(self._VAL_RATIO * len(object_by_age[key])))
                    ]
                )
                train_set.append(
                    object_by_age[key][
                        int(round(self._VAL_RATIO * len(object_by_age[key]))) :
                    ]
                )
            elif key == 2:
                val_set.append(
                    object_by_age[key][
                        0 : int(round(self._VAL_RATIO_2 * len(object_by_age[key])))
                    ]
                )
                train_set.append(
                    object_by_age[key][
                        int(
                            round(self._VAL_RATIO_2 * len(object_by_age[key]))
                        ) : self._TRAIN_SAMPLE_2
                        + int(round(self._VAL_RATIO_2 * len(object_by_age[key])))
                    ]
                )

        self.train = [obj for set_ in train_set for obj in set_]
        self.val = [obj for set_ in val_set for obj in set_]

    def _save_to_npy(self):
        def _save(type_):

            print(type_)
            if type_ == self._TRAIN_FOLDER:
                list_fr_obj = self.train
            else:
                list_fr_obj = self.val

            image_dir = os.path.join(self._TRAIN_DIR, type_, self._IMG_FOLDER)
            mask_dir = os.path.join(self._TRAIN_DIR, type_, self._MASK_FOLDER)

            if not os.path.isdir(image_dir):
                os.makedirs(image_dir)
            if not os.path.isdir(mask_dir):
                os.makedirs(mask_dir)

            for idx, forest_obj in enumerate(list_fr_obj):

                np.save(
                    os.path.join(
                        image_dir,
                        f"{idx}.npy",
                    ),
                    forest_obj.sentinel,
                )
                np.save(
                    os.path.join(
                        mask_dir,
                        f"{idx}.npy",
                    ),
                    forest_obj.age,
                )

        _save(type_=self._TRAIN_FOLDER)
        _save(type_=self._VAL_FOLDER)


def gen_training_set(small_dir, res):

    s1s2_tifs = glob.glob(os.path.join(small_dir, "sentinel", res, "*.tif"))
    spec_tifs = glob.glob(os.path.join(small_dir, "spec", res, "*.tif"))
    age_tifs = glob.glob(os.path.join(small_dir, "age", res, "*.tif"))
    timber_tifs = glob.glob(os.path.join(small_dir, "timber", res, "*.tif"))

    list_obj_forest = Forest.gen_training_obj_from_tif(
        s1s2_tifs, spec_tifs, age_tifs, timber_tifs, cnn_mode="2d"
    )
    print("--- generating training set for species segmentation ---")
    species = TreeSpecies(list_obj_forest)
    print("--- generating training set for age segmentation ---")
    age = TreeAge(list_obj_forest)
    return species, age


def get_trainVal_stats(species_obj, age_obj):

    spec_train = [obj.major_spec for obj in species_obj.train]
    unique, counts = np.unique(spec_train, return_counts=True)
    spec_count_train = {c: u for u, c in zip(unique, counts)}

    spec_val = [obj.major_spec for obj in species_obj.val]
    unique, counts = np.unique(spec_val, return_counts=True)
    spec_count_val = {c: u for u, c in zip(unique, counts)}

    age_train = [obj.major_age for obj in age_obj.train]
    unique, counts = np.unique(age_train, return_counts=True)
    age_count_train = {c: u for u, c in zip(unique, counts)}

    age_val = [obj.major_age for obj in age_obj.val]
    unique, counts = np.unique(age_val, return_counts=True)
    age_count_val = {c: u for u, c in zip(unique, counts)}

    return spec_count_train, spec_count_val, age_count_train, age_count_val


def gen_infer_set(img_dir, region):

    img_dir = os.path.join(img_dir, region)
    out_file = f"data/data_infer/input/{region}_13b.npy"

    list_tif = glob.glob(os.path.join(img_dir, "*.tif"))
    infer_input = Forest.gen_infer_obj_from_tif(list_tif, cnn_mode="3d")

    np.save(out_file, infer_input)


def gen_predicted_map(region, npy_file, out_tif, forest_attr="spec"):

    pb1 = Forest.stich_image(region, npy_file, 0)
    pb2 = Forest.stich_image(region, npy_file, 1)
    pb3 = Forest.stich_image(region, npy_file, 2)

    if forest_attr == "spec":
        pb4 = Forest.stich_image(region, npy_file, 3)
        Forest.agg_infered_stitch(
            region, out_tif, pb1, pb2, pb3, pb4, forest_attr=forest_attr
        )
    elif forest_attr == "age":
        Forest.agg_infered_stitch(
            region, out_tif, pb1, pb2, pb3, forest_attr=forest_attr
        )


def sample_run():
    """Generate training dataset"""
    SMALL_DIR = r"D:\co2_data\DL\small_img"
    RES = "10m"
    species_obj, age_obj = gen_training_set(SMALL_DIR, RES)

    """Check number of samples for training and validation"""
    (
        spec_count_train,
        spec_count_val,
        age_count_train,
        age_count_val,
    ) = get_trainVal_stats(species_obj, age_obj)

    """Generate input to inference"""
    REGION = "ena"
    IMG_DIR = r"D:\co2_data\DL\large_img\sentinel\preprocessing"
    gen_infer_set(IMG_DIR, REGION, forest_attr="spec")

    """Generate predicted spec/age map"""
    CASE = "3d_aspp_enc_bot_dec_7780_2"

    gen_predicted_map(IMG_DIR, REGION, CASE, forest_attr="spec")
    gen_predicted_map(IMG_DIR, REGION, CASE, forest_attr="age")
