#%%

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.abspath(".."))
from utils.tif import read_tif


def stats(x, bins, label):
    _ = plt.hist(x, bins=bins, label=label)
    plt.legend()


prv_spec_tif = r"D:\Takejima-sensei\ena_private_forest\2018_all\tif\wgs84_prv_spec.tif"
pca_s2l2a = r"D:\co2_data\DL\large_img\sentinel\S2_RAW\preprocessed_s2\preprocessed_clip\PCA\ena_prv.tif"

prv_spec_arr = read_tif(prv_spec_tif)[0]
prv_spec_arr = prv_spec_arr.reshape(-1)

pca_s2l2a = read_tif(pca_s2l2a)[0]
pca_s2l2a = pca_s2l2a.reshape(-1)

sugi_index = np.where(prv_spec_arr[prv_spec_arr == 1])[0]
hinoki_index = np.where(prv_spec_arr[prv_spec_arr == 2])[0]
pine_index = np.where(prv_spec_arr[prv_spec_arr == 3])[0]
cf_index = np.where(prv_spec_arr[prv_spec_arr == 4])[0]
bf_index = np.where(prv_spec_arr[prv_spec_arr == 5])[0]

sugi_arr = pca_s2l2a[sugi_index]
hinoki_arr = pca_s2l2a[hinoki_index]
pine_arr = pca_s2l2a[pine_index]
cf_arr = pca_s2l2a[cf_index]
bf_arr = pca_s2l2a[bf_index]


# stats(hinoki_arr[hinoki_arr > 0], "hinoki_arr")
# stats(bf_arr[bf_arr > 0], "bf_arr")
# stats(sugi_arr[sugi_arr > 0], "sugi_arr")
# stats(pine_arr[pine_arr > 0], "pine_arr")
# stats(cf_arr[cf_arr > 0], 200, "cf_arr")
