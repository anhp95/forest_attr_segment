#%%
import numpy as np
import matplotlib.pyplot as plt

from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans

from utils.tif import read_tif, write_tif

# %%


def stats(x, bins, label):

    _ = plt.hist(x, bins=bins, label=label)
    plt.legend()


class ReClassification:
    def __init__(self, input_low_res, input_s2_tifs, n_clusters, out_tif_recls):
        self.input_low_res = input_low_res
        self.input_s2_tifs = input_s2_tifs
        self.n_clusters = n_clusters
        self.out_tif_recls = out_tif_recls

    def extract_bow(self, spec_arr, label_val):
        index = np.where(spec_arr == label_val)[0]

        list_arr = []
        for tif in self.input_s2_tifs:
            tif_arr = read_tif(tif)[0]
            tif_arr = tif_arr.reshape(-1)
            tif_arr = tif_arr[index]
            list_arr.append(tif_arr)

        stacked_arr = np.stack((list_arr), axis=-1)

        model = KMeans(n_clusters=self.n_clusters)

        pred = model.fit_predict(stacked_arr)
        valid_len = len(pred) / self.n_clusters
        bow = model.cluster_centers_

        list_valid_means = [
            bow[i] for i in range(self.n_clusters) if np.sum(pred == i) >= valid_len
        ]

        X_train = np.vstack(list_valid_means)
        y_train = np.zeros(len(list_valid_means)) + label_val

        X_pred = np.vstack([bow[i] for i in pred])

        return X_train, y_train, X_pred, index

    def reclassify(self):

        lr_arr, metadata = read_tif(self.input_low_res)
        rows, cols = lr_arr.shape
        lr_arr = lr_arr.reshape(-1)

        X_sugi, y_sugi, sugi_pred, sugi_index = self.extract_bow(lr_arr, label_val=1)
        X_hino, y_hino, hino_pred, hino_index = self.extract_bow(lr_arr, label_val=4)
        X_cf, y_cf, cf_pred, cf_index = self.extract_bow(lr_arr, label_val=3)
        X_bf, y_bf, bf_pred, bf_index = self.extract_bow(lr_arr, label_val=2)

        X = np.vstack((X_sugi, X_bf, X_cf, X_hino))
        y = np.hstack((y_sugi, y_bf, y_cf, y_hino))

        model = GaussianNB()
        model.fit(X, y)

        lr_arr[sugi_index] = model.predict(sugi_pred)
        lr_arr[hino_index] = model.predict(hino_pred)
        lr_arr[bf_index] = model.predict(bf_pred)
        lr_arr[cf_index] = model.predict(cf_pred)

        print("write high-res tif")

        print("area of sugi:", np.count_nonzero(lr_arr[lr_arr == 1]) / 100)
        print("area of BF:", np.count_nonzero(lr_arr[lr_arr == 2]) / 100)
        print("area of C:", np.count_nonzero(lr_arr[lr_arr == 3]) / 100)
        print("area of cypress:", np.count_nonzero(lr_arr[lr_arr == 4]) / 100)

        lr_arr[lr_arr < 0] = 0

        write_tif(lr_arr.reshape(rows, cols), metadata, self.out_tif_recls)


# %%
# spec_arr = read_tif(out_tif_recls)[0]
# spec_arr = spec_arr.reshape(-1)
# sugi_index = np.where(spec_arr == 1)[0]
# bf_index = np.where(spec_arr == 2)[0]
# cf_index = np.where(spec_arr == 3)[0]
# hinoki_index = np.where(spec_arr == 4)[0]

# for tif in [input_s2_tifs[5]]:
#     tif_arr = read_tif(tif)[0]
#     tif_arr = tif_arr.reshape(-1)
#     stats(tif_arr[hinoki_index], 500, "hinoki")
#     stats(tif_arr[cf_index], 500, "cf")
#     stats(tif_arr[sugi_index], 500, "sugi")
#     stats(tif_arr[bf_index], 500, "bf")
