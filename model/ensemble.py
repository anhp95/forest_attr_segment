#%%
import os
import glob
import numpy as np

import xgboost
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

DATA_DIR = "../data/data_train"


def gen_train_val(input_shape, forest_attr, sub_folder):
    train_dir = os.path.join(
        DATA_DIR,
        f"data_{forest_attr}_{input_shape[0]}d_{input_shape[1]}x{input_shape[2]}",
    )
    img_npy = glob.glob(os.path.join(train_dir, f"{sub_folder}/image/*.npy"))
    mask_npy = glob.glob(os.path.join(train_dir, f"{sub_folder}/mask/*.npy"))

    img_arr = []
    mask_arr = []
    for img, mask in zip(img_npy, mask_npy):

        img_arr.append(np.load(img).reshape(input_shape[1] * input_shape[2], -1))
        mask_arr.append(np.load(mask).reshape(input_shape[1] * input_shape[2], -1))

    return np.vstack((img_arr)), np.vstack((mask_arr)).ravel()


def train(X_train, y_train, class_weight):
    # estimators = [
    #     ("rf", RandomForestClassifier()),
    #     ("svm", SVC()),
    #     ("xgb", xgboost.XGBClassifier()),
    # ]

    # ensemble_cls = VotingClassifier(estimators=estimators, voting="hard")
    # ensemble_cls.fit(X_train, y_train)
    RF = RandomForestClassifier(class_weight=class_weight)
    RF.fit(X_train, y_train)
    return RF


def main(input_shape, forest_attr, class_weight):
    X_train, y_train = gen_train_val(input_shape, forest_attr, sub_folder="train")
    X_val, y_val = gen_train_val(input_shape, forest_attr, sub_folder="val")

    model = train(X_train, y_train, class_weight)

    y_pred = model.predict(X_val)
    print(f"OA: {accuracy_score(y_val, y_pred)}")
    # print(f"f1: {f1_score(y_val, y_pred)}")
    # print(f"kappa: {cohen_kappa_score(y_val, y_pred)}")


if __name__ == "__main__":

    forest_attr = "age"
    input_shape = (14, 32, 32)

    if forest_attr == "age":
        class_weight = {0: 1, 1: 0.1, 2: 0.05}
    elif forest_attr == "spec":
        class_weight = {0: 1, 1: 0.153, 2: 0.252, 3: 0.241}

    print(f"{forest_attr}: {input_shape} - {class_weight}")
    main(input_shape, forest_attr, class_weight)

# %%
