#%%

import pandas as pd
from simpledbf import Dbf5
from deep_translator import GoogleTranslator


TREE_SPEC = {
    "Cypress": "Cypress",
    "Sugi": "Sugi",
    "Beech": "BF",
    "Betula grossa": "BF",
    "Black pine": "C",
    "Camba": "BF",
    "Fir": "Fir",
    "Heavenly cypress": "Cypress",
    "Himekomatsu": "C",
    "Larch": "C",
    "Maple": "BF",
    "Other L": "BF",
    "Other N": "C",
    "Quercus crispula": "BF",
    "Quercus serrata": "BF",
    "Red pine": "Pine",
    "Spanish mackerel": "C",
    "Tsuga": "C",
    "Tsuga diversifolia": "C",
    "Zelkova": "BF",
    "Horse chestnut": "BF",
    "Chestnut": "BF",
    "Take": "Bamboo",
    "Other unstanding trees": "Others",
    "Bamboo dough": "Bamboo",
    "Oak": "BF",
}


def handleInvalidText(text):
    translator = GoogleTranslator(source="ja", target="en")

    try:
        text = translator.translate(text)
        return text
    except Exception:
        return text


def spec_to_label(spec):
    if spec == "Sugi":
        return 1
    elif spec == "Cypress":
        return 2
    elif spec == "Pine":
        return 3
    elif spec == "C":
        return 4
    elif spec == "BF":
        return 5
    return 6


def translate_dbf(dbf_file):
    df = Dbf5(dbf_file, codec="shift-jis").to_dataframe()
    org_spec = df.SPEC1.tolist()
    unique_spec = list(set(org_spec))

    trans_unique_spec = [handleInvalidText(spec) for spec in unique_spec]
    trans_dict = {key: val for key, val in zip(unique_spec, trans_unique_spec)}
    refine_spec = [
        trans_dict[spec]
        if trans_dict[spec] not in TREE_SPEC
        else TREE_SPEC[trans_dict[spec]]
        for spec in org_spec
    ]

    label_spec = [spec_to_label(spec) for spec in refine_spec]
    new_df = pd.DataFrame()
    new_df["spec"] = refine_spec
    new_df["label"] = label_spec
    new_df.to_csv(dbf_file.replace(".dbf", ".csv"))


#%%
# if __name__ == '__main__':
# dbf_file = r"D:\Takejima-sensei\ena_private_forest\2018_all\backup\2018_all.dbf"
# translate_dbf(dbf_file)
