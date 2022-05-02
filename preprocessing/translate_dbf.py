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


class MainForestAttr:
    def __init__(self, s1, s2, s3, ag1, ag2, ag3, r1, r2, r3):
        self.s1 = s1
        self.s2 = s2
        self.s3 = s3
        self.ag1 = ag1
        self.ag2 = ag2
        self.ag3 = ag3
        self.r1 = r1
        self.r2 = r2
        self.r3 = r3

        self.main_spec = []
        self.main_age = []

        self.trans_spec = []
        self.cls_age = []
        self.cls_spec = []

        self._get_main_attr()
        self._do_cls_age()

        self._do_trans_spec()
        self._do_cls_spec()

    def _get_main_attr(self):
        agg_ratio = [[r1, r2, r3] for r1, r2, r3 in zip(self.r1, self.r2, self.r3)]
        agg_spec = [[s1, s2, s3] for s1, s2, s3 in zip(self.s1, self.s2, self.s3)]
        agg_age = [
            [ag1, ag2, ag3] for ag1, ag2, ag3 in zip(self.ag1, self.ag2, self.ag3)
        ]
        for spec, age, ratio in zip(
            agg_spec,
            agg_age,
            agg_ratio,
        ):
            main_ratio = max(ratio)
            main_ratio_index = ratio.index(main_ratio)
            self.main_age.append(age[main_ratio_index])
            self.main_spec.append(spec[main_ratio_index])

    def _do_cls_age(self):
        for age in self.main_age:
            if age <= 20:
                cls_age = 1
            elif age > 21 and age < 50:
                cls_age = 2
            elif age >= 50:
                cls_age = 3
            self.cls_age.append(cls_age)

    def _do_trans_spec(self):
        unique_spec = list(set(self.main_spec))

        trans_unique_spec = [handleInvalidText(spec) for spec in unique_spec]
        trans_dict = {key: val for key, val in zip(unique_spec, trans_unique_spec)}
        self.trans_spec = [
            trans_dict[spec]
            if trans_dict[spec] not in TREE_SPEC
            else TREE_SPEC[trans_dict[spec]]
            for spec in self.main_spec
        ]

    def _do_cls_spec(self):
        for spec in self.trans_spec:
            cls_spec = 0
            if spec == "Sugi":
                cls_spec = 1
            elif spec == "BF":
                cls_spec = 2
            elif spec == "Pine" or spec == "C":
                cls_spec = 3
            elif spec == "Cypress":
                cls_spec = 4

            self.cls_spec.append(cls_spec)

    def gen_from_df(df):
        return MainForestAttr(
            df.SPEC1.tolist(),
            df.SPEC2.tolist(),
            df.SPEC3.tolist(),
            df.AGE1.tolist(),
            df.AGE2.tolist(),
            df.AGE3.tolist(),
            df.RATIO1.tolist(),
            df.RATIO2.tolist(),
            df.RATIO3.tolist(),
        )

    def to_df(self):
        df = pd.DataFrame()

        df["main_spec"] = self.main_spec
        df["main_age"] = self.main_age

        df["trans_spec"] = self.trans_spec
        df["cls_age"] = self.cls_age
        df["cls_spec"] = self.cls_spec

        return df


def main(dbf_file):
    df = Dbf5(dbf_file, codec="shift-jis").to_dataframe()
    main_forest_attr = MainForestAttr.gen_from_df(df)
    df_MFA = main_forest_attr.to_df()
    return df_MFA


# %%
if __name__ == "__main__":

    dbf_file = r"D:\Takejima-sensei\ena_private_forest\all_en\2018_all.dbf"

    df = main(dbf_file)
    csv_file = dbf_file.replace("dbf", "csv")
    print("")
    df.to_csv(csv_file)

# %%
