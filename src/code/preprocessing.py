import argparse
import os
import requests
import tempfile

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# Since we get a headerless CSV file, we specify the column names here.
feature_columns_names = [
    "gt_comp_dis_pressure",
    "gt_exhaust_pressure",
    "gt_inlet_temp",
    "gt_air_filter_diff_pressure",
    "solar_radiation",
    "solar_energy",
    "uv_index",
    "gt_ambient_pressure",
    "cloud_cover",
    "sea_level_pressure"
]
label_column = "gt_energy_yield"

feature_columns_dtype = {
    "gt_comp_dis_pressure": np.float64,
    "gt_exhaust_pressure": np.float64,
    "gt_inlet_temp": np.float64,
    "gt_air_filter_diff_pressure": np.float64,
    "solar_radiation": np.float64,
    "solar_energy": np.float64,
    "uv_index": np.float64,
    "gt_ambient_pressure": np.float64,
    "cloud_cover": np.float64,
    "sea_level_pressure": np.float64
}
label_column_dtype = {"gt_energy_yield": np.float64}


def merge_two_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z


if __name__ == "__main__":
    base_dir = "/opt/ml/processing"

    df = pd.read_csv(
        f"{base_dir}/input/abalone-dataset.csv",
        header=None,
        names=feature_columns_names + [label_column],
        dtype=merge_two_dicts(feature_columns_dtype, label_column_dtype),
    )
    numeric_features = list(feature_columns_names)
    # Removing features with multi-colinearity
    numeric_features.remove("solar_energy","uv_index")
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
        ]
    )

    y = df.pop("gt_energy_yield")
    X_pre = preprocess.fit_transform(df)
    y_pre = y.to_numpy().reshape(len(y), 1)

    X = np.concatenate((y_pre, X_pre), axis=1)

    np.random.shuffle(X)
    train, validation, test = np.split(X, [int(0.7 * len(X)), int(0.85 * len(X))])

    pd.DataFrame(train).to_csv(f"{base_dir}/train/train.csv", header=False, index=False)
    pd.DataFrame(validation).to_csv(
        f"{base_dir}/validation/validation.csv", header=False, index=False
    )
    pd.DataFrame(test).to_csv(f"{base_dir}/test/test.csv", header=False, index=False)
