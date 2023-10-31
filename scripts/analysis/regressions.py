import json
from pathlib import Path

import geopandas as gpd
import h3pandas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import statsmodels.formula.api as smf
from sklearn.feature_selection import r_regression
from sklearn.linear_model import LinearRegression
from statsmodels.tools.eval_measures import rmse

from src.common.utils import Paths


def process_pci(pci_path: Path) -> pl.DataFrame:
    pci = pl.read_parquet(pci_path)
    urc = (
        gpd.read_file(Paths.RAW_DATA / "UAC_EW.gpkg")
        .assign(urc=lambda x: x["RUC11"].str.split().str[0])
        .loc[:, ["LSOA11CD", "urc", "geometry"]]
    )
    _ = (
        gpd.read_file(Paths.RAW_DATA / "UAC_S.gpkg")
        .assign(
            urc=lambda x: x["Rank"].map(
                {1: "Urban", 2: "Urban", 3: "Rural", 4: "Rural", 5: "Rural", 6: "Rural"}
            )
        )
        .rename(columns={"DataZone": "LSOA11CD"})
        .loc[:, ["LSOA11CD", "urc", "geometry"]]
    )
    urc = pd.concat([urc, _], axis=0)

    unh3 = (
        pci[["h3_05", "easting", "northing"]]
        .unique("h3_05")
        .to_pandas()
        .set_index("h3_05")
        .h3.h3_to_geo_boundary()
        .to_crs(27700)
    )
    urc = gpd.sjoin(unh3, urc)
    urc = (
        pl.from_pandas(urc.reset_index()[["h3_05", "urc"]])
        .groupby("h3_05")
        .agg(pl.col("urc").mode().apply(lambda x: x[0]))
    )
    pci = pci.join(urc, on="h3_05")

    return pci.with_columns(
        [
            np.log(pl.col("matched")).prefix("log_"),
            np.log(pl.col("total")).prefix("log_"),
            np.log(pl.col("distance") + 1).prefix("log_"),
        ]
    )


def multiple_regression(pci):
    lr = LinearRegression()
    lr.fit(pci[["log_total", "log_distance"]], pci[["log_matched"]])

    pci["pred"] = lr.predict(pci[["log_total", "log_distance"]])
    pci["resid"] = pci["log_matched"] - pci["pred"]

    score = lr.score(pci[["log_total", "log_distance"]], pci[["log_matched"]])
    beta = -(lr.coef_[0][1] / lr.coef_[0][0])
    error = rmse(pci["pred"], pci["log_matched"])
    pearson = r_regression(pci[["pred"]], pci["log_matched"])[0]
    metrics = {"score": score, "beta": beta, "rmse": error, "pearson": pearson}

    return metrics, pci


def lmm_calc(pci):
    md = smf.mixedlm(
        "log_matched ~ log_distance + log_total -1",
        pci,
        groups="h3_05",
        re_formula="~log_total + log_distance",
    )
    mdf = md.fit()

    re = pd.DataFrame.from_dict(mdf.random_effects, orient="index").reset_index()
    re["beta"] = -(re["log_distance"] / mdf.params["log_total"])

    pci["resid"] = mdf.resid
    pci["pred"] = pci["log_matched"] + pci["resid"]

    error = rmse(pci["pred"], pci["log_matched"])
    pearson = r_regression(pci[["pred"]], pci["log_matched"])[0]
    metrics = {"pearson": pearson, "rmse": error}

    return metrics, pci, re


if __name__ == "__main__":
    pci = process_pci(Paths.OUT / "pci_h3.parquet").to_pandas()

    lr_metrics, lr = multiple_regression(pci)
    with open(Paths.OUT / "regressions/lr_metrics.json", "w") as f:
        json.dump(lr_metrics, f)
    lr.to_parquet(Paths.OUT / "regressions/lr.parquet", index=False)

    lmm_metrics, lmm_pci, lmm_re = lmm_calc(pci)

    with open("./data/out/regressions/mm_metrics.json", "w") as f:
        json.dump(lmm_metrics, f)

    lmm_pci.to_parquet(Paths.OUT / "regressions/mm.parquet", index=False)
    lmm_re.to_parquet(Paths.OUT / "regressions/mm_random_effects.parquet")
