import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import statsmodels.formula.api as smf
from sklearn.feature_selection import r_regression
from sklearn.linear_model import LinearRegression
from statsmodels.tools.eval_measures import rmse

from src.common.utils import Paths


def process_pci(pci_path: Path) -> pl.DataFrame:
    return (
        pl.scan_parquet(pci_path)
        .with_columns(
            [
                np.log(pl.col("matched")).prefix("log_"),
                np.log(pl.col("total")).prefix("log_"),
                np.log(pl.col("distance") + 1).prefix("log_"),
            ]
        )
        .collect()
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
        "log_matched ~ log_distance + log_total",
        pci,
        groups=pci["h3_05"],
        re_formula="log_distance",
    )

    mdf = md.fit()
    re = pd.DataFrame.from_dict(mdf.random_effects, orient="index")
    re["beta"] = -(re["log_distance"] / mdf.params["log_total"])

    pci["resid"] = mdf.resid
    pci["pred"] = pci["log_matched"] + pci["resid"]

    error = rmse(pci["pred"], pci["log_matched"])
    pearson = r_regression(pci[["pred"]], pci["log_matched"])[0]
    metrics = {"pearson": pearson, "rmse": error}

    return metrics, pci, re


if __name__ == "__main__":
    h3_pci = process_pci(Paths.OUT / "h3_pci.parquet").to_pandas()

    h3_lr_metrics, h3_lr = multiple_regression(h3_pci)
    with open(Paths.OUT / "regressions/h3_lr_metrics.json", "w") as f:
        json.dump(h3_lr_metrics, f)
    h3_lr.to_parquet(Paths.OUT / "regressions/h3_lr.parquet", index=False)

    lmm_metrics, lmm_pci, lmm_re = lmm_calc(h3_pci)

    with open("./data/out/regressions/mm_metrics.json", "w") as f:
        json.dump(lmm_metrics, f)

    lmm_pci.to_parquet(Paths.OUT / "regressions/mm.parquet", index=False)
    lmm_re.to_parquet(Paths.OUT / "regressions/mm_random_effects.parquet")
