import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm

from scripts.analysis.pci import create_pci, process_authors
from src.common.utils import Const, Paths


def process_places(places):
    places = (
        places.drop_nulls(subset=["easting", "northing"])
        .filter(
            (pl.col("word").is_in(list(Const.EXCLUDE)).is_not())
            & (pl.col("author") != "deleted")
        )
        .with_columns(pl.col("created_utc").str.strptime(pl.Datetime))
        .sort("created_utc")
    )

    top_h3 = (
        places.groupby(["h3_05", "word"])
        .count()
        .sort("count", descending=True)
        .unique(subset="h3_05")
        .rename({"word": "top_h3_word", "count": "total_count"})
    )
    return places.join(top_h3, on="h3_05")


if __name__ == "__main__":
    places = pl.read_csv(Paths.PROCESSED / "geocoded.csv")
    gazetteer = pl.read_parquet(Paths.PROCESSED / "gazetteer.parquet")
    places = places.join(gazetteer, on=["easting", "northing"], how="left")
    places.write_parquet(Paths.OUT / "places_full.parquet")

    places = process_places(places)
    places.write_parquet(Paths.OUT / "places.parquet")

    places = places.drop(["easting", "northing"]).rename(
        {"h3_easting": "easting", "h3_northing": "northing"}
    )

    # PCI H3
    h3_authors = process_authors(
        places.drop_nulls(subset="LAD21NM"),
        ["h3_05", "easting", "northing", "top_h3_word"],
    )

    h3_pci = pd.concat(
        create_pci(
            chunk,
            h3_authors,
            ["top_h3_word", "h3_05", "easting", "northing"],
            id="h3_05",
        )
        for _, chunk in tqdm(h3_authors.groupby(np.arange(len(h3_authors)) // 1000))
    )
    h3_pci.to_parquet(Paths.OUT / "pci_h3.parquet")
