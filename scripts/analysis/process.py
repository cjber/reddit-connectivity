import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm

from scripts.analysis.pci import create_pci, process_authors
from src.common.utils import Const, Paths


def process_places(places):
    return places.drop_nulls(subset=["easting", "northing"]).filter(
        pl.col("word").is_in(list(Const.EXCLUDE)).is_not()
    )


if __name__ == "__main__":
    places = pl.read_csv(Paths.PROCESSED / "geocoded.csv").with_columns(
        pl.col("place_id").cast(int)
    )
    gazetteer = pl.read_parquet(Paths.PROCESSED / "gazetteer.parquet")
    places = places.join(gazetteer, on=["easting", "northing"], how="left")

    places.write_parquet(Paths.OUT / "places_full.parquet")

    places = process_places(places)
    top_h3 = (
        places.groupby(["h3_05", "word"])
        .count()
        .sort("count", reverse=True)
        .unique(subset="h3_05")
        .rename({"word": "top_h3_word", "count": "total_count"})
    )
    places = places.join(top_h3, on="h3_05")
    places.write_parquet(Paths.OUT / "places.parquet")

    # PCI calculations
    places = (
        places.filter(pl.col("author") != "deleted")
        .with_columns(pl.col("author").cast(pl.Categorical).cast(int))
        .with_columns(pl.lit(1).count().over("place_id").alias("place_count"))
    )

    # PCI H3
    h3_authors = process_authors(
        places.drop_nulls(subset="lad19nm"),
        ["h3_05", "h3_easting", "h3_northing", "top_h3_word"],
    ).rename(columns={"h3_easting": "easting", "h3_northing": "northing"})

    h3_pci = pd.concat(
        create_pci(
            chunk,
            h3_authors,
            ["top_h3_word", "h3_05", "easting", "northing"],
            id="h3_05",
        )
        for _, chunk in tqdm(h3_authors.groupby(np.arange(len(h3_authors)) // 1000))
    )
    h3_pci.to_parquet(Paths.OUT / "h3_pci.parquet")
