from math import dist, sqrt

import pandas as pd
import polars as pl


def process_authors(places: pd.DataFrame, group: list[str]):
    authors = (
        places.groupby(group)
        .agg(
            [
                pl.col("author").count().alias("author_count"),
                pl.col("author").unique().alias("author_unique"),
                pl.col("author").n_unique().alias("author_nunique"),
            ]
        )
        .to_pandas()
    )
    authors["author_unique"] = authors["author_unique"].apply(set)
    return authors


def create_pci(chunk, authors, var: list[str], id, distance=True):
    more_vars = ["author_count", "author_nunique", "author_unique"]
    groups = var + more_vars
    for item in groups:
        chunk[f"target_{item}"] = len(chunk) * [authors[item]]

    chunk = chunk.explode([f"target_{item}" for item in groups]).loc[
        lambda x: (x[f"{id}"] != x[f"target_{id}"]), :
    ]

    chunk["matched"] = [
        len(origin & target)
        for origin, target in zip(chunk["author_unique"], chunk["target_author_unique"])
    ]

    chunk["total"] = [
        len(x[0]) * len(x[1])
        for x in zip(chunk["author_unique"], chunk["target_author_unique"])
    ]
    chunk["PCI"] = chunk["matched"] / (chunk["total"]).apply(sqrt)

    if distance:
        chunk["distance"] = [
            dist((e, n), (te, tn))
            for e, n, te, tn in zip(
                chunk["easting"],
                chunk["northing"],
                chunk["target_easting"],
                chunk["target_northing"],
            )
        ]
    return chunk[chunk["PCI"] != 0].drop(
        ["author_unique", "target_author_unique"], axis=1
    )
