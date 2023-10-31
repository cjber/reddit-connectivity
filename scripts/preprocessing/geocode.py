from math import dist

import geopandas as gpd
import pandas as pd
import polars as pl
from tqdm import tqdm

from src.common.utils import Paths


def weighted_mean_pts(global_context: pd.DataFrame, token) -> dict:
    global_context = (
        global_context.sort("distance_from_centre").groupby("word").head(50)
    )
    tokens = global_context.filter(pl.col("word") == token)

    dists = []
    for row in tokens.rows():
        pts = (
            global_context.with_columns(
                pl.struct(["easting", "northing"])
                .apply(lambda x: dist((x["easting"], x["northing"]), (row[1], row[2])))
                .alias("distance")
            )
            .sort("distance")
            .unique(subset="word")
        )
        dists.append(pts["distance"].mean())
    return tokens[dists.index(min(dists))]


def contexts(df: pd.DataFrame) -> gpd.GeoDataFrame:
    df_unique = df.unique(subset="word")

    centre_pt = pl.DataFrame(
        {
            "word": ["centre"],
            "easting": [float(df["centre_easting"][0])],
            "northing": [float(df["centre_northing"][0])],
            "place_id": [-1],
        },
    )

    outs = pl.DataFrame()
    for row in df_unique.rows():
        idxs = df.filter(pl.col("word") == row[9]).select("idx")
        global_context = (
            df.filter(pl.col("idx").is_in(idxs["idx"]))
            .select(["word", "easting", "northing", "place_id"])
            .unique()
        )
        global_context = pl.concat([global_context, centre_pt], how="vertical")
        global_context = global_context.with_columns(
            pl.struct(["easting", "northing"])
            .apply(
                lambda x: dist(
                    (x["easting"], x["northing"]),
                    (centre_pt["easting"][0], centre_pt["northing"][0]),
                )
            )
            .alias("distance_from_centre")
        )
        chosen_point = weighted_mean_pts(global_context, row[9])
        outs = pl.concat([outs, chosen_point])
    return outs


def main():
    IDX_FILE = Paths.PROCESSED / "ner_idx.txt"
    if IDX_FILE.exists():
        IDX_FILE.unlink()

    if not (Paths.PROCESSED / "geocoded.csv").exists():
        pd.DataFrame(
            {
                "idx": [],
                "subreddit": [],
                "text": [],
                "score": [],
                "thread": [],
                "created_utc": [],
                "author": [],
                "entity_group": [],
                "confidence": [],
                "word": [],
                "start": [],
                "end": [],
                "subreddit_lad19nm": [],
                "centre_easting": [],
                "centre_northing": [],
                "subreddit_type": [],
                "easting": [],
                "northing": [],
                "place_id": [],
                "distance_from_centre": [],
            }
        ).to_csv(Paths.PROCESSED / "geocoded.csv", index=False)

    gazetteer = (
        pl.scan_parquet(Paths.PROCESSED / "gazetteer.parquet")
        .with_columns(pl.col("name").str.to_lowercase())
        .collect()
    )
    centres = (
        pl.scan_parquet(Paths.RAW / "subreddits.parquet")
        .rename(
            {
                "easting": "centre_easting",
                "northing": "centre_northing",
                "lad19nm": "subreddit_lad19nm",
            }
        )
        .drop(["name", "__index_level_0__"])
        .with_columns(pl.col("subreddit").str.to_lowercase())
        .collect()
    )

    ner = iter(
        pl.read_csv(Paths.PROCESSED / "ner.csv")
        .with_columns(
            [
                pl.col("subreddit").str.to_lowercase(),
                pl.col("word").str.to_lowercase(),
            ]
        )
        .join(centres, on="subreddit", how="left")
        .drop_nulls(subset=["centre_easting", "centre_northing"])
        .partition_by("subreddit")
    )

    for subreddit in tqdm(ner, total=186):
        sub_geo = subreddit.join(gazetteer, left_on="word", right_on="name")

        if len(sub_geo) > 0:
            subreddit = subreddit.join(contexts(sub_geo), on="word", how="left")
            subreddit.to_pandas().to_csv(
                Paths.PROCESSED / "geocoded.csv",
                mode="a",
                index=False,
                header=False,
            )


if __name__ == "__main__":
    main()
