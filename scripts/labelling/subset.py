import polars as pl

from src.common.utils import Const, Paths, preprocess

full = pl.read_parquet(Paths.RAW_DATA / "comments_combined-2023_02_23.parquet")

(
    full.sample(50_000, seed=Const.SEED)
    .with_columns(
        pl.col("text")
        .apply(lambda s: preprocess(s, is_twitter=False))
        .apply(lambda s: " ".join(s.split(" ")[: Const.MAX_TOKEN_LEN]))
    )
    .filter(pl.col("text").str.n_chars() > 10)
    .sample(10_000, seed=Const.SEED)
    .write_parquet(Paths.PROCESSED / "label" / "comments_unlabelled.parquet")
)
