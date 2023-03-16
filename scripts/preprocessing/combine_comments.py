import jsonlines
import polars as pl
from tqdm import tqdm

from src.common.utils import Paths

if __name__ == "__main__":
    all_files = (Paths.RAW_DATA / "comments-2022_04_17").glob("*.jsonl")
    dicts = []
    for file in tqdm(all_files, total=258):
        with jsonlines.open(file) as reader:
            dicts.extend(iter(reader))

    (
        pl.DataFrame(dicts)
        .with_columns(pl.from_epoch(pl.col("created_utc")))
        .drop("idx")
        .write_parquet(Paths.PROCESSED / "comments.parquet")
    )
