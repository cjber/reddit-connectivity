import h3pandas
import polars as pl

from src.common.utils import Paths

if __name__ == "__main__":
    (
        pl.read_parquet(Paths.OUT / "places.parquet")
        .select("h3_05")
        .unique()
        .to_pandas()
        .set_index("h3_05")
        .h3.h3_to_geo_boundary()
        .to_parquet(Paths.OUT / "h3_poly.parquet")
    )
