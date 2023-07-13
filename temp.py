import geopandas as gpd
import h3pandas
import matplotlib.pyplot as plt

from src.common.utils import Paths

wales = gpd.GeoDataFrame(
    {"geometry": [gpd.read_file(Paths.RAW_DATA / "wales_bdry.gpkg").unary_union]}
)
wales["RGN21CD"] = "WALES"
scotland = gpd.GeoDataFrame(
    {"geometry": [gpd.read_file(Paths.RAW_DATA / "scot_bdry.gpkg").unary_union]}
)
scotland["RGN21CD"] = "SCOT"
EN_REGIONS = (
    gpd.read_file(Paths.RAW_DATA / "en_regions.gpkg")[["RGN21CD", "geometry"]]
    .append(wales)
    .append(scotland)
)
regions = EN_REGIONS.to_crs(4326).h3.polyfill_resample(5)

unions = [
    poly.unary_union
    for idx, poly in (
        regions.reset_index().drop_duplicates(subset="h3_polyfill").groupby("RGN21CD")
    )
]

unions = gpd.GeoDataFrame({"geometry": unions}, crs=4326)
unions.to_parquet("./data/out/region_poly.parquet")
