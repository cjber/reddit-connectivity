from pathlib import Path

import geopandas as gpd
import h3pandas  # noqa
import pandas as pd
import polars as pl

from src.common.utils import Paths


def process_osnames(path: Path, header_path: Path) -> pl.DataFrame:
    os_header = pl.read_csv(header_path)
    mappings = {
        "landcover": "natural",
        "landform": "natural",
        "hydrography": "natural",
        "populatedPlace": "populated",
        "transportNetwork": "populated",
        "other": "populated",
    }

    df = (
        pl.concat(
            [
                pl.read_csv(file, new_columns=os_header.columns, has_header=False)
                .select(["NAME1", "TYPE", "LOCAL_TYPE", "GEOMETRY_X", "GEOMETRY_Y"])
                .filter(
                    (pl.col("LOCAL_TYPE").str.contains("Station").is_not())
                    & (pl.col("LOCAL_TYPE") != "Postcode")
                )
                for file in path.glob("*.csv")
                if not file.name.startswith("0")
            ]
        )
        .with_columns(pl.col("TYPE").apply(lambda s: mappings[s]))
        .filter(pl.col("NAME1").is_in(["Newcastle", "Aberdeen"]).is_not())
        .rename(
            {
                "NAME1": "name",
                "TYPE": "type",
                "GEOMETRY_X": "easting",
                "GEOMETRY_Y": "northing",
            }
        )
        .select(["name", "type", "easting", "northing"])
    )

    fix_df = pl.DataFrame(
        {
            "name": ["Newcastle", "Aberdeen"],
            "type": ["populated", "populated"],
            "easting": [425048, 398593],
            "northing": [564892, 799603],
        }
    )

    return pl.concat([df, fix_df]).with_columns(
        [pl.col("easting").cast(pl.Float64), pl.col("northing").cast(pl.Float64)]
    )


def process_gbpn(path: Path) -> pl.DataFrame:
    df = pd.read_csv(
        path,
        usecols=["PlaceName", "Lat", "Lng", "Type"],
        dtype={"Lat": "str", "Lng": "float"},
    ).rename(columns={"PlaceName": "name", "Type": "type"})
    df["type"] = df["type"].replace(
        {
            "Settlement": "populated",
            "Heritage SIte": "natural",
            "Heritage Site": "natural",
            "Heritage Site.": "natural",
            "Heritage Site. ": "natural",
            "Island": "natural",
            "Valley": "natural",
            "Area": "populated",
            "Downs, Moorland": "natural",
            "Beach": "natural",
            "Coastal Feature, Headland, Point": "natural",
            "Hill, Mountain": "natural",
            "Community": "populated",
            "Civil Parish": "populated",
            "Wood, Forest": "natural",
            "Island Group": "natural",
            "Lake, Pool, Pond, Freshwater Marsh": "natural",
            "Landscape Feature": "natural",
            "Cliff, Slope": "natural",
            "Hill, Mountain.": "natural",
            "Antiquity": "natural",
            "Range of Mountains, Range of Hills": "natural",
            "Mountain, Hill": "natural",
            "Waterfall": "natural",
            "Bay": "natural",
            "Urban Greenspace": "natural",
            "Sea, Estuary, Creek": "natural",
            "Coastal Marsh, Saltings": "natural",
            "Corrie (Glacial Valley)": "natural",
            "Park": "natural",
            "Woodland, Forest": "natural",
            "Village": "populated",
            "Historic County": "populated",
        }
    )
    # manual fixes
    df.loc[lambda x: x["name"] == "Linn Park", "type"] = "natural"

    # only select natural names
    df = df[df["type"] == "natural"]

    df = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(
            df["Lng"],
            df["Lat"].str.extract(r"(\d*\.?\d*)", expand=False),
        ),
        crs=4326,
    ).to_crs(27700)

    df["easting"] = df.geometry.x
    df["northing"] = df.geometry.y

    return pl.from_pandas(df[["name", "type", "easting", "northing"]])


def join_gazetteer(
    gbpn: pl.DataFrame,
    osnames: pl.DataFrame,
    lad: gpd.GeoDataFrame,
    lsoa: gpd.GeoDataFrame,
) -> pl.DataFrame:
    gazetteer = pl.concat([osnames, gbpn], how="vertical").to_pandas()
    gazetteer = gpd.GeoDataFrame(
        gazetteer,
        geometry=gpd.points_from_xy(gazetteer["easting"], gazetteer["northing"]),
        crs=27700,
    )
    glad = pl.from_pandas(
        gpd.sjoin(lad, gazetteer, how="right")
        .assign(lad_easting=lambda x: x.geometry.x, lad_northing=lambda x: x.geometry.y)
        .drop(["geometry", "index_left"], axis=1)
    )[["name", "easting", "northing", "LAD21NM", "LAD21CD"]]
    glsoa = pl.from_pandas(
        gpd.sjoin(lsoa, gazetteer, how="right")
        .assign(lad_easting=lambda x: x.geometry.x, lad_northing=lambda x: x.geometry.y)
        .drop(["geometry", "index_left"], axis=1)
    )[["name", "easting", "northing", "LSOA21NM", "LSOA21CD"]]
    gazetteer = pl.from_pandas(
        gazetteer.to_crs(4326)
        .assign(lng=lambda x: x.geometry.x, lat=lambda x: x.geometry.y)
        .h3.geo_to_h3(resolution=5)
        .h3.h3_to_geo()
        .to_crs(27700)
        .assign(
            h3_easting=lambda x: x.geometry.x.astype(int),
            h3_northing=lambda x: x.geometry.y.astype(int),
        )
        .reset_index()
        .rename(columns={"h3_04": "h3_05"})
        .drop("geometry", axis=1)
    )
    gazetteer = (
        gazetteer.join(glad, on=["name", "easting", "northing"])
        .join(glsoa, on=["name", "easting", "northing"])
        .with_columns(
            (
                pl.col("name")
                + pl.col("easting").cast(str)
                + pl.col("northing").cast(str)
            )
            .cast(pl.Categorical)
            .cast(pl.Int32)
            .alias("place_id")
        )
    )
    return gazetteer


if __name__ == "__main__":
    gbpn = process_gbpn(Paths.RAW_DATA / "gbpn-2021_14_06.csv")
    osnames = process_osnames(
        path=Paths.RAW_DATA / "os_opname-2023_02_21",
        header_path=Paths.RAW_DATA / "os_opname-2023_02_21/0_header.csv",
    )
    lad = gpd.read_file(Paths.RAW_DATA / "lad-2021.gpkg")[
        ["LAD21CD", "LAD21NM", "geometry"]
    ]
    lsoa = gpd.read_file(
        Paths.RAW_DATA
        / "LSOA_Dec_2021_Boundaries_Full_Clipped_EW_BFC_2022_-6437031168783062454.gpkg"
    )[["LSOA21CD", "LSOA21NM", "geometry"]]

    gazetteer = join_gazetteer(gbpn, osnames, lad, lsoa)
    gazetteer.write_parquet(Paths.PROCESSED / "gazetteer.parquet")
