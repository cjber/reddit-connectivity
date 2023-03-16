import geopandas as gpd
import h3pandas
import pandas as pd


def aggregate_to_lad(places: pd.DataFrame, lad: pd.DataFrame):
    return gpd.sjoin(lad, places, how="right").drop("index_left", axis=1)


def aggregate_to_h3(places, resolution: int):
    return (
        places.to_crs(4326)
        .assign(lng=lambda x: x.geometry.x, lat=lambda x: x.geometry.y)
        .h3.geo_to_h3(resolution=resolution)
        .h3.h3_to_geo()
        .to_crs(27700)
        .assign(
            h3_easting=lambda x: x.geometry.x.astype(int),
            h3_northing=lambda x: x.geometry.y.astype(int),
        )
        .reset_index()
        .drop("geometry", axis=1)
    )
