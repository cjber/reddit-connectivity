import os
import re
from pathlib import Path

import polars as pl

pl.Config.set_tbl_formatting("NOTHING")
pl.Config.set_tbl_dataframe_shape_below(True)
pl.Config.set_tbl_rows(6)


class Paths:
    RAW_DATA = Path(os.environ["DATA_DIR"])
    DATA = Path("data")
    PROCESSED = DATA / "processed"
    OUT = DATA / "out"


class Const:
    GER_MODEL = "bert-base-uncased"
    MAX_TOKEN_LEN = 256

    with open(Paths.PROCESSED / "exclude.txt", "r") as exclude:
        EXCLUDE = {line.strip() for line in exclude.readlines()}


class Label:
    labels: dict[str, int] = {"O": 0, "B-location": 1, "I-location": 2}
    idx: dict[int, str] = {v: k for k, v in labels.items()}
    count: int = len(labels)


def preprocess(word):
    word = "@user" if word.startswith("@") and len(word) > 1 else word
    word = "http" if re.compile(r"http\S+").search(word) else word
    word = "n" if re.compile(r"\r").search(word) else word
    word = "n" if re.compile(r"\n").search(word) else word
    word = word.encode("ascii", "ignore").decode("ascii")
    word = word.strip()

    if len(word) < 1:
        word = "n"
    return word
