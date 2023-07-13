import jsonlines
import polars as pl
from transformers import pipeline

from src.common.utils import Paths

text = pl.read_parquet(Paths.PROCESSED / "label" / "comments_unlabelled.parquet")[
    "text"
].to_list()
generator = pipeline(
    task="ner",
    model="cjber/reddit-ner-place_names",
    tokenizer="cjber/reddit-ner-place_names",
    aggregation_strategy="first",
    device=0,
)

out = generator(text)

out_format = []
for t, o in zip(text, out):
    d = {
        "text": t,
        "label": [[ent["start"], ent["end"], ent["entity_group"]] for ent in o],
    }
    out_format.append(d)

with jsonlines.open("doccano_input.jsonl", "w") as writer:
    for line in out_format:
        writer.write(line)
