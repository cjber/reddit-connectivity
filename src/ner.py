import warnings
from pathlib import Path

import pandas as pd
import polars as pl
import pytorch_lightning as pyl
from datasets import Dataset
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from transformers import pipeline

from src.common.utils import Paths
from src.pl_data.jsonl_dataset import JSONLDataset

pyl.seed_everything(42)


def load_pretrained_model(model: pyl.LightningModule, checkpoint: Path, device: str):
    model = model.load_from_checkpoint(checkpoint).to(device)

    model.eval()
    model.freeze()
    return model


def load_dataset(datadir: Path, batch_size: int):
    ds = JSONLDataset(path=datadir)
    return DataLoader(dataset=ds, batch_size=batch_size)


def main():
    OUT_FILE = Paths.PROCESSED / "ner.csv"
    IDX_FILE = Paths.PROCESSED / "ner_idx.txt"

    pipe = pipeline(
        task="ner",
        model="cjber/reddit-ner-place_names",
        tokenizer="cjber/reddit-ner-place_names",
        aggregation_strategy="first",
        device=0,
    )
    idx = 0

    if IDX_FILE.exists():
        with open(IDX_FILE) as file:
            for line in file:
                idx = int(line)

    if not OUT_FILE.exists():
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
            }
        ).to_csv(OUT_FILE, index=False)

    comments = Dataset(
        pl.scan_parquet(Paths.PROCESSED / "comments.parquet")
        .with_columns(
            pl.col("text")
            .str.replace(r"http\S+", "")
            .str.replace(r"\r", "")
            .str.replace(r"\n", "")
            .str.strip()
        )
        .filter(pl.col("text").str.lengths() >= 3)[idx:]
        .select(["subreddit", "text", "author"])
        .collect()
        .to_arrow()
    )

    out_df = pd.DataFrame()
    for _, row in tqdm(comments.to_pandas().iterrows(), total=len(comments)):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="You seem to be using")
            out = pipe(row["text"])

        for item in out:
            new_row = pd.concat([row, pd.Series(item)], ignore_index=True)
            out_df = pd.concat(
                [out_df, new_row.to_frame().T], ignore_index=True)

        if (idx % 10_000 == 0) or (idx == len(comments) - 1):
            with open(IDX_FILE, "a") as idx_file:
                idx_file.write(f"{idx}\n")
            out_df.to_csv(OUT_FILE, mode="a", index=False, header=False)
            out_df = pd.DataFrame()
        idx += 1


if __name__ == "__main__":
    main()
