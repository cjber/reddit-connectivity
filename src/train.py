import os
from argparse import ArgumentParser

import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning import Callback, seed_everything
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import CSVLogger, MLFlowLogger

from src.common.utils import Paths
from src.pl_data.datamodule import DataModule
from src.pl_data.test_dataset import TestDataset
from src.pl_data.wnut_dataset import WNUTDataset
from src.pl_module.ger_model import GERModel

parser = ArgumentParser()

parser.add_argument("--fast_dev_run", type=bool, default=False)
parser.add_argument("--batch_size", type=int, default=12)
parser.add_argument("--seed", nargs="+", type=int, default=[42])
parser.add_argument("--upload", type=bool, default=False)

args, unknown = parser.parse_known_args()


def build_callbacks() -> list[Callback]:
    return [
        LearningRateMonitor(
            logging_interval="step",
            log_momentum=False,
        ),
        EarlyStopping(
            monitor="val_loss",
            mode="min",
            verbose=True,
            min_delta=0.0,
            patience=3,
        ),
        ModelCheckpoint(
            filename="checkpoint",
            monitor="val_f1",
            mode="max",
            save_top_k=1,
            verbose=True,
        ),
    ]


def run(
    dataset,
    testdataset,
    pl_model: pl.LightningModule,
    name: str,
    seed: int,
    args=args,
) -> None:
    seed_everything(seed, workers=True)

    datamodule: pl.LightningDataModule = DataModule(
        dataset=dataset,
        num_workers=int(os.cpu_count() // 2),
        batch_size=args.batch_size,
        seed=seed,
    )
    testmodule: pl.LightningDataModule = DataModule(
        dataset=testdataset,
        num_workers=int(os.cpu_count() // 2),
        batch_size=args.batch_size,
        seed=seed,
    )
    model: pl.LightningModule = pl_model()
    callbacks: list[Callback] = build_callbacks()
    csv_logger = CSVLogger(save_dir="logs", name=f"seed_{seed}", version=name)
    mlflow_logger = MLFlowLogger(
        experiment_name="ner_model_logs",
        tracking_uri="https://dagshub.com/cjber/reddit-connectivity.mlflow",
    )

    if args.fast_dev_run:
        trainer_kwargs = {"gpus": None, "auto_select_gpus": False}
    else:
        trainer_kwargs = {"gpus": -1, "auto_select_gpus": True, "precision": 16}

    trainer: pl.Trainer = pl.Trainer.from_argparse_args(
        args,
        **trainer_kwargs,
        deterministic=True,
        default_root_dir="ckpts",
        logger=[csv_logger, mlflow_logger],
        log_every_n_steps=10,
        callbacks=callbacks,
        max_epochs=5,
    )

    trainer.tune(model=model, datamodule=datamodule)
    trainer.fit(model=model, datamodule=datamodule)

    if args.upload:
        model.model.push_to_hub("cjber/reddit-ner-place_names")
        model.tokenizer.push_to_hub("cjber/reddit-ner-place_names")
    else:
        test = trainer.test(model=model, ckpt_path="best", datamodule=testmodule)
        pd.DataFrame(test).to_csv(f"logs/seed_{seed}_{name}_test.csv")
    csv_logger.save()


if __name__ == "__main__":
    labelled = Paths.PROCESSED / "labelled.jsonl"
    dataset = WNUTDataset(doccano=labelled) if args.upload else WNUTDataset()

    run(
        dataset=dataset,
        testdataset=TestDataset(path=labelled),
        pl_model=GERModel,
        name="ger",
        seed=args.seed[0],
    )