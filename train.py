import argparse

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from omegaconf import OmegaConf
from hydra.utils import instantiate

from sift.learners import SIFT


def train(args):
    config = OmegaConf.load(args.cfg)
    if args.exp == "SIFT":
        module = SIFT(
            **config.module,
            **config.env
        )

    logger = WandbLogger(project=args.exp, log_model="all", save_dir="logs")
    logger.experiment.config.update(config)

    # save checkpoints based on avg_reward
    checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="avg_val_reward", mode="max", verbose=True)

    trainer = Trainer(
        max_epochs=config.trainer.epochs,
        deterministic=True,
        check_val_every_n_epoch=config.trainer.val_period,
        logger=logger,
        callbacks=checkpoint_callback
    )
    trainer.fit(module)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp", type=str, required=True, choices=["BAR", "SIFT"], help="Which experiment to run."
    )
    parser.add_argument(
        "--cfg", type=str, required=True, help="Path to config.yaml file"
    )

    args = parser.parse_args()
    train(args)