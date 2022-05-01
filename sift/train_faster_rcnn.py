"""
Faster-RCNN Training Script.
This script is a simplified version of the training script in detectron2/tools.
"""

import wandb
import torch
from torch.utils.data import DataLoader
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator

from sift.data import VOCCorruptionDataModule, build_transforms
from sift.conf.detection_configs import faster_rcnn_resnet_config
from sift.networks import build_resnet50_fpn_backbone, build_convnext_fpn_backbone


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains a number pre-defined logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can use the cleaner
    "SimpleTrainer", or write your own training loop.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        return COCOEvaluator()

    @classmethod
    def build_train_loader(cls, cfg):
        datamodule = VOCCorruptionDataModule(
            root=cfg.voc.root,
            year=cfg.voc.year,
            batch_size=1,
            transforms=build_transforms(cfg.voc.transforms),
            corruption_transforms=build_transforms(cfg.voc.corruption_transforms)
        )
        datamodule.setup(stage="fit")
        return datamodule.train_dataloader()


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    faster_rcnn_resnet_config(cfg, args)
    if not torch.cuda.is_available():
        cfg.MODEL.DEVICE = "cpu"
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        return res

    wandb.login()
    wandb.init(
        project="Faster-RCNN",
        sync_tensorboard=True,
    )

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument(
        "--backbone",
        choices=["ResNet50", "ConvNext"],
        default="ResNet50",
        help="The backbone network used for training FasterRCNN and/or SIFT",
    )

    parser.add_argument(
        "--dataset",
        choices=["COCO", "VOC"],
        default="VOC",
        help="The dataset used for training FasterRCNN and/or SIFT/BAR",
    )

    args = parser.parse_args()

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
