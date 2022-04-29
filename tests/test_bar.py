import os
import sys
import copy
import pytest
import logging

import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
from detectron2.structures import Instances

sys.path.append(os.path.dirname(os.getcwd()))
from sift.learners import BARLightning


@pytest.fixture
def config_path():
    return "../sift/conf/bar_config.yaml"


@pytest.fixture
def config(config_path):
    return OmegaConf.load(config_path)


@pytest.fixture
def image():
    return torch.randn((3, 224, 224))


@pytest.fixture
def gt_boxes():
    return torch.tensor([[2, 2, 5, 6]], dtype=torch.int16)


@pytest.fixture
def ct_boxes():
    return torch.tensor([[4, 2, 7, 6]], dtype=torch.int16)


@pytest.fixture
def labels():
    return torch.tensor([1], dtype=torch.int16)


@pytest.fixture
def gt(image, gt_boxes, labels):
    gt_annotations = {"gt_boxes": gt_boxes, "gt_classes": labels}
    gt_instance = Instances((224, 224), **gt_annotations)
    gt = {"image": image, "region_image": image, "instances": gt_instance}
    return gt


@pytest.fixture
def ct(image, ct_boxes, labels):
    ct_annotations = {"ct_boxes": ct_boxes, "ct_classes": labels}
    ct_instance = Instances((224, 224), **ct_annotations)
    ct = {"image": image, "region_image": image, "instances": ct_instance}
    return ct


@pytest.fixture
def bar(config):
    return BARLightning(config)


def test_reward(bar, gt, ct):
    bar.env.gt = gt
    bar.env.ct = ct
    bar.env.idx = 0

    # 1st iteration
    iou_1 = bar.env._compute_iou()
    assert round(float(iou_1.squeeze()), 4) == 0.2
    reward_1 = bar.env._update_reward()
    print(reward_1)
    bar.env.t += 1

    # 2nd itertation
    bar.env.ct["instances"].ct_boxes[0] = torch.tensor(
        [[3, 2, 6, 6]],
        dtype=torch.int16
    )
    iou_2 = bar.env._compute_iou()
    assert round(float(iou_2.squeeze()), 4) == 0.5
    reward_2 = bar.env._update_reward()
    print(reward_2)
    bar.env.t += 1

    # 3rd itertation
    bar.env.ct["instances"].ct_boxes[0] = torch.tensor(
        [[2, 2, 5, 6]],
        dtype=torch.int16
    )
    iou_3 = bar.env._compute_iou()
    assert round(float(iou_3.squeeze()), 4) == 1.0
    reward_3 = bar.env._update_reward()
    print(reward_3)
    bar.env.t += 1


def test_action(bar, gt, ct):
    bar.env.gt = copy.deepcopy(gt)
    bar.env.ct = copy.deepcopy(ct)
    bar.env.idx = 0
    bar.env._update_region()

    target_ct_boxes = torch.Tensor(
        [[3, 2, 6, 6],
         [4, 2, 7, 6],
         [4, 1, 7, 5],
         [4, 2, 7, 6],
         [4, 1, 7, 6],
         [3, 2, 7, 6],
         [4, 2, 6, 6],
         [4, 2, 7, 5]]
    )

    for i, target_ct_box in enumerate(target_ct_boxes):
        new_ct_box = _perform_action(bar, action=i)
        assert torch.equal(new_ct_box, target_ct_box)


def _perform_action(bar, action: int):
    bar.env._perform_action(action)
    bar.env._update_region()
    new_ct_box = copy.deepcopy(bar.env.ct["instances"].ct_boxes[0])
    return new_ct_box


def test_train(bar):
    trainer = pl.Trainer(
        max_epochs=200
    )

    trainer.fit(bar)
