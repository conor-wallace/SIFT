import os
import sys
import copy
import pytest
import logging

import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
from detectron2.structures import Instances
from detectron2.utils.visualizer import Visualizer
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.getcwd()))
from sift.learners import SIFT


@pytest.fixture
def config_path():
    return "../sift/conf/sift_config.yaml"


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
def sift(config):
    return SIFT(
        **config.train,
        **config.env
    )


def test_augmentation(sift):
    sift.env._load_next_sample()
    vis = Visualizer(sift.env.gt["image"].permute(1, 2, 0))

    print(sift.env.gt["instances"].gt_boxes)
    print(sift.env.ct["instances"].ct_boxes)

    for i in range(sift.env.gt["instances"].gt_boxes.shape[0]):
        sift.env.idx = i
        iou = sift.env._compute_iou()
        print(iou)
        image = vis.draw_box(tuple(sift.env.gt["instances"].gt_boxes[i]), edge_color="blue")
        plt.imshow(image.get_image())
        image = vis.draw_box(tuple(sift.env.ct["instances"].ct_boxes[i]), edge_color="red")
        plt.imshow(image.get_image())
        plt.show()


def test_regression():
    gt_box = torch.tensor([2, 2, 5, 6], dtype=torch.int16)
    ct_box = torch.tensor([4, 2, 7, 6], dtype=torch.int16)
    ct_w = ct_box[2] - ct_box[0]
    ct_h = ct_box[3] - ct_box[1]

    t_xmin = (gt_box[0] - ct_box[0]) / ct_w
    t_ymin = (gt_box[1] - ct_box[1]) / ct_h
    t_xmax = (gt_box[2] - ct_box[2]) / ct_w
    t_ymax = (gt_box[3] - ct_box[3]) / ct_h

    ct_box[0] = t_xmin * ct_w + ct_box[0]
    ct_box[1] = t_ymin * ct_h + ct_box[1]
    ct_box[2] = t_xmax * ct_w + ct_box[2]
    ct_box[3] = t_ymax * ct_h + ct_box[3]


def test_reward(sift, gt, ct):
    sift.env.gt = gt
    sift.env.ct = ct
    sift.env.idx = 0

    # 1st iteration
    iou_1 = sift.env._compute_iou()
    assert round(float(iou_1.squeeze()), 4) == 0.2
    reward_1 = sift.env._update_reward()
    print(reward_1)
    sift.env.t += 1

    # 2nd itertation
    sift.env.ct["instances"].ct_boxes[0] = torch.tensor(
        [[3, 2, 6, 6]],
        dtype=torch.int16
    )
    iou_2 = sift.env._compute_iou()
    assert round(float(iou_2.squeeze()), 4) == 0.5
    reward_2 = sift.env._update_reward()
    print(reward_2)
    sift.env.t += 1

    # 3rd itertation
    sift.env.ct["instances"].ct_boxes[0] = torch.tensor(
        [[2, 2, 5, 6]],
        dtype=torch.int16
    )
    iou_3 = sift.env._compute_iou()
    assert round(float(iou_3.squeeze()), 4) == 1.0
    reward_3 = sift.env._update_reward()
    print(reward_3)
    sift.env.t += 1

def test_action(sift, gt, ct):
    sift.env.gt = copy.deepcopy(gt)
    sift.env.ct = copy.deepcopy(ct)
    sift.env.idx = 0
    sift.env._update_region()

    action = torch.Tensor([-0.6667, 0., -0.6667, 0.])
    sift.env._perform_action(action)
    sift.env._update_region()

    sift.env._load_next_sample()
    vis = Visualizer(sift.env.gt["image"].permute(1, 2, 0))

    print(sift.env.gt["instances"].gt_boxes)
    print(sift.env.ct["instances"].ct_boxes)

    for i in range(sift.env.gt["instances"].gt_boxes.shape[0]):
        gt_box = tuple(sift.env.gt["instances"].gt_boxes[i])
        ct_box = tuple(sift.env.ct["instances"].ct_boxes[i])
        ct_w = ct_box[2] - ct_box[0]
        ct_h = ct_box[3] - ct_box[1]

        t_xmin = (gt_box[0] - ct_box[0]) / ct_w
        t_ymin = (gt_box[1] - ct_box[1]) / ct_h
        t_xmax = (gt_box[2] - ct_box[2]) / ct_w
        t_ymax = (gt_box[3] - ct_box[3]) / ct_h

        sift.env.idx = i
        iou = sift.env._compute_iou()
        print(iou)
        image = vis.draw_box(tuple(sift.env.ct["instances"].ct_boxes[i]), edge_color="red")
        action = torch.Tensor([t_xmin, t_ymin, t_xmax, t_ymax])
        print(action)
        sift.env._perform_action(action)
        iou = sift.env._compute_iou()
        print(iou)
        image = vis.draw_box(tuple(sift.env.ct["instances"].ct_boxes[i]), edge_color="orange")
        image = vis.draw_box(tuple(sift.env.gt["instances"].gt_boxes[i]), edge_color="blue")
        plt.imshow(image.get_image())
        plt.show()


# def test_train(sift):
#     trainer = pl.Trainer(
#         max_epochs=200
#     )

#     trainer.fit(sift)
