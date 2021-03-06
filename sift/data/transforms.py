from typing import Tuple, Optional

import albumentations as A
from detectron2.utils.registry import Registry

TRANSFORMS_REGISTRY = Registry("TRANSFORMS")


def build_transforms(transforms_name: str):
    return TRANSFORMS_REGISTRY.get(transforms_name)()


@TRANSFORMS_REGISTRY.register()
def voc_transforms(
    image_size: Tuple[int, int] = (224, 224),
    hflip: Optional[float] = 0.0,
    vflip: Optional[float] = 0.0,
    brightness_contrast: Optional[float] = 0.0,
) -> A.Compose:
    transforms = A.Compose(
        [
            A.Resize(width=image_size[1], height=image_size[0]),
            A.HorizontalFlip(p=hflip),
            A.VerticalFlip(p=vflip),
            A.RandomBrightnessContrast(p=brightness_contrast),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["class_labels"]
        ),
    )
    return transforms


@TRANSFORMS_REGISTRY.register()
def voc_corruption_transforms(
    scale: Optional[float] = 0.3,
    prob: Optional[float] = 0.5
) -> A.Compose:
    transforms = A.Compose(
        [
            A.RandomScale(scale_limit=scale, p=prob),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["class_labels"]
        ),
    )
    return transforms
