# --------------------------------------------------------
# Copyright (c) 2023 Microsoft
# Licensed under The MIT License
# Written by Zhipeng Huang
# --------------------------------------------------------

# from torchvision.datasets import ImageFolder
from utils.my_dataset_folder import ImageFolder
import os
from typing import Optional, Tuple, Dict, List, Union
import torch
import argparse

from utils import logger

from .. import register_dataset
from ..dataset_base import BaseImageDataset
from ...transforms import image_pil as T
from ...collate_fns import register_collate_fn


@register_dataset(name="imagenet_fast", task="classification")
class ImagenetDataset(BaseImageDataset, ImageFolder):
    """
    ImageNet Classification Dataset that uses PIL for reading and augmenting images. The dataset structure should
    follow the ImageFolder class in :class:`torchvision.datasets.imagenet`

    Args:
        opts: command-line arguments
        is_training (Optional[bool]): A flag used to indicate training or validation mode. Default: True
        is_evaluation (Optional[bool]): A flag used to indicate evaluation (or inference) mode. Default: False

    .. note::
        We recommend to use this dataset class over the imagenet_opencv.py file.

    """

    def __init__(
        self,
        opts,
        is_training: Optional[bool] = True,
        is_evaluation: Optional[bool] = False,
        *args,
        **kwargs
    ) -> None:
        BaseImageDataset.__init__(
            self, opts=opts, is_training=is_training, is_evaluation=is_evaluation
        )
        root = self.root
        # ImageFolder.__init__(
        #     self, root=root, transform=None, target_transform=None, is_valid_file=None
        # )
        # assert is_training ^ is_evaluation
        prefix = 'train' if is_training else 'val'
        map_txt = os.path.join(root, '..', f"{prefix}_map.txt")
        ImageFolder.__init__(
            self, root=root, transform=None, target_transform=None, is_valid_file=None, map_txt=map_txt
        )
        # self.n_classes = len(list(self.class_to_idx.keys()))
        self.n_classes = len(self.classes)
        setattr(opts, "model.classification.n_classes", self.n_classes)
        setattr(opts, "dataset.collate_fn_name_train", "imagenet_collate_fn")
        setattr(opts, "dataset.collate_fn_name_val", "imagenet_collate_fn")
        setattr(opts, "dataset.collate_fn_name_eval", "imagenet_collate_fn")

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add dataset-specific arguments to the parser."""
        group = parser.add_argument_group(
            title="".format(cls.__name__), description="".format(cls.__name__)
        )
        group.add_argument(
            "--dataset.imagenet_fast.crop-ratio",  # --dataset.imagenet.crop-ratio
            type=float,
            default=0.875,
            help="Crop ratio",
        )
        return parser

    def _training_transforms(self, size: Union[Tuple, int], *args, **kwargs):
        """
            Training data augmentation methods.
                Image --> RandomResizedCrop --> RandomHorizontalFlip --> Optional(AutoAugment or RandAugment)
                --> Tensor --> Optional(RandomErasing) --> Optional(MixUp) --> Optional(CutMix)

        .. note::
            1. AutoAugment, RandAugment and TrivialAugmentWide are mutually exclusive.
            2. Mixup and CutMix are applied on batches are implemented in trainer.
        """
        aug_list = [
            T.RandomResizedCrop(opts=self.opts, size=size),
            T.RandomHorizontalFlip(opts=self.opts),
        ]
        auto_augment = getattr(
            self.opts, "image_augmentation.auto_augment.enable", False
        )
        rand_augment = getattr(
            self.opts, "image_augmentation.rand_augment.enable", False
        )
        trivial_augment_wide = getattr(
            self.opts, "image_augmentation.trivial_augment_wide.enable", False
        )
        if bool(auto_augment) + bool(rand_augment) + bool(trivial_augment_wide) > 1:
            logger.error(
                "AutoAugment, RandAugment and TrivialAugmentWide are mutually exclusive. Use either of them, but not more than one"
            )
        elif auto_augment:
            aug_list.append(T.AutoAugment(opts=self.opts))
        elif rand_augment:
            if getattr(
                self.opts, "image_augmentation.rand_augment.use_timm_library", False
            ):
                aug_list.append(T.RandAugmentTimm(opts=self.opts))
            else:
                aug_list.append(T.RandAugment(opts=self.opts))
        elif trivial_augment_wide:
            aug_list.append(T.TrivialAugmentWide(opts=self.opts))

        aug_list.append(T.ToTensor(opts=self.opts))

        if getattr(self.opts, "image_augmentation.random_erase.enable", False):
            aug_list.append(T.RandomErasing(opts=self.opts))

        return T.Compose(opts=self.opts, img_transforms=aug_list)

    def _validation_transforms(self, size: Union[Tuple, int], *args, **kwargs):
        """
        Validation augmentation
            Image --> Resize --> CenterCrop --> ToTensor
        """
        aug_list = [
            T.Resize(opts=self.opts),
            T.CenterCrop(opts=self.opts),
            T.ToTensor(opts=self.opts),
        ]

        return T.Compose(opts=self.opts, img_transforms=aug_list)

    def _evaluation_transforms(self, size: Union[Tuple, int], *args, **kwargs):
        """Same as the validation_transforms"""
        return self._validation_transforms(size=size)

    def __getitem__(self, batch_indexes_tup: Tuple) -> Dict:
        """
        :param batch_indexes_tup: Tuple of the form (Crop_size_W, Crop_size_H, Image_ID)
        :return: dictionary containing input image, label, and sample_id.
        """
        crop_size_h, crop_size_w, img_index = batch_indexes_tup
        if self.is_training:
            transform_fn = self._training_transforms(size=(crop_size_h, crop_size_w))
        else:
            # same for validation and evaluation
            transform_fn = self._validation_transforms(size=(crop_size_h, crop_size_w))

        img_path, target = self.samples[img_index]

        input_img = self.read_image_pil(img_path)

        if input_img is None:
            # Sometimes images are corrupt
            # Skip such images
            logger.log("Img index {} is possibly corrupt.".format(img_index))
            input_tensor = torch.zeros(
                size=(3, crop_size_h, crop_size_w), dtype=self.img_dtype
            )
            target = -1
            data = {"image": input_tensor}
        else:
            data = {"image": input_img}
            data = transform_fn(data)

        data["samples"] = data.pop("image")
        data["targets"] = target
        data["sample_id"] = img_index

        return data

    def __len__(self) -> int:
        return len(self.samples)

    def __repr__(self) -> str:
        from utils.tensor_utils import image_size_from_opts

        im_h, im_w = image_size_from_opts(opts=self.opts)

        if self.is_training:
            transforms_str = self._training_transforms(size=(im_h, im_w))
        else:
            transforms_str = self._validation_transforms(size=(im_h, im_w))

        return "{}(\n\troot={}\n\tis_training={}\n\tsamples={}\n\tn_classes={}\n\ttransforms={}\n)".format(
            self.__class__.__name__,
            self.root,
            self.is_training,
            len(self.samples),
            self.n_classes,
            transforms_str,
        )
