# --------------------------------------------------------
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License
# Written by Guoqiang Wei
# --------------------------------------------------------

import os
import math

import torch
import torch.distributed as dist
from torch.utils.data import Sampler
from torchvision import transforms

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform, Mixup

from dataset_folder import ImageFolder


def build_dataloader(args):
    world_size = dist.get_world_size()
    global_rank = dist.get_rank()

    # build dataset
    dataset_train, args.num_classes = build_dataset(is_train=True, args=args)
    print(f'[{args.rank}/{global_rank}]: build train dataset')
    dataset_val, _ = build_dataset(is_train=False, args=args)
    print(f'[{args.rank}/{global_rank}]: build val dataset')

    # build sampler
    if args.repeated_aug:
        sampler_train = RepeatAugSampler(dataset_train, num_replicas=world_size, rank=global_rank, shuffle=True)
    else:
        sampler_train = torch.utils.data.DistributedSampler(dataset_train, num_replicas=world_size, rank=global_rank, shuffle=True)
    if args.dist_eval:
        sampler_val = torch.utils.data.DistributedSampler(dataset_val, num_replicas=world_size, rank=global_rank, shuffle=False)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    # build dataloader
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=int(2 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    # data augmentation
    mixup_fn = None
    if args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
            mode=args.mixup_mode,
            label_smoothing=args.smoothing,
            num_classes=args.num_classes
        )

    return dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    if args.data_set == 'IMNET':
        prefix = 'train' if is_train else 'val'
        root = os.path.join(args.data_path, prefix)
        dataset = ImageFolder(
            root,
            transform=transform,
            map_txt=os.path.join(args.map_path, f"{prefix}_map.txt")
        )
        num_classes = 1000
    else:
        assert NotImplementedError(f'{args.data_set} not implemented')

    return dataset, num_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            transform.transforms[0] = transforms.RandomCrop(args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * args.input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)


class RepeatAugSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset for distributed,
    with repeated augmentation.
    It ensures that different each augmented version of a sample will be visible to a
    different process (GPU). Heavily based on torch.utils.data.DistributedSampler

    This sampler was taken from https://github.com/facebookresearch/deit/blob/0c4b8f60/samplers.py
    Used in
    Copyright (c) 2015-present, Facebook, Inc.
    """

    def __init__(
            self,
            dataset,
            num_replicas=None,
            rank=None,
            shuffle=True,
            num_repeats=3,
            selected_round=256,
            selected_ratio=0,
    ):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.num_repeats = num_repeats
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * num_repeats * 1. / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        selected_ratio = selected_ratio or num_replicas
        if selected_round:
            self.num_selected_samples = int(
                math.floor(len(self.dataset) // selected_round * selected_round / selected_ratio)
            )
        else:
            self.num_selected_samples = int(math.ceil(len(self.dataset) / selected_ratio))

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices = [ele for ele in indices for i in range(3)]
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices[:self.num_selected_samples])

    def __len__(self):
        return self.num_selected_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
