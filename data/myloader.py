# 2021.06.15-Changed for implementation of TNT model
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
""" Loader Factory, Fast Collate, CUDA Prefetcher

Prefetcher and Fast Collate inspired by NVIDIA APEX example at
https://github.com/NVIDIA/apex/commit/d5e2bb4bdeedd27b1dfaf5bb2b24d6c000dee9be#diff-cf86c282ff7fba81fad27a559379d5bf

Hacked together by / Copyright 2020 Ross Wightman
"""

import torch.utils.data
import torch.distributed as dist
import numpy as np

from timm.data.transforms_factory import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data.distributed_sampler import OrderedDistributedSampler
from timm.data.random_erasing import RandomErasing
from timm.data.mixup import FastCollateMixup
from timm.data.loader import fast_collate, PrefetchLoader, MultiEpochsDataLoader

from .rasampler import RASampler
from torchvision import datasets

import os
import sys
import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, '../'))

from data_utils.datasets import *

_DATASET_CATALOG = {
    ### preparing for meta training
    "sun397": SUN397,
    "stl10": STL10,
    "fru92": Fru92Dataset,
    "veg200": Veg200Dataset,
    "oxford-iiit-pets": OxfordIIITPet,
    "eurosat": EuroSAT,
    ### preparing for task adapting
    "cifar10": CIFAR10,
    "cifar100": CIFAR100,
    "cub200": CUB200Dataset,
    "nabirds": NabirdsDataset,
    "oxford-flowers": FlowersDataset,
    "stanford-dogs": DogsDataset,
    "stanford-cars": CarsDataset,
    "fgvc-aircraft": AircraftDataset,
    "food101": Food101,
    "dtd": DTD,
    "svhn": SVHN,
    "gtsrb": GTSRB
}

_DATA_DIR_CATALOG = {
    ### preparing for meta training
    "sun397": "torchvision_dataset/",
    "stl10": "torchvision_dataset/",
    "fru92": "finegrained_dataset/vegfru-dataset",
    "veg200": "finegrained_dataset/vegfru-dataset",
    "oxford-iiit-pets": "torchvision_dataset/",
    "eurosat": "torchvision_dataset/",
    ### preparing for task adapting
    "cifar10": "torchvision_dataset/",
    "cifar100": "torchvision_dataset/",
    "cub200": "FGVC/CUB_200_2011/",
    "nabirds": "FGVC/nabirds/",
    "oxford-flowers": "FGVC/OxfordFlower/",
    "stanford-dogs": "FGVC/Stanford-dogs/",
    "stanford-cars": "FGVC/Stanford-cars/",
    "fgvc-aircraft": "FGVC/fgvc-aircraft-2013b/",
    "food101": "torchvision_dataset/",
    "dtd": "torchvision_dataset/",
    "svhn": "torchvision_dataset/",
    "gtsrb": "torchvision_dataset/"
}

_NUM_CLASSES_CATALOG = {
    ### preparing for meta training
    "sun397": 397,
    "stl10": 10,
    "fru92": 92,
    "veg200": 200,
    "oxford-iiit-pets": 37,
    "eurosat": 10,
    ### preparing for task adapting
    "cifar10": 10,
    "cifar100": 100,
    "cub200": 200,
    "nabirds": 555,
    "oxford-flowers": 102,
    "stanford-dogs": 120,
    "stanford-cars": 196,
    "fgvc-aircraft": 100,
    "food101": 101,
    "dtd": 47,
    "svhn": 10,
    "gtsrb": 43
}


def get_dataset_classes(dataset):
    """Given a dataset, return the name list of dataset classes."""
    if hasattr(dataset, "classes"):
        return dataset.classes
    elif hasattr(dataset, "_class_ids"):
        return dataset._class_ids
    elif hasattr(dataset, "labels"):
        return dataset.labels
    else:
        raise NotImplementedError


def _construct_loader(args, dataset, split, batch_size, shuffle, drop_last):
    """Constructs the data loader for the given dataset."""
    dataset_name = dataset
    # Construct the dataset
    if dataset_name.startswith("vtab-"):
        # import the tensorflow here only if needed
        args.data_dir = os.path.join(args.base_dir, "VTAB/")
        from data_utils.datasets.tf_dataset import TFDataset
        dataset = TFDataset(args, split)
    else:
        assert (
            dataset_name in _DATASET_CATALOG.keys()
        ), "Dataset '{}' not supported".format(dataset_name)
        args.data_dir = os.path.join(args.base_dir, _DATA_DIR_CATALOG[dataset_name])
        #print('name:',dataset_name,str(_DATASET_CATALOG[dataset_name]), split)
        dataset = _DATASET_CATALOG[dataset_name](args, split)

    return dataset


def construct_train_dataset(args, dataset=None):
    """Train loader wrapper."""
    if args.distributed:
        drop_last = True
    else:
        drop_last = False
    return _construct_loader(
        args=args,
        split="train",
        batch_size=int(args.batch_size / args.num_gpu),
        shuffle=True,
        drop_last=drop_last,
        dataset=dataset if dataset else args.dataset
    )


def construct_val_dataset(args, dataset=None, batch_size=None):
    """Validation loader wrapper."""
    if batch_size is None:
        bs = int(args.batch_size / args.num_gpu)
    else:
        bs = batch_size
    return _construct_loader(
        args=args,
        split="val",
        batch_size=bs,
        shuffle=False,
        drop_last=False,
        dataset=dataset if dataset else args.dataset
    )


def construct_test_dataset(args, dataset=None):
    """Test loader wrapper."""
    return _construct_loader(
        args=args,
        split="test",
        batch_size=int(args.batch_size / args.num_gpu),
        shuffle=False,
        drop_last=False,
        dataset=dataset if dataset else args.dataset
    )


def shuffle(loader, cur_epoch):
    """"Shuffles the data."""
    assert isinstance(
        loader.sampler, (RandomSampler, DistributedSampler)
    ), "Sampler type '{}' not supported".format(type(loader.sampler))
    # RandomSampler handles shuffling automatically
    if isinstance(loader.sampler, DistributedSampler):
        # DistributedSampler shuffles data based on epoch
        loader.sampler.set_epoch(cur_epoch)


def _dataset_class_num(dataset_name):
    """Query to obtain class nums of datasets."""
    return _NUM_CLASSES_CATALOG[dataset_name]



def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


# def create_loader(
#         dataset,
#         input_size,
#         batch_size,
#         is_training=False,
#         use_prefetcher=True,
#         no_aug=False,
#         re_prob=0.,
#         re_mode='const',
#         re_count=1,
#         re_split=False,
#         scale=None,
#         ratio=None,
#         hflip=0.5,
#         vflip=0.,
#         color_jitter=0.4,
#         auto_augment=None,
#         num_aug_splits=0,
#         interpolation='bilinear',
#         mean=IMAGENET_DEFAULT_MEAN,
#         std=IMAGENET_DEFAULT_STD,
#         num_workers=1,
#         distributed=False,
#         crop_pct=None,
#         collate_fn=None,
#         pin_memory=False,
#         fp16=False,
#         tf_preprocessing=False,
#         use_multi_epochs_loader=False,
#         repeated_aug=False
# ):
#     re_num_splits = 0
#     if re_split:
#         # apply RE to second half of batch if no aug split otherwise line up with aug split
#         re_num_splits = num_aug_splits or 2
#     dataset.transform = create_transform(
#         input_size,
#         is_training=is_training,
#         use_prefetcher=use_prefetcher,
#         no_aug=no_aug,
#         scale=scale,
#         ratio=ratio,
#         hflip=hflip,
#         vflip=vflip,
#         color_jitter=color_jitter,
#         auto_augment=auto_augment,
#         interpolation=interpolation,
#         mean=mean,
#         std=std,
#         crop_pct=crop_pct,
#         tf_preprocessing=tf_preprocessing,
#         re_prob=re_prob,
#         re_mode=re_mode,
#         re_count=re_count,
#         re_num_splits=re_num_splits,
#         separate=num_aug_splits > 0,
#     )
#
#     sampler = None
#     if distributed:
#         if is_training:
#             if repeated_aug:
#                 print('using repeated_aug')
#                 num_tasks = get_world_size()
#                 global_rank = get_rank()
#                 sampler = RASampler(
#                     dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
#                 )
#             else:
#                 sampler = torch.utils.data.distributed.DistributedSampler(dataset)
#         else:
#             # This will add extra duplicate entries to result in equal num
#             # of samples per-process, will slightly alter validation results
#             sampler = OrderedDistributedSampler(dataset)
#     else:
#         if is_training and repeated_aug:
#             print('using repeated_aug')
#             num_tasks = get_world_size()
#             global_rank = get_rank()
#             sampler = RASampler(
#                     dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
#                 )
#
#     if collate_fn is None:
#         collate_fn = fast_collate if use_prefetcher else torch.utils.data.dataloader.default_collate
#
#     loader_class = torch.utils.data.DataLoader
#
#     if use_multi_epochs_loader:
#         loader_class = MultiEpochsDataLoader
#
#     loader = loader_class(
#         dataset,
#         batch_size=batch_size,
#         shuffle=sampler is None and is_training,
#         num_workers=num_workers,
#         sampler=sampler,
#         collate_fn=collate_fn,
#         pin_memory=pin_memory,
#         drop_last=is_training,
#     )
#     if use_prefetcher:
#         prefetch_re_prob = re_prob if is_training and not no_aug else 0.
#         loader = PrefetchLoader(
#             loader,
#             mean=mean,
#             std=std,
#             fp16=fp16,
#             re_prob=prefetch_re_prob,
#             re_mode=re_mode,
#             re_count=re_count,
#             re_num_splits=re_num_splits
#         )
#
#     return loader


def create_loader(
        dataset,
        input_size,
        batch_size,
        is_training=False,
        use_prefetcher=True,
        no_aug=False,
        re_prob=0.,
        re_mode='const',
        re_count=1,
        re_split=False,
        scale=None,
        ratio=None,
        hflip=0.5,
        vflip=0.,
        color_jitter=0.4,
        auto_augment=None,
        num_aug_splits=0,
        interpolation='bilinear',
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        num_workers=1,
        distributed=False,
        crop_pct=None,
        collate_fn=None,
        pin_memory=False,
        fp16=False,
        tf_preprocessing=False,
        use_multi_epochs_loader=False,
        repeated_aug=False,
        args=None  # 新增提示学习任务的参数
):
    re_num_splits = 0

    dataset = construct_train_dataset(args) if is_training else construct_val_dataset(args)

    if re_split:
        # apply RE to second half of batch if no aug split otherwise line up with aug split
        re_num_splits = num_aug_splits or 2
        dataset.transform = create_transform(
            input_size,
            is_training=is_training,
            use_prefetcher=use_prefetcher,
            no_aug=no_aug,
            scale=scale,
            ratio=ratio,
            hflip=hflip,
            vflip=vflip,
            color_jitter=color_jitter,
            auto_augment=auto_augment,
            interpolation=interpolation,
            mean=mean,
            std=std,
            crop_pct=crop_pct,
            tf_preprocessing=tf_preprocessing,
            re_prob=re_prob,
            re_mode=re_mode,
            re_count=re_count,
            re_num_splits=re_num_splits,
            separate=num_aug_splits > 0,
        )

    sampler = None
    if distributed:
        if is_training:
            if repeated_aug:
                print('using repeated_aug')
                num_tasks = get_world_size()
                global_rank = get_rank()
                sampler = RASampler(
                    dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
                )
            else:
                sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            # This will add extra duplicate entries to result in equal num
            # of samples per-process, will slightly alter validation results
            sampler = OrderedDistributedSampler(dataset)
    else:
        if is_training and repeated_aug:
            print('using repeated_aug')
            num_tasks = get_world_size()
            global_rank = get_rank()
            sampler = RASampler(
                    dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
                )

    if collate_fn is None:
        collate_fn = fast_collate if use_prefetcher else torch.utils.data.dataloader.default_collate

    loader_class = torch.utils.data.DataLoader

    if use_multi_epochs_loader:
        loader_class = MultiEpochsDataLoader

    loader = loader_class(
        dataset,
        batch_size=batch_size,
        shuffle=sampler is None and is_training,
        num_workers=num_workers,
        sampler=sampler,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=is_training,
    )
    if use_prefetcher:
        prefetch_re_prob = re_prob if is_training and not no_aug else 0.
        loader = PrefetchLoader(
            loader,
            mean=mean,
            std=std,
            fp16=fp16,
            re_prob=prefetch_re_prob,
            re_mode=re_mode,
            re_count=re_count,
            re_num_splits=re_num_splits
        )

    return loader