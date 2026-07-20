# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from torchvision
# ------------------------------------------------------------------------

"""
Copy-Paste from torchvision, but add utility of caching images on memory
"""
import torch 
from torchvision.datasets.vision import VisionDataset
from PIL import Image
import os
import os.path
import tqdm
from io import BytesIO


class CocoDetection(VisionDataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(self, root, annFile, transform=None, target_transform=None, transforms=None,
                 cache_mode=False, local_rank=0, local_size=1):
        super(CocoDetection, self).__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.cache_mode = cache_mode
        self.local_rank = local_rank
        self.local_size = local_size
        if cache_mode:
            self.cache = {}
            self.cache_images()

    def cache_images(self):
        self.cache = {}
        for index, img_id in zip(tqdm.trange(len(self.ids)), self.ids):
            if index % self.local_size != self.local_rank:
                continue
            path = self.coco.loadImgs(img_id)[0]['filename']
            with open(os.path.join(self.root, path), 'rb') as f:
                self.cache[path] = f.read()

    def get_image(self, path):
        if self.cache_mode:
            if path not in self.cache.keys():
                with open(os.path.join(self.root, path), 'rb') as f:
                    self.cache[path] = f.read()
            return Image.open(BytesIO(self.cache[path])).convert('RGB')
        return Image.open(os.path.join(self.root, path)).convert('RGB')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name_IR']
        img = self.get_image(path)
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.ids)

# %% Modification to capture both spectra 
import os
from typing import Any, Callable, List, Optional, Tuple

import torch
import torch.utils.data as data

from torchvision.utils import _log_api_usage_once

class CocoDetection_RGBT(VisionDataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(self, root_rgb, root_thermal, ann_file, transform=None, target_transform=None, transforms=None,
                 cache_mode=False, local_rank=0, local_size=1):
        super(CocoDetection_RGBT, self).__init__(root_rgb, transforms, transform, target_transform)
        self.root_rgb = root_rgb
        self.root_thermal = root_thermal
        self.ann_file = ann_file 
        from pycocotools.coco import COCO
        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.cache_mode = cache_mode
        self.local_rank = local_rank
        self.local_size = local_size
        
        if cache_mode:
            self.cache = {}
            self.cache_images()
        
    
    def cache_images(self):
        self.cache = {}
        for index, img_id in zip(tqdm.trange(len(self.ids)), self.ids):
            if index % self.local_size != self.local_rank:
                continue
            path_thermal = self.coco.loadImgs(img_id)[0]['file_name_IR']
            with open(os.path.join(self.root_thermal, path_thermal), 'rb') as f:
                self.cache[path_thermal] = f.read()
            path_rgb = self.coco.loadImgs(img_id)[0]['file_name_RGB']
            with open(os.path.join(self.root_rgb, path_rgb), 'rb') as f:
                self.cache[path_rgb] = f.read()
                
    def get_image(self, path, spectrum):
        if spectrum == 'ir': 
            if self.cache_mode:
                if path not in self.cache.keys():
                    with open(os.path.join(self.root_thermal, path), 'rb') as f:
                        self.cache[path] = f.read()
                return Image.open(BytesIO(self.cache[path])).convert('RGB')
            return Image.open(os.path.join(self.root_thermal, path)).convert('RGB')
        if spectrum == 'v': 
            if self.cache_mode:
                if path not in self.cache.keys():
                    with open(os.path.join(self.root_rgb, path), 'rb') as f:
                        self.cache[path] = f.read()
                return Image.open(BytesIO(self.cache[path])).convert('RGB')
            return Image.open(os.path.join(self.root_rgb, path)).convert('RGB')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        path_thermal = coco.loadImgs(img_id)[0]['file_name_IR']
        path_RGB = coco.loadImgs(img_id)[0]['file_name_RGB']

        img_thermal = self.get_image(path_thermal, 'ir')
        img_RGB = self.get_image(path_RGB, 'v')

        if self.transforms is not None:
            img_RGB, img_thermal, target = self.transforms(img_RGB, img_thermal, target)

        return img_RGB, img_thermal, target

    def __len__(self):
        return len(self.ids)
        
class CocoDetection_RGBT_LLVIP(VisionDataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(self, root_rgb, root_thermal, ann_file, transform=None, target_transform=None, transforms=None,
                 cache_mode=False, local_rank=0, local_size=1):
        super(CocoDetection_RGBT_LLVIP, self).__init__(root_rgb, transforms, transform, target_transform)
        self.root_rgb = root_rgb
        self.root_thermal = root_thermal
        self.ann_file = ann_file 
        from pycocotools.coco import COCO
        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.cache_mode = cache_mode
        self.local_rank = local_rank
        self.local_size = local_size
        
        if cache_mode:
            self.cache = {}
            self.cache_images()
        
    
    def cache_images(self):
        self.cache = {}
        for index, img_id in zip(tqdm.trange(len(self.ids)), self.ids):
            if index % self.local_size != self.local_rank:
                continue
            path_thermal = self.coco.loadImgs(img_id)[0]['file_name']
            with open(os.path.join(self.root_thermal, path_thermal), 'rb') as f:
                self.cache[path_thermal] = f.read()
            path_rgb = self.coco.loadImgs(img_id)[0]['file_name']
            with open(os.path.join(self.root_rgb, path_rgb), 'rb') as f:
                self.cache[path_rgb] = f.read()
                
    def get_image(self, path, spectrum):
        if spectrum == 'ir': 
            if self.cache_mode:
                if path not in self.cache.keys():
                    with open(os.path.join(self.root_thermal, path), 'rb') as f:
                        self.cache[path] = f.read()
                return Image.open(BytesIO(self.cache[path])).convert('RGB')
            return Image.open(os.path.join(self.root_thermal, path)).convert('RGB')
        if spectrum == 'v': 
            if self.cache_mode:
                if path not in self.cache.keys():
                    with open(os.path.join(self.root_rgb, path), 'rb') as f:
                        self.cache[path] = f.read()
                return Image.open(BytesIO(self.cache[path])).convert('RGB')
            return Image.open(os.path.join(self.root_rgb, path)).convert('RGB')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        #print(index)
        img_id = self.ids[index]
        #print(img_id)
        ann_ids = coco.getAnnIds(imgIds=img_id)
        #print(ann_ids) 
        target = coco.loadAnns(ann_ids)
        #print(target)
        # LLVIP_style 
        path_RGB = coco.loadImgs(img_id)[0]['file_name']
        path_thermal = coco.loadImgs(img_id)[0]['file_name']
        
        #path_RGB = coco.loadImgs(img_id)[0]['file_name'] # base value 

        img_thermal = self.get_image(path_thermal, 'ir')
        img_RGB = self.get_image(path_RGB, 'v')

        if self.transforms is not None:
            img_RGB, img_thermal, target = self.transforms(img_RGB, img_thermal, target)

        return img_RGB, img_thermal, target

    def __len__(self):
        return len(self.ids)
        
class CocoDetection_RGBT_FLIR(VisionDataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(self, root_rgb, root_thermal, ann_file, transform=None, target_transform=None, transforms=None,
                 cache_mode=False, local_rank=0, local_size=1):
        super(CocoDetection_RGBT_FLIR, self).__init__(root_rgb, transforms, transform, target_transform)
        self.root_rgb = root_rgb
        self.root_thermal = root_thermal
        self.ann_file = ann_file 
        from pycocotools.coco import COCO
        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.cache_mode = cache_mode
        self.local_rank = local_rank
        self.local_size = local_size
        
        if cache_mode:
            self.cache = {}
            self.cache_images()
        
    
    def cache_images(self):
        self.cache = {}
        for index, img_id in zip(tqdm.trange(len(self.ids)), self.ids):
            if index % self.local_size != self.local_rank:
                continue
            path_thermal = self.coco.loadImgs(img_id)[0]['file_name_IR']
            with open(os.path.join(self.root_thermal, path_thermal), 'rb') as f:
                self.cache[path_thermal] = f.read()
            path_rgb = self.coco.loadImgs(img_id)[0]['file_name_RGB']
            with open(os.path.join(self.root_rgb, path_rgb), 'rb') as f:
                self.cache[path_rgb] = f.read()
                
    def get_image(self, path, spectrum):
        if spectrum == 'ir': 
            if self.cache_mode:
                if path not in self.cache.keys():
                    with open(os.path.join(self.root_thermal, path), 'rb') as f:
                        self.cache[path] = f.read()
                return Image.open(BytesIO(self.cache[path])).convert('RGB')
            return Image.open(os.path.join(self.root_thermal, path)).convert('RGB')
        if spectrum == 'v': 
            if self.cache_mode:
                if path not in self.cache.keys():
                    with open(os.path.join(self.root_rgb, path), 'rb') as f:
                        self.cache[path] = f.read()
                return Image.open(BytesIO(self.cache[path])).convert('RGB')
            return Image.open(os.path.join(self.root_rgb, path)).convert('RGB')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        #print(index)
        img_id = self.ids[index]
        #print(img_id)
        ann_ids = coco.getAnnIds(imgIds=img_id)
        #print(ann_ids) 
        target = coco.loadAnns(ann_ids)
        #print(target)

        # FLIR style 
        path_RGB = coco.loadImgs(img_id)[0]['file_name_RGB']
        path_thermal = coco.loadImgs(img_id)[0]['file_name_IR']
        
        #path_RGB = coco.loadImgs(img_id)[0]['file_name'] # base value 

        img_thermal = self.get_image(path_thermal, 'ir')
        img_RGB = self.get_image(path_RGB, 'v')

        if self.transforms is not None:
            img_RGB, img_thermal, target = self.transforms(img_RGB, img_thermal, target)

        return img_RGB, img_thermal, target

    def __len__(self):
        return len(self.ids)


# =============================================================================
# KAIST Dataset — Multispectral Pedestrian Detection (Section 4.1.1, Table 3)
# =============================================================================
# KAIST: large-scale multispectral pedestrian detection dataset from driving
# environments. 7,601 training pairs, 2,252 test pairs (1,455 day + 797 night).
# Single class: "person". Uses improved annotations from Zhang et al. 2019.
#
# Expected COCO directory structure:
#   root/
#   ├── annotations/
#   │   ├── train.json      # "file_name_RGB" + "file_name_IR" keys
#   │   └── test.json
#   ├── train_RGB/          # visible images
#   ├── train_thermal/      # infrared images
#   ├── test_RGB/
#   └── test_thermal/

class CocoDetection_RGBT_KAIST(VisionDataset):
    """
    MS Coco Detection Dataset for KAIST multispectral pedestrian detection.

    Each annotation entry references both spectra:
        - "file_name_RGB" (or "file_name" fallback) — visible image
        - "file_name_IR"  — thermal infrared image
    """

    def __init__(self, root_rgb, root_thermal, ann_file, transform=None,
                 target_transform=None, transforms=None,
                 cache_mode=False, local_rank=0, local_size=1):
        super(CocoDetection_RGBT_KAIST, self).__init__(
            root_rgb, transforms, transform, target_transform)
        self.root_rgb = root_rgb
        self.root_thermal = root_thermal
        self.ann_file = ann_file
        from pycocotools.coco import COCO
        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.cache_mode = cache_mode
        self.local_rank = local_rank
        self.local_size = local_size

        if cache_mode:
            self.cache = {}
            self.cache_images()

    def cache_images(self):
        """Pre-load all images into memory for faster access."""
        self.cache = {}
        for index, img_id in zip(tqdm.trange(len(self.ids)), self.ids):
            if index % self.local_size != self.local_rank:
                continue
            path_rgb = self._get_rgb_path(img_id)
            with open(os.path.join(self.root_rgb, path_rgb), 'rb') as f:
                self.cache[path_rgb] = f.read()
            path_thermal = self._get_thermal_path(img_id)
            with open(os.path.join(self.root_thermal, path_thermal), 'rb') as f:
                self.cache[path_thermal] = f.read()

    def _get_rgb_path(self, img_id):
        """Resolve visible image path from annotation metadata."""
        img_info = self.coco.loadImgs(img_id)[0]
        # KAIST COCO annotations may use "file_name_RGB" or fall back to "file_name"
        return img_info.get('file_name_RGB', img_info['file_name'])

    def _get_thermal_path(self, img_id):
        """Resolve thermal image path from annotation metadata."""
        img_info = self.coco.loadImgs(img_id)[0]
        # KAIST COCO annotations use "file_name_IR" for the thermal image
        return img_info.get('file_name_IR', img_info['file_name'])

    def get_image(self, path, spectrum):
        """
        Load a single image by spectrum type.

        Args:
            path (str): Relative image path.
            spectrum (str): 'ir' for thermal infrared, 'v' for visible.

        Returns:
            PIL.Image: RGB-mode image.
        """
        root = self.root_thermal if spectrum == 'ir' else self.root_rgb
        if self.cache_mode:
            if path not in self.cache:
                with open(os.path.join(root, path), 'rb') as f:
                    self.cache[path] = f.read()
            return Image.open(BytesIO(self.cache[path])).convert('RGB')
        return Image.open(os.path.join(root, path)).convert('RGB')

    def __getitem__(self, index):
        """
        Returns:
            tuple: (img_RGB, img_thermal, target) — target is COCO annotation list.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        path_RGB = self._get_rgb_path(img_id)
        path_thermal = self._get_thermal_path(img_id)

        img_thermal = self.get_image(path_thermal, 'ir')
        img_RGB = self.get_image(path_RGB, 'v')

        if self.transforms is not None:
            img_RGB, img_thermal, target = self.transforms(img_RGB, img_thermal, target)

        return img_RGB, img_thermal, target

    def __len__(self):
        return len(self.ids)


# =============================================================================
# GIR Dataset — Ground-based Infrared-Visible (Section 4.1.1, Table 4)
# =============================================================================
# GIR: custom dataset built from RGBT210 video sequences for this paper.
# 5,105 image pairs (4,084 train / 1,021 test, 8:2 split).
# 5 classes: person, dog, car, bicycle, motorcycle.
# Features: varied scales, occlusions, complex backgrounds, small IR targets.
# Rigorous annotation: tight bounding boxes, cross-verification by researchers.
#
# Expected COCO directory structure:
#   root/
#   ├── annotations/
#   │   ├── train.json      # "file_name_RGB" + "file_name_IR"
#   │   └── val.json
#   ├── train_RGB/          # visible training images
#   ├── train_thermal/      # infrared training images
#   ├── val_RGB/
#   └── val_thermal/

class CocoDetection_RGBT_GIR(VisionDataset):
    """
    MS Coco Detection Dataset for GIR multispectral object detection.

    Each annotation entry references both spectra:
        - "file_name_RGB" — visible image
        - "file_name_IR"  — thermal infrared image
    """

    def __init__(self, root_rgb, root_thermal, ann_file, transform=None,
                 target_transform=None, transforms=None,
                 cache_mode=False, local_rank=0, local_size=1):
        super(CocoDetection_RGBT_GIR, self).__init__(
            root_rgb, transforms, transform, target_transform)
        self.root_rgb = root_rgb
        self.root_thermal = root_thermal
        self.ann_file = ann_file
        from pycocotools.coco import COCO
        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.cache_mode = cache_mode
        self.local_rank = local_rank
        self.local_size = local_size

        if cache_mode:
            self.cache = {}
            self.cache_images()

    def cache_images(self):
        """Pre-load all images into memory for faster access."""
        self.cache = {}
        for index, img_id in zip(tqdm.trange(len(self.ids)), self.ids):
            if index % self.local_size != self.local_rank:
                continue
            path_rgb = self._get_rgb_path(img_id)
            with open(os.path.join(self.root_rgb, path_rgb), 'rb') as f:
                self.cache[path_rgb] = f.read()
            path_thermal = self._get_thermal_path(img_id)
            with open(os.path.join(self.root_thermal, path_thermal), 'rb') as f:
                self.cache[path_thermal] = f.read()

    def _get_rgb_path(self, img_id):
        img_info = self.coco.loadImgs(img_id)[0]
        return img_info.get('file_name_RGB', img_info['file_name'])

    def _get_thermal_path(self, img_id):
        img_info = self.coco.loadImgs(img_id)[0]
        return img_info.get('file_name_IR', img_info['file_name'])

    def get_image(self, path, spectrum):
        """
        Load a single image by spectrum type.

        Args:
            path (str): Relative image path.
            spectrum (str): 'ir' for thermal infrared, 'v' for visible.

        Returns:
            PIL.Image: RGB-mode image.
        """
        root = self.root_thermal if spectrum == 'ir' else self.root_rgb
        if self.cache_mode:
            if path not in self.cache:
                with open(os.path.join(root, path), 'rb') as f:
                    self.cache[path] = f.read()
            return Image.open(BytesIO(self.cache[path])).convert('RGB')
        return Image.open(os.path.join(root, path)).convert('RGB')

    def __getitem__(self, index):
        """
        Returns:
            tuple: (img_RGB, img_thermal, target) — target is COCO annotation list.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        path_RGB = self._get_rgb_path(img_id)
        path_thermal = self._get_thermal_path(img_id)

        img_thermal = self.get_image(path_thermal, 'ir')
        img_RGB = self.get_image(path_RGB, 'v')

        if self.transforms is not None:
            img_RGB, img_thermal, target = self.transforms(img_RGB, img_thermal, target)

        return img_RGB, img_thermal, target

    def __len__(self):
        return len(self.ids)

