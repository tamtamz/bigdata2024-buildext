# Copyright (c) OpenMMLab. All rights reserved.
import copy
from pathlib import Path
from typing import List, Union
import json
from collections import defaultdict

from mmengine.fileio import get_local_path

from mmdet.registry import DATASETS
from .base_det_dataset import BaseDetDataset


@DATASETS.register_module()
class BuildingDataset(BaseDetDataset):

    METAINFO = {
        'classes':
        ('building',),
        # palette is a list of color tuples, which is used for visualization.
        'palette':
        [(220, 20, 60)]
    }

    img_directory = 'images'
    anno_directory = 'annotations'


    def __init__(self, data_root, anno_files, **kwargs):
        self.anno_files = anno_files

        super().__init__(data_root=data_root, **kwargs)

    def load_data_list(self) -> List[dict]:
        data_root = Path(self.data_root)
        
        img_id = 0
        data_list = []
        for anno_file in self.anno_files:
            anno_path = data_root / self.anno_directory / anno_file
            with anno_path.open() as f:
                annos = json.load(f)

            img_annos = defaultdict(list)
            for instance in annos['annotations']:
                img_annos[instance['image_id']].append(instance)

            for img in annos['images']:
                instances = []
                for instance in img_annos[img['id']]:
                    if all(len(m) < 6 for m in instance['segmentation']):
                        continue
                    
                    bbox = instance['bbox']
                    instance = {
                        'bbox': [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]],
                        'bbox_label': 0,
                        'mask': instance['segmentation'],
                        'ignore_flag': 0
                    }
                    instances.append(instance)

                data_info = {
                    'img_id': img_id,
                    'img_path': str(data_root / self.img_directory / anno_path.stem / img['file_name']),
                    'width': img['width'],
                    'height': img['height'],
                    'instances': instances
                }

                data_list.append(data_info)
                
                img_id += 1

        return data_list

    def filter_data(self) -> List[dict]:
        """Filter annotations according to filter_cfg.

        Returns:
            List[dict]: Filtered results.
        """
        if self.test_mode:
            return self.data_list

        if self.filter_cfg is None:
            return self.data_list

        filter_empty_gt = self.filter_cfg.get('filter_empty_gt', False)
        min_size = self.filter_cfg.get('min_size', 0)

        # obtain images that contain annotation
        ids_with_ann = set(data_info['img_id'] for data_info in self.data_list)
        # obtain images that contain annotations of the required categories
        ids_in_cat = set()
        for i, class_id in enumerate(self.cat_ids):
            ids_in_cat |= set(self.cat_img_map[class_id])
        # merge the image id sets of the two conditions and use the merged set
        # to filter out images if self.filter_empty_gt=True
        ids_in_cat &= ids_with_ann

        valid_data_infos = []
        for i, data_info in enumerate(self.data_list):
            img_id = data_info['img_id']
            width = data_info['width']
            height = data_info['height']
            if filter_empty_gt and img_id not in ids_in_cat:
                continue
            if min(width, height) >= min_size:
                valid_data_infos.append(data_info)

        return valid_data_infos
