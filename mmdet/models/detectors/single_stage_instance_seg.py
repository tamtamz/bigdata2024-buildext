# Copyright (c) OpenMMLab. All rights reserved.
import copy
import warnings

import numpy as np
import torch

from mmdet.registry import MODELS
from .base import BaseDetector

INF = 1e8


@MODELS.register_module()
class SingleStageInstanceSegmentor(BaseDetector):
    """Base class for single-stage instance segmentors."""

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 mask_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):

        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        super(SingleStageInstanceSegmentor, self).__init__(init_cfg=init_cfg)
        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        else:
            self.neck = None
        if bbox_head is not None:
            bbox_head.update(train_cfg=copy.deepcopy(train_cfg))
            bbox_head.update(test_cfg=copy.deepcopy(test_cfg))
            self.bbox_head = MODELS.build(bbox_head)
        else:
            self.bbox_head = None

        assert mask_head, f'`mask_head` must ' \
                          f'be implemented in {self.__class__.__name__}'
        mask_head.update(train_cfg=copy.deepcopy(train_cfg))
        mask_head.update(test_cfg=copy.deepcopy(test_cfg))
        self.mask_head = MODELS.build(mask_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def extract_feat(self, img):
        """Directly extract features from the backbone and neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        raise NotImplementedError(
            f'`forward_dummy` is not implemented in {self.__class__.__name__}')

    def forward_train(self,
                      img,
                      img_metas,
                      gt_masks,
                      gt_labels,
                      gt_bboxes=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """
        Args:
            img (Tensor): Input images of shape (B, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_masks (list[:obj:`BitmapMasks`] | None) : The segmentation
                masks for each box.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes (list[Tensor]): Each item is the truth boxes
                of each image in [tl_x, tl_y, br_x, br_y] format.
                Default: None.
            gt_bboxes_ignore (list[Tensor] | None): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        gt_masks = [
            gt_mask.to_tensor(dtype=torch.bool, device=img.device)
            for gt_mask in gt_masks
        ]
        x = self.extract_feat(img)
        losses = dict()

        # CondInst and YOLACT have bbox_head
        if self.bbox_head:
            # bbox_head_preds is a tuple
            bbox_head_preds = self.bbox_head(x)
            # positive_infos is a list of obj:`InstanceData`
            # It contains the information about the positive samples
            # CondInst, YOLACT
            det_losses, positive_infos = self.bbox_head.loss(
                *bbox_head_preds,
                gt_bboxes=gt_bboxes,
                gt_labels=gt_labels,
                gt_masks=gt_masks,
                img_metas=img_metas,
                gt_bboxes_ignore=gt_bboxes_ignore,
                **kwargs)
            losses.update(det_losses)
        else:
            positive_infos = None

        mask_loss = self.mask_head.forward_train(
            x,
            gt_labels,
            gt_masks,
            img_metas,
            positive_infos=positive_infos,
            gt_bboxes=gt_bboxes,
            gt_bboxes_ignore=gt_bboxes_ignore,
            **kwargs)
        # avoid loss override
        assert not set(mask_loss.keys()) & set(losses.keys())

        losses.update(mask_loss)
        return losses

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test-time augmentation.

        Args:
            img (torch.Tensor): Images with shape (B, C, H, W).
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list(tuple): Formatted bbox and mask results of multiple \
                images. The outer list corresponds to each image. \
                Each tuple contains two type of results of single image:

                - bbox_results (list[np.ndarray]): BBox results of
                  single image. The list corresponds to each class.
                  each ndarray has a shape (N, 5), N is the number of
                  bboxes with this category, and last dimension
                  5 arrange as (x1, y1, x2, y2, scores).
                - mask_results (list[np.ndarray]): Mask results of
                  single image. The list corresponds to each class.
                  each ndarray has a shape (N, img_h, img_w), N
                  is the number of masks with this category.
        """
        feat = self.extract_feat(img)
        if self.bbox_head:
            outs = self.bbox_head(feat)
            # results_list is list[obj:`InstanceData`]
            results_list = self.bbox_head.get_results(
                *outs, img_metas=img_metas, cfg=self.test_cfg, rescale=rescale)
        else:
            results_list = None

        results_list = self.mask_head.simple_test(
            feat, img_metas, rescale=rescale, instances_list=results_list)

        format_results_list = []
        for results in results_list:
            format_results_list.append(self.format_results(results))

        return format_results_list

    def format_results(self, results):
        """Format the model predictions according to the interface with
        dataset.

        Args:
            results (:obj:`InstanceData`): Processed
                results of single images. Usually contains
                following keys.

                - scores (Tensor): Classification scores, has shape
                  (num_instance,)
                - labels (Tensor): Has shape (num_instances,).
                - masks (Tensor): Processed mask results, has
                  shape (num_instances, h, w).

        Returns:
            tuple: Formatted bbox and mask results.. It contains two items:

                - bbox_results (list[np.ndarray]): BBox results of
                  single image. The list corresponds to each class.
                  each ndarray has a shape (N, 5), N is the number of
                  bboxes with this category, and last dimension
                  5 arrange as (x1, y1, x2, y2, scores).
                - mask_results (list[np.ndarray]): Mask results of
                  single image. The list corresponds to each class.
                  each ndarray has shape (N, img_h, img_w), N
                  is the number of masks with this category.
        """
        data_keys = results.keys()
        assert 'scores' in data_keys
        assert 'labels' in data_keys

        assert 'masks' in data_keys, \
            'results should contain ' \
            'masks when format the results '
        mask_results = [[] for _ in range(self.mask_head.num_classes)]

        num_masks = len(results)

        if num_masks == 0:
            bbox_results = [
                np.zeros((0, 5), dtype=np.float32)
                for _ in range(self.mask_head.num_classes)
            ]
            return bbox_results, mask_results

        labels = results.labels.detach().cpu().numpy()

        if 'bboxes' not in results:
            # create dummy bbox results to store the scores
            results.bboxes = results.scores.new_zeros(len(results), 4)

        det_bboxes = torch.cat([results.bboxes, results.scores[:, None]],
                               dim=-1)
        det_bboxes = det_bboxes.detach().cpu().numpy()
        bbox_results = [
            det_bboxes[labels == i, :]
            for i in range(self.mask_head.num_classes)
        ]

        masks = results.masks.detach().cpu().numpy()

        for idx in range(num_masks):
            mask = masks[idx]
            mask_results[labels[idx]].append(mask)

        return bbox_results, mask_results

    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError
