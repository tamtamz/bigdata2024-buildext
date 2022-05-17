# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
from mmengine.data import InstanceData

from mmdet.models.dense_heads import FCOSHead


class TestFCOSHead(TestCase):

    def test_fcos_head_loss(self):
        """Tests fcos head loss when truth is empty and non-empty."""
        s = 256
        img_metas = [{
            'img_shape': (s, s, 3),
            'scale_factor': 1,
        }]
        fcos_head = FCOSHead(
            num_classes=4,
            in_channels=1,
            feat_channels=1,
            stacked_convs=1,
            norm_cfg=None)

        # Fcos head expects a multiple levels of features per image
        feats = (
            torch.rand(1, 1, s // stride[1], s // stride[0])
            for stride in fcos_head.prior_generator.strides)
        cls_scores, bbox_preds, centernesses = fcos_head.forward(feats)

        # Test that empty ground truth encourages the network to
        # predict background
        gt_instances = InstanceData()
        gt_instances.bboxes = torch.empty((0, 4))
        gt_instances.labels = torch.LongTensor([])

        empty_gt_losses = fcos_head.loss(cls_scores, bbox_preds, centernesses,
                                         [gt_instances], img_metas)
        # When there is no truth, the cls loss should be nonzero but
        # box loss and centerness loss should be zero
        empty_cls_loss = empty_gt_losses['loss_cls'].item()
        empty_box_loss = empty_gt_losses['loss_bbox'].item()
        empty_ctr_loss = empty_gt_losses['loss_centerness'].item()
        self.assertGreater(empty_cls_loss, 0, 'cls loss should be non-zero')
        self.assertEqual(
            empty_box_loss, 0,
            'there should be no box loss when there are no true boxes')
        self.assertEqual(
            empty_ctr_loss, 0,
            'there should be no centerness loss when there are no true boxes')

        # When truth is non-empty then all cls, box loss and centerness loss
        # should be nonzero for random inputs
        gt_instances = InstanceData()
        gt_instances.bboxes = torch.Tensor(
            [[23.6667, 23.8757, 238.6326, 151.8874]])
        gt_instances.labels = torch.LongTensor([2])

        one_gt_losses = fcos_head.loss(cls_scores, bbox_preds, centernesses,
                                       [gt_instances], img_metas)
        onegt_cls_loss = one_gt_losses['loss_cls'].item()
        onegt_box_loss = one_gt_losses['loss_bbox'].item()
        onegt_ctr_loss = one_gt_losses['loss_centerness'].item()
        self.assertGreater(onegt_cls_loss, 0, 'cls loss should be non-zero')
        self.assertGreater(onegt_box_loss, 0, 'box loss should be non-zero')
        self.assertGreater(onegt_ctr_loss, 0,
                           'centerness loss should be non-zero')

        # Test the `center_sampling` works fine.
        fcos_head.center_sampling = True
        ctrsamp_losses = fcos_head.loss(cls_scores, bbox_preds, centernesses,
                                        [gt_instances], img_metas)
        ctrsamp_cls_loss = ctrsamp_losses['loss_cls'].item()
        ctrsamp_box_loss = ctrsamp_losses['loss_bbox'].item()
        ctrsamp_ctr_loss = ctrsamp_losses['loss_centerness'].item()
        self.assertGreater(ctrsamp_cls_loss, 0, 'cls loss should be non-zero')
        self.assertGreater(ctrsamp_box_loss, 0, 'box loss should be non-zero')
        self.assertGreater(ctrsamp_ctr_loss, 0,
                           'centerness loss should be non-zero')

        # Test the `norm_on_bbox` works fine.
        fcos_head.norm_on_bbox = True
        normbox_losses = fcos_head.loss(cls_scores, bbox_preds, centernesses,
                                        [gt_instances], img_metas)
        normbox_cls_loss = normbox_losses['loss_cls'].item()
        normbox_box_loss = normbox_losses['loss_bbox'].item()
        normbox_ctr_loss = normbox_losses['loss_centerness'].item()
        self.assertGreater(normbox_cls_loss, 0, 'cls loss should be non-zero')
        self.assertGreater(normbox_box_loss, 0, 'box loss should be non-zero')
        self.assertGreater(normbox_ctr_loss, 0,
                           'centerness loss should be non-zero')
