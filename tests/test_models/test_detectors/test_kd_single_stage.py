# Copyright (c) OpenMMLab. All rights reserved.
import unittest
from unittest import TestCase

import torch
from parameterized import parameterized

from mmdet import *  # noqa
from mmdet.data_elements import DetDataSample
from mmdet.testing import demo_mm_inputs, get_detector_cfg
from mmdet.utils import register_all_modules


class TestKDSingleStageDetector(TestCase):

    def setUp(self):
        register_all_modules()

    @parameterized.expand(['ld/ld_r18_gflv1_r101_fpn_coco_1x.py'])
    def test_init(self, cfg_file):
        model = get_detector_cfg(cfg_file)
        model.backbone.init_cfg = None

        from mmdet.models import build_detector
        detector = build_detector(model)
        self.assertTrue(detector.backbone)
        self.assertTrue(detector.neck)
        self.assertTrue(detector.bbox_head)

    @parameterized.expand([('ld/ld_r18_gflv1_r101_fpn_coco_1x.py', ('cpu',
                                                                    'cuda'))])
    def test_single_stage_forward_train(self, cfg_file, devices):
        model = get_detector_cfg(cfg_file)
        model.backbone.init_cfg = None

        from mmdet.models import build_detector
        assert all([device in ['cpu', 'cuda'] for device in devices])

        for device in devices:
            detector = build_detector(model)

            if device == 'cuda':
                if not torch.cuda.is_available():
                    return unittest.skip('test requires GPU and torch+cuda')
                detector = detector.cuda()

            packed_inputs = demo_mm_inputs(2, [[3, 128, 128], [3, 125, 130]])
            batch_inputs, data_samples = detector.data_preprocessor(
                packed_inputs, True)

            # Test forward train
            losses = detector.forward(batch_inputs, data_samples, mode='loss')
            self.assertIsInstance(losses, dict)

    @parameterized.expand([('ld/ld_r18_gflv1_r101_fpn_coco_1x.py', ('cpu',
                                                                    'cuda'))])
    def test_single_stage_forward_test(self, cfg_file, devices):
        model = get_detector_cfg(cfg_file)
        model.backbone.init_cfg = None

        from mmdet.models import build_detector
        assert all([device in ['cpu', 'cuda'] for device in devices])

        for device in devices:
            detector = build_detector(model)

            if device == 'cuda':
                if not torch.cuda.is_available():
                    return unittest.skip('test requires GPU and torch+cuda')
                detector = detector.cuda()

            packed_inputs = demo_mm_inputs(2, [[3, 128, 128], [3, 125, 130]])
            batch_inputs, data_samples = detector.data_preprocessor(
                packed_inputs, False)

            # Test forward test
            detector.eval()
            with torch.no_grad():
                batch_results = detector.forward(
                    batch_inputs, data_samples, mode='predict')
                self.assertEqual(len(batch_results), 2)
                self.assertIsInstance(batch_results[0], DetDataSample)
