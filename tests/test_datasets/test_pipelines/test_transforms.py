# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
import unittest

import mmcv
import numpy as np

from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
from mmdet.core.mask import BitmapMasks
from mmdet.datasets.pipelines import (Expand, MinIoURandomCrop, MixUp, Mosaic,
                                      PhotoMetricDistortion, RandomAffine,
                                      RandomCrop, RandomFlip, Resize,
                                      SegRescale, YOLOXHSVRandomAug)
from .utils import create_random_bboxes


class TestResize(unittest.TestCase):

    def setUp(self):
        """Setup the model and optimizer which are used in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        rng = np.random.RandomState(0)
        self.data_info1 = dict(
            img=np.random.random((1333, 800, 3)),
            gt_seg_map=np.random.random((1333, 800, 3)),
            gt_bboxes=np.array([[0, 0, 112, 112]]),
            gt_masks=BitmapMasks(
                rng.rand(1, 1333, 800), height=1333, width=800))
        self.data_info2 = dict(
            img=np.random.random((300, 400, 3)),
            gt_bboxes=np.array([[200, 150, 600, 450]]))
        self.data_info3 = dict(img=np.random.random((300, 400, 3)))

    def test_resize(self):
        # test keep_ratio is True
        transform = Resize(scale=(2000, 2000), keep_ratio=True)
        results = transform(copy.deepcopy(self.data_info1))
        self.assertEqual(results['img_shape'], (2000, 1200))
        self.assertEqual(results['scale'], (1200, 2000))
        self.assertEqual(results['scale_factor'], (1200 / 800, 2000 / 1333))

        # test resize_bboxes/seg/masks
        transform = Resize(scale_factor=(1.5, 2))
        results = transform(copy.deepcopy(self.data_info1))
        self.assertTrue((results['gt_bboxes'] == np.array([[0, 0, 168,
                                                            224]])).all())
        self.assertEqual(results['gt_masks'].height, 2666)
        self.assertEqual(results['gt_masks'].width, 1200)
        self.assertEqual(results['gt_seg_map'].shape[:2], (2666, 1200))

        # test clip_object_border = False
        transform = Resize(scale=(200, 150), clip_object_border=False)
        results = transform(self.data_info2)
        self.assertTrue((results['gt_bboxes'] == np.array([100, 75, 300,
                                                           225])).all())

        # test only with image
        transform = Resize(scale=(200, 150), clip_object_border=False)
        results = transform(self.data_info3)
        self.assertTupleEqual(results['img'].shape[:2], (150, 200))

    def test_repr(self):
        transform = Resize(scale=(2000, 2000), keep_ratio=True)
        self.assertEqual(
            repr(transform), ('Resize(scale=(2000, 2000), '
                              'scale_factor=None, keep_ratio=True, '
                              'clip_object_border=True), backend=cv2), '
                              'interpolation=bilinear)'))


class TestRandomFlip(unittest.TestCase):

    def setUp(self):
        """Setup the model and optimizer which are used in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        rng = np.random.RandomState(0)
        self.results1 = {
            'img': np.random.random((224, 224, 3)),
            'gt_bboxes': np.array([[0, 1, 100, 101]]),
            'gt_masks':
            BitmapMasks(rng.rand(1, 224, 224), height=224, width=224),
            'gt_seg_map': np.random.random((224, 224))
        }

        self.results2 = {'img': self.results1['img']}

    def test_transform(self):
        # test with image, gt_bboxes, gt_masks, gt_seg_map
        transform = RandomFlip(1.0)
        results_update = transform.transform(copy.deepcopy(self.results1))
        self.assertTrue(
            (results_update['gt_bboxes'] == np.array([[124, 1, 224,
                                                       101]])).all())
        # test only with image
        transform = RandomFlip(1.0)
        results_update = transform.transform(copy.deepcopy(self.results2))
        self.assertTrue(
            (results_update['img'] == self.results2['img'][:, ::-1]).all())

    def test_repr(self):
        transform = RandomFlip(0.1)
        transform_str = str(transform)
        self.assertIsInstance(transform_str, str)


class TestMinIoURandomCrop(unittest.TestCase):

    def test_transform(self):
        results = dict()
        img = mmcv.imread(
            osp.join(osp.dirname(__file__), '../../data/color.jpg'), 'color')
        results['img'] = img
        results['img_shape'] = img.shape[:2]
        gt_bboxes = create_random_bboxes(1, results['img_shape'][1],
                                         results['img_shape'][0])
        results['gt_labels'] = np.ones(gt_bboxes.shape[0], dtype=np.int64)
        results['gt_bboxes'] = gt_bboxes
        transform = MinIoURandomCrop()
        results = transform.transform(copy.deepcopy(results))

        self.assertEqual(results['gt_labels'].shape[0],
                         results['gt_bboxes'].shape[0])
        self.assertEqual(results['gt_labels'].dtype, np.int64)
        self.assertEqual(results['gt_bboxes'].dtype, np.float32)

        patch = np.array(
            [0, 0, results['img_shape'][1], results['img_shape'][0]])
        ious = bbox_overlaps(patch.reshape(-1, 4),
                             results['gt_bboxes']).reshape(-1)
        mode = transform.mode
        if mode == 1:
            self.assertTrue(np.equal(results['gt_bboxes'], gt_bboxes).all())
        else:
            self.assertTrue((ious >= mode).all())

    def test_repr(self):
        transform = MinIoURandomCrop()
        self.assertEqual(
            repr(transform), ('MinIoURandomCrop'
                              '(min_ious=(0.1, 0.3, 0.5, 0.7, 0.9), '
                              'min_crop_size=0.3, '
                              'bbox_clip_border=True)'))


class TestPhotoMetricDistortion(unittest.TestCase):

    def test_transform(self):
        img = mmcv.imread(
            osp.join(osp.dirname(__file__), '../../data/color.jpg'), 'color')
        transform = PhotoMetricDistortion()

        # test uint8 input
        results = dict()
        results['img'] = img
        results = transform.transform(copy.deepcopy(results))
        self.assertEqual(results['img'].dtype, np.float32)

        # test float32 input
        results = dict()
        results['img'] = img.astype(np.float32)
        results = transform.transform(copy.deepcopy(results))
        self.assertEqual(results['img'].dtype, np.float32)

    def test_repr(self):
        transform = PhotoMetricDistortion()
        self.assertEqual(
            repr(transform), ('PhotoMetricDistortion'
                              '(brightness_delta=32, '
                              'contrast_range=(0.5, 1.5), '
                              'saturation_range=(0.5, 1.5), '
                              'hue_delta=18)'))


class TestExpand(unittest.TestCase):

    def setUp(self):
        """Setup the model and optimizer which are used in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        rng = np.random.RandomState(0)
        self.results = {
            'img': np.random.random((224, 224, 3)),
            'img_shape': (224, 224),
            'gt_bboxes': np.array([[0, 1, 100, 101]]),
            'gt_masks':
            BitmapMasks(rng.rand(1, 224, 224), height=224, width=224),
            'gt_seg_map': np.random.random((224, 224))
        }

    def test_transform(self):

        transform = Expand()
        results = transform.transform(copy.deepcopy(self.results))
        self.assertEqual(
            results['img_shape'],
            (results['gt_masks'].height, results['gt_masks'].width))
        self.assertEqual(results['img_shape'], results['gt_seg_map'].shape)

    def test_repr(self):
        transform = Expand()
        self.assertEqual(
            repr(transform), ('Expand'
                              '(mean=(0, 0, 0), to_rgb=True, '
                              'ratio_range=(1, 4), '
                              'seg_ignore_label=None, '
                              'prob=0.5)'))


class TestSegRescale(unittest.TestCase):

    def setUp(self) -> None:
        seg_map = np.random.randint(0, 255, size=(32, 32), dtype=np.int32)
        self.results = {'gt_seg_map': seg_map}

    def test_transform(self):
        # test scale_factor != 1
        transform = SegRescale(scale_factor=2)
        results = transform(copy.deepcopy(self.results))
        self.assertEqual(results['gt_seg_map'].shape[:2], (64, 64))
        # test scale_factor = 1
        transform = SegRescale(scale_factor=1)
        results = transform(copy.deepcopy(self.results))
        self.assertEqual(results['gt_seg_map'].shape[:2], (32, 32))

    def test_repr(self):
        transform = SegRescale(scale_factor=2)
        self.assertEqual(
            repr(transform), ('SegRescale(scale_factor=2, backend=cv2)'))


class TestRandomCrop(unittest.TestCase):

    def test_init(self):
        # test invalid crop_type
        with self.assertRaisesRegex(ValueError, 'Invalid crop_type'):
            RandomCrop(crop_size=(10, 10), crop_type='unknown')

        crop_type_list = ['absolute', 'absolute_range']
        for crop_type in crop_type_list:
            # test h > 0 and w > 0
            for crop_size in [(0, 0), (0, 1), (1, 0)]:
                with self.assertRaises(AssertionError):
                    RandomCrop(crop_size=crop_size, crop_type=crop_type)
            # test type(h) = int and type(w) = int
            for crop_size in [(1.0, 1), (1, 1.0), (1.0, 1.0)]:
                with self.assertRaises(AssertionError):
                    RandomCrop(crop_size=crop_size, crop_type=crop_type)

        # test crop_size[0] <= crop_size[1]
        with self.assertRaises(AssertionError):
            RandomCrop(crop_size=(10, 5), crop_type='absolute_range')

        # test h in (0, 1] and w in (0, 1]
        crop_type_list = ['relative_range', 'relative']
        for crop_type in crop_type_list:
            for crop_size in [(0, 1), (1, 0), (1.1, 0.5), (0.5, 1.1)]:
                with self.assertRaises(AssertionError):
                    RandomCrop(crop_size=crop_size, crop_type=crop_type)

    def test_transform(self):
        # test relative and absolute crop
        src_results = {
            'img': np.random.randint(0, 255, size=(32, 24), dtype=np.int32)
        }
        target_shape = (16, 12)
        for crop_type, crop_size in zip(['relative', 'absolute'], [(0.5, 0.5),
                                                                   (16, 12)]):
            transform = RandomCrop(crop_size=crop_size, crop_type=crop_type)
            results = transform(copy.deepcopy(src_results))
            self.assertEqual(results['img'].shape[:2], target_shape)

        # test absolute_range crop
        transform = RandomCrop(crop_size=(10, 20), crop_type='absolute_range')
        results = transform(copy.deepcopy(src_results))
        h, w = results['img'].shape
        self.assertTrue(10 <= h <= 20)
        self.assertTrue(10 <= w <= 20)
        # test relative_range crop
        transform = RandomCrop(
            crop_size=(0.5, 0.5), crop_type='relative_range')
        results = transform(copy.deepcopy(src_results))
        h, w = results['img'].shape
        self.assertTrue(16 <= h <= 32)
        self.assertTrue(12 <= w <= 24)

        # test with gt_bboxes, gt_bboxes_labels, gt_ignore_flags,
        # gt_masks, gt_seg_map
        img = np.random.randint(0, 255, size=(10, 10), dtype=np.uint8)
        gt_bboxes = np.array([[0, 0, 7, 7], [2, 3, 9, 9]], dtype=np.float32)
        gt_bboxes_labels = np.array([0, 1], dtype=np.int64)
        gt_ignore_flags = np.array([0, 1], dtype=np.bool8)
        gt_masks_ = np.zeros((2, 10, 10), np.uint8)
        gt_masks_[0, 0:7, 0:7] = 1
        gt_masks_[1, 2:7, 3:8] = 1
        gt_masks = BitmapMasks(gt_masks_.copy(), height=10, width=10)
        gt_seg_map = np.random.randint(0, 255, size=(10, 10), dtype=np.uint8)
        src_results = {
            'img': img,
            'gt_bboxes': gt_bboxes,
            'gt_bboxes_labels': gt_bboxes_labels,
            'gt_ignore_flags': gt_ignore_flags,
            'gt_masks': gt_masks,
            'gt_seg_map': gt_seg_map
        }
        transform = RandomCrop(
            crop_size=(5, 5),
            allow_negative_crop=False,
            recompute_bbox=False,
            bbox_clip_border=True)
        results = transform(copy.deepcopy(src_results))
        h, w = results['img'].shape
        self.assertEqual(h, 5)
        self.assertEqual(w, 5)
        self.assertEqual(results['gt_bboxes'].shape[0], 2)
        self.assertEqual(results['gt_bboxes_labels'].shape[0], 2)
        self.assertEqual(results['gt_ignore_flags'].shape[0], 2)
        self.assertTupleEqual(results['gt_seg_map'].shape[:2], (5, 5))

        # test recompute_bbox = True
        gt_masks_ = np.zeros((2, 10, 10), np.uint8)
        gt_masks = BitmapMasks(gt_masks_.copy(), height=10, width=10)
        gt_bboxes = np.array([[0.1, 0.1, 0.2, 0.2]])
        src_results = {
            'img': img,
            'gt_bboxes': gt_bboxes,
            'gt_masks': gt_masks
        }
        target_gt_bboxes = np.zeros((1, 4), dtype=np.float32)
        transform = RandomCrop(
            crop_size=(10, 10),
            allow_negative_crop=False,
            recompute_bbox=True,
            bbox_clip_border=True)
        results = transform(copy.deepcopy(src_results))
        self.assertTrue((results['gt_bboxes'] == target_gt_bboxes).all())

        # test bbox_clip_border = False
        src_results = {'img': img, 'gt_bboxes': gt_bboxes}
        transform = RandomCrop(
            crop_size=(10, 10),
            allow_negative_crop=False,
            recompute_bbox=True,
            bbox_clip_border=False)
        results = transform(copy.deepcopy(src_results))
        self.assertTrue(
            (results['gt_bboxes'] == src_results['gt_bboxes']).all())

        # test the crop does not contain any gt-bbox
        # allow_negative_crop = False
        img = np.random.randint(0, 255, size=(10, 10), dtype=np.uint8)
        gt_bboxes = np.zeros((0, 4), dtype=np.float32)
        src_results = {'img': img, 'gt_bboxes': gt_bboxes}
        transform = RandomCrop(crop_size=(5, 5), allow_negative_crop=False)
        results = transform(copy.deepcopy(src_results))
        self.assertIsNone(results)

        # allow_negative_crop = True
        img = np.random.randint(0, 255, size=(10, 10), dtype=np.uint8)
        gt_bboxes = np.zeros((0, 4), dtype=np.float32)
        src_results = {'img': img, 'gt_bboxes': gt_bboxes}
        transform = RandomCrop(crop_size=(5, 5), allow_negative_crop=True)
        results = transform(copy.deepcopy(src_results))
        self.assertTrue(isinstance(results, dict))

    def test_repr(self):
        crop_type = 'absolute'
        crop_size = (10, 10)
        allow_negative_crop = False
        recompute_bbox = True
        bbox_clip_border = False
        transform = RandomCrop(
            crop_size=crop_size,
            crop_type=crop_type,
            allow_negative_crop=allow_negative_crop,
            recompute_bbox=recompute_bbox,
            bbox_clip_border=bbox_clip_border)
        self.assertEqual(
            repr(transform),
            f'RandomCrop(crop_size={crop_size}, crop_type={crop_type}, '
            f'allow_negative_crop={allow_negative_crop}, '
            f'recompute_bbox={recompute_bbox}, '
            f'bbox_clip_border={bbox_clip_border})')


class TestMosaic(unittest.TestCase):

    def setUp(self):
        """Setup the model and optimizer which are used in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        rng = np.random.RandomState(0)
        self.results = {
            'img':
            np.random.random((224, 224, 3)),
            'img_shape': (224, 224),
            'gt_bboxes_labels':
            np.array([1, 2, 3], dtype=np.int64),
            'gt_bboxes':
            np.array([[10, 10, 20, 20], [20, 20, 40, 40], [40, 40, 80, 80]],
                     dtype=np.float32),
            'gt_ignore_flags':
            np.array([0, 0, 1], dtype=np.bool8),
            'gt_masks':
            BitmapMasks(rng.rand(3, 224, 224), height=224, width=224),
        }

    def test_transform(self):
        # test assertion for invalid img_scale
        with self.assertRaises(AssertionError):
            transform = Mosaic(img_scale=640)

        # test assertion for invalid probability
        with self.assertRaises(AssertionError):
            transform = Mosaic(prob=1.5)

        transform = Mosaic(img_scale=(10, 12))
        # test assertion for invalid mix_results
        with self.assertRaises(AssertionError):
            results = transform(copy.deepcopy(self.results))

        self.results['mix_results'] = [copy.deepcopy(self.results)] * 3
        results = transform(copy.deepcopy(self.results))
        self.assertTrue(results['img'].shape[:2] == (20, 24))
        self.assertTrue(results['gt_bboxes_labels'].shape[0] ==
                        results['gt_bboxes'].shape[0])
        self.assertTrue(results['gt_bboxes_labels'].dtype == np.int64)
        self.assertTrue(results['gt_bboxes'].dtype == np.float32)
        self.assertTrue(results['gt_ignore_flags'].dtype == np.bool8)

        # test removing outside bboxes when gt_bbox is empty.
        self.results['gt_bboxes'] = np.empty((0, 4), dtype=np.float32)
        self.results['gt_bboxes_labels'] = np.empty((0, ), dtype=np.int64)
        self.results['gt_ignore_flags'] = np.empty((0, 4), dtype=np.bool8)
        self.results['mix_results'] = [copy.deepcopy(self.results)] * 3
        results = transform(copy.deepcopy(self.results))
        self.assertIsInstance(results, dict)

    def test_repr(self):
        transform = Mosaic(img_scale=(640, 640), )
        self.assertEqual(
            repr(transform), ('Mosaic(img_scale=(640, 640), '
                              'center_ratio_range=(0.5, 1.5), '
                              'pad_val=114.0, '
                              'min_bbox_size=0, '
                              'skip_filter=True'
                              'prob=1.0)'))


class TestMixUp(unittest.TestCase):

    def setUp(self):
        """Setup the model and optimizer which are used in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        rng = np.random.RandomState(0)
        self.results = {
            'img':
            np.random.random((224, 224, 3)),
            'img_shape': (224, 224),
            'gt_bboxes_labels':
            np.array([1, 2, 3], dtype=np.int64),
            'gt_bboxes':
            np.array([[10, 10, 20, 20], [20, 20, 40, 40], [40, 40, 80, 80]],
                     dtype=np.float32),
            'gt_ignore_flags':
            np.array([0, 0, 1], dtype=np.bool8),
            'gt_masks':
            BitmapMasks(rng.rand(3, 224, 224), height=224, width=224),
        }

    def test_transform(self):
        # test assertion for invalid img_scale
        with self.assertRaises(AssertionError):
            transform = MixUp(img_scale=640)

        transform = MixUp(img_scale=(10, 12))
        # test assertion for invalid mix_results
        with self.assertRaises(AssertionError):
            results = transform(copy.deepcopy(self.results))

        with self.assertRaises(AssertionError):
            self.results['mix_results'] = [copy.deepcopy(self.results)] * 2
            results = transform(copy.deepcopy(self.results))

        self.results['mix_results'] = [copy.deepcopy(self.results)]
        results = transform(copy.deepcopy(self.results))
        self.assertTrue(results['img'].shape[:2] == (224, 224))
        self.assertTrue(results['gt_bboxes_labels'].shape[0] ==
                        results['gt_bboxes'].shape[0])
        self.assertTrue(results['gt_bboxes_labels'].dtype == np.int64)
        self.assertTrue(results['gt_bboxes'].dtype == np.float32)
        self.assertTrue(results['gt_ignore_flags'].dtype == np.bool8)

        # test filter bbox :
        # 2 boxes with sides 10 and 20 are filtered as min_bbox_size=30
        transform = MixUp(
            img_scale=(224, 224),
            ratio_range=(1.0, 1.0),
            min_bbox_size=30,
            skip_filter=False)
        results = transform(copy.deepcopy(self.results))
        print(results['gt_bboxes'])
        self.assertTrue(results['gt_bboxes'].shape[0] == 4)
        self.assertTrue(results['gt_bboxes_labels'].shape[0] == 4)
        self.assertTrue(results['gt_bboxes_labels'].shape[0] ==
                        results['gt_bboxes'].shape[0])
        self.assertTrue(results['gt_bboxes_labels'].dtype == np.int64)
        self.assertTrue(results['gt_bboxes'].dtype == np.float32)
        self.assertTrue(results['gt_ignore_flags'].dtype == np.bool8)

    def test_repr(self):
        transform = MixUp(
            img_scale=(640, 640),
            ratio_range=(0.8, 1.6),
            pad_val=114.0,
        )
        self.assertEqual(
            repr(transform), ('MixUp(dynamic_scale=(640, 640), '
                              'ratio_range=(0.8, 1.6), '
                              'flip_ratio=0.5, '
                              'pad_val=114.0, '
                              'max_iters=15, '
                              'min_bbox_size=5, '
                              'min_area_ratio=0.2, '
                              'max_aspect_ratio=20, '
                              'bbox_clip_border=True, '
                              'skip_filter=True)'))


class TestRandomAffine(unittest.TestCase):

    def setUp(self):
        """Setup the model and optimizer which are used in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        self.results = {
            'img':
            np.random.random((224, 224, 3)),
            'img_shape': (224, 224),
            'gt_bboxes_labels':
            np.array([1, 2, 3], dtype=np.int64),
            'gt_bboxes':
            np.array([[10, 10, 20, 20], [20, 20, 40, 40], [40, 40, 80, 80]],
                     dtype=np.float32),
            'gt_ignore_flags':
            np.array([0, 0, 1], dtype=np.bool8),
        }

    def test_transform(self):
        # test assertion for invalid translate_ratio
        with self.assertRaises(AssertionError):
            transform = RandomAffine(max_translate_ratio=1.5)

        # test assertion for invalid scaling_ratio_range
        with self.assertRaises(AssertionError):
            transform = RandomAffine(scaling_ratio_range=(1.5, 0.5))

        with self.assertRaises(AssertionError):
            transform = RandomAffine(scaling_ratio_range=(0, 0.5))

        transform = RandomAffine()
        results = transform(copy.deepcopy(self.results))
        self.assertTrue(results['img'].shape[:2] == (224, 224))
        self.assertTrue(results['gt_bboxes_labels'].shape[0] ==
                        results['gt_bboxes'].shape[0])
        self.assertTrue(results['gt_bboxes_labels'].dtype == np.int64)
        self.assertTrue(results['gt_bboxes'].dtype == np.float32)
        self.assertTrue(results['gt_ignore_flags'].dtype == np.bool8)

        # test filter bbox
        transform = RandomAffine(
            max_rotate_degree=0.,
            max_translate_ratio=0.,
            scaling_ratio_range=(1., 1.),
            max_shear_degree=0.,
            border=(0, 0),
            min_bbox_size=30,
            max_aspect_ratio=20,
            skip_filter=False)
        results = transform(copy.deepcopy(self.results))
        self.assertTrue(results['gt_bboxes'].shape[0] == 1)
        self.assertTrue(results['gt_bboxes_labels'].shape[0] == 1)
        self.assertTrue(results['gt_bboxes_labels'].shape[0] ==
                        results['gt_bboxes'].shape[0])
        self.assertTrue(results['gt_bboxes_labels'].dtype == np.int64)
        self.assertTrue(results['gt_bboxes'].dtype == np.float32)
        self.assertTrue(results['gt_ignore_flags'].dtype == np.bool8)

    def test_repr(self):
        transform = RandomAffine(
            scaling_ratio_range=(0.1, 2),
            border=(-320, -320),
        )
        self.assertEqual(
            repr(transform), ('RandomAffine(max_rotate_degree=10.0, '
                              'max_translate_ratio=0.1, '
                              'scaling_ratio_range=(0.1, 2), '
                              'max_shear_degree=2.0, '
                              'border=(-320, -320), '
                              'border_val=(114, 114, 114), '
                              'min_bbox_size=2, '
                              'min_area_ratio=0.2, '
                              'max_aspect_ratio=20, '
                              'bbox_clip_border=True, '
                              'skip_filter=True)'))


class TestYOLOXHSVRandomAug(unittest.TestCase):

    def setUp(self):
        """Setup the model and optimizer which are used in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        img = mmcv.imread(
            osp.join(osp.dirname(__file__), '../../data/color.jpg'), 'color')
        self.results = {
            'img':
            img,
            'img_shape': (224, 224),
            'gt_bboxes_labels':
            np.array([1, 2, 3], dtype=np.int64),
            'gt_bboxes':
            np.array([[10, 10, 20, 20], [20, 20, 40, 40], [40, 40, 80, 80]],
                     dtype=np.float32),
            'gt_ignore_flags':
            np.array([0, 0, 1], dtype=np.bool8),
        }

    def test_transform(self):
        transform = YOLOXHSVRandomAug()
        results = transform(copy.deepcopy(self.results))
        self.assertTrue(
            results['img'].shape[:2] == self.results['img'].shape[:2])
        self.assertTrue(results['gt_bboxes_labels'].shape[0] ==
                        results['gt_bboxes'].shape[0])
        self.assertTrue(results['gt_bboxes_labels'].dtype == np.int64)
        self.assertTrue(results['gt_bboxes'].dtype == np.float32)
        self.assertTrue(results['gt_ignore_flags'].dtype == np.bool8)

    def test_repr(self):
        transform = YOLOXHSVRandomAug()
        self.assertEqual(
            repr(transform), ('YOLOXHSVRandomAug(hue_delta=5, '
                              'saturation_delta=30, '
                              'value_delta=30)'))
