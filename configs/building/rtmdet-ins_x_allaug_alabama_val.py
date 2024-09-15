_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/datasets/building_alabama_val.py'
]

auto_scale_lr = dict(enable=False, base_batch_size=16)

pretrained_path = 'rtmdet-ins_x_8xb16-300e_coco_20221124_111313-33d4595b.pth'

model = dict(
    type='RTMDet',
    init_cfg=dict(
        type='Pretrained',
        checkpoint=pretrained_path),
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[103.53, 116.28, 123.675],
        std=[57.375, 57.12, 58.395],
        bgr_to_rgb=False,
        batch_augments=None),
    backbone=dict(
        type='CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=1.33,
        widen_factor=1.25,
        channel_attention=True,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU', inplace=True),
    ),
    neck=dict(
        type='CSPNeXtPAFPN',
        in_channels=[320, 640, 1280],
        out_channels=320,
        num_csp_blocks=4,
        expand_ratio=0.5,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU', inplace=True)),
    bbox_head=dict(
        type='RTMDetInsSepBNHead',
        num_classes=1,
        in_channels=320,
        stacked_convs=2,
        share_conv=True,
        pred_kernel_size=1,
        feat_channels=320,
        act_cfg=dict(type='SiLU', inplace=True),
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        anchor_generator=dict(
            type='MlvlPointGenerator', offset=0, strides=[8, 16, 32]),
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        loss_mask=dict(
            type='DiceLoss', loss_weight=2.0, eps=5e-6, reduction='mean')),
    train_cfg=dict(
        assigner=dict(type='DynamicSoftLabelAssigner', topk=13),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=300,
        mask_thr_binary=0.5),
)

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='RandomAffine', max_rotate_degree=180., scaling_ratio_range=(0.5, 1.0), max_shear_degree=10.),
    dict(
        type='LoadAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(
        type='mmdet.RandomFlip',
        prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1)),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    batch_sampler=None,
    pin_memory=True,
    dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=5, num_workers=10, dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader

max_epochs = 24
base_lr = 0.001
interval = 3

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=interval)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49),
]

val_evaluator = dict(proposal_nums=(100, 1, 10), metric=['bbox', 'segm'])
test_evaluator = val_evaluator

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=1e-4))

# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[16, 22],
        gamma=0.1)
]

# hooks
default_hooks = dict(
    checkpoint=dict(
        interval=interval,
        max_keep_ckpts=10  # only keep latest 3 checkpoints
    ))
