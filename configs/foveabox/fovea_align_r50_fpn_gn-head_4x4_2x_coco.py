_base_ = './fovea_r50_fpn_4x4_1x_coco.py'
model = dict(
    bbox_head=dict(
        with_deform=True,
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)))
# learning policy
max_epochs = 24
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
train_cfg = dict(max_epochs=max_epochs)
optim_wrapper = dict(clip_grad=dict(max_norm=35, norm_type=2))
