_base_ = [
    '../_base_/models/upernet_r50.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]

model = dict(
    pretrained='',
    backbone=dict(
        type='ActiveLarge',
        drop_path_rate=0.2,
    ),
    decode_head=dict(
        in_channels=[96, 192, 384, 768],
        num_classes=150
    ),
    auxiliary_head=dict(
        in_channels=384,
        num_classes=150
    )
)

optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(
                     custom_keys={'norm': dict(decay_mult=0.)})
                 )
lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False
                 )
data = dict(samples_per_gpu=2)

runner = dict(type='IterBasedRunnerAmp')

fp16 = None
optimizer_config = dict(
    type="DistOptimizerHook",
    update_interval=1,
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=True,
)
