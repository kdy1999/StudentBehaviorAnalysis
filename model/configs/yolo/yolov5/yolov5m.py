_base_ = './yolov5s.py'
model = dict(
    backbone=dict(deepen_factor=0.67, widen_factor=0.75),
    neck=dict(
        in_channels=[192, 384, 768], out_channels=0, num_csp_blocks=2),
    # bbox_head=dict(in_channels=[192, 384, 768], feat_channels=192,stacked_convs=0,trident=False,decouple=False),
    bbox_head=dict(in_channels=[192, 384, 768], feat_channels=192,stacked_convs=2,trident=False,decouple=True),
    loss=dict(
        type="FL",#BCE,EFL,FL,EQLV2,DYFL,GHM
    ),
)


datasets=dict(
    ##########使用EFL,EQLV2等长尾LOSS时，不应该使用太强的重复采样
    # weight_factor=[1,15,3,10,1,5,5,2],
    #########EFL,EQLV2
    weight_factor=[1,4,1,4,1,1,1,1],
    img_scale = (960,960),
    factor_sample=2,
    repeat_num=3,
)

schedules=dict(
    lr0= 0.0025,
    lrf= 0.12,
    momentum= 0.843,
    weight_decay= 0.00036,
    warmup_epochs= 3.0,
    warmup_momentum= 0.5,
    warmup_bias_lr= 0.05,
    box= 0.0296,
    cls= 0.243,
    cls_pw= 0.631,
    obj= 0.301,
    obj_pw= 0.911,
    iou_t= 0.2,
    anchor_t=2.9,
    # anchors= 3.63
    fl_gamma= 0.0,
    )

