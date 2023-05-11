# model settings
anchors=[[10,13, 16,30, 33,23],
        [30,61, 62,45, 59,119],
        [116,90, 156,198, 373,326]]
model = dict(
    type='yolov5',
    backbone=dict(type='yolov5', arch='P5', deepen_factor=0.33, widen_factor=0.5),
    neck=dict(
        type='yolov5',
        in_channels=[128, 256, 512],
        out_channels=128,
        num_csp_blocks=1),
    bbox_head=dict(
        type='yolov5', num_classes=13, in_channels=128, feat_channels=128,
        anchors=anchors,stacked_convs=0,decouple=False,stride=[8,16,32],trident=False,big_conv=False),
    loss=dict(
        type="BCE"
    )
)


# dataset settings
datasets=dict(
    path='../../datasets/minedata9',
    train='train.txt',
    val='val.txt',
    repeat='all.txt',
    weight_factor=[1,10,3,10,1,5,5,5],
    img_scale = (800, 800),
    factor_sample=2,
    repeat_num=3,
    rect=True,
    is_coco=False,
    num_classes=13,
    pipeline= dict(
        hsv_h=0.0138,
        hsv_s=0.664,
        hsv_v=0.464,
        degrees=0.373,
        translate=0.245,
        # scale= 0.898
        scale=0.5,
        shear=0.602,
        perspective=0.0,
        flipud=0.00856,
        fliplr=0.5,
        mosaic=0.5,
        mixup=0.243,
        copy_paste=0.0,
    ),
    block2=dict(
        block_num=3,
        factor=0.25
    ),
    anchors=dict(
        anchors=anchors
        ),
    names=[ "person",'phone',["front","behind","left","right"],"using phone","eating",
            "watching blackboard","standing","sleeping","writing or reading"],
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


