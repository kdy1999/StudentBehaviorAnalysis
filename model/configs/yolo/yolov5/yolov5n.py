_base_ = './yolov5s.py'
model = dict(
    backbone=dict(deepen_factor=0.33, widen_factor=0.25),
    neck=dict(
        in_channels=[64, 128, 256], out_channels=64, num_csp_blocks=2),
    # bbox_head=dict(in_channels=[192, 384, 768], feat_channels=192,stacked_convs=0,trident=False,decouple=False),
    bbox_head=dict(in_channels=[64, 128, 256], feat_channels=64 ,stacked_convs=2,trident=False,decouple=True),
    loss=dict(
        type="EFL",#BCE,EFL,FL,EQLV2,DYFL,GHM
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