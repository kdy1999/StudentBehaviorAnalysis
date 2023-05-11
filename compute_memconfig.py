import yaml
import torch
import pynvml
import argparse
from time import time
from runs.detect import Detect  # 1.5
# from connector_mp1 import Detect # 1.2

parser = argparse.ArgumentParser()
parser.add_argument("--imgpath", type=str, required=True)
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--maxt", type=float, default=10.)
parser.add_argument("--bzsave", type=int, default=5)
parser.add_argument("--ntries", type=int, default=5)

args = parser.parse_args()

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(args.device)
used = pynvml.nvmlDeviceGetMemoryInfo(handle).used

num_pose, num, gmem, t = "bz_pose", "bz", "gmem(Mb)", "runtime(s)"
print(f"{num_pose:>8s} {num:>3s} {gmem:>10s} {t:>12s}")

for bz_pose in range(5, 11):
    mem_config = []
    est_time = []
    img_lst = []
    torch.cuda.empty_cache()
    detector = Detect(weights=[
        "model/checkpoint/last_block.pt", "model/checkpoint/hrnet.pt"
    ],
                      configs=[
                          "model/configs/yolo/yolov5/yolov5m.py",
                          "model/configs/pose/hrnet/hrnet.py"
                      ],
                      max_bz_pose=bz_pose)
    detector.load_model(detector.weights, detector.configs)
    detector.predict(args.imgpath)

    while True:
        try:
            img_lst.append(args.imgpath)
            t0 = time()
            for _ in range(args.ntries):
                detector.predict(img_lst)
            delta_time = (time() - t0) / args.ntries
            delta = pynvml.nvmlDeviceGetMemoryInfo(handle).used - used
            delta = int(delta / 1024 / 1024)
            if mem_config and delta < mem_config[-1]:
                break
            mem_config.append(delta)
            est_time.append(delta_time)
            print(
                f"{bz_pose:>8} {len(img_lst):>3} {delta:>10} {f'{delta_time:.3f}':>12}"
            )
            if delta_time > args.maxt:
                break
        except:
            import traceback
            traceback.print_exc()
            break
    if bz_pose == args.bzsave:
        with open('mem_config_new.yaml', 'w') as yaml_file:
            yaml.dump(
                {
                    'mem_config': mem_config,
                    'est_time': est_time,
                    'device': 0,
                    'bz_pose': bz_pose
                }, yaml_file)
"""
参数：
img_path为测试图片路径，建议使用人数较多的图片以提高预测的冗余
device为显卡的ID，默认为0
maxt为可容忍的最大运行时长，即运行时长超过maxt时，不考虑更大的输入， 默认为10.0
bzsave为存储的bz_size_pose，即姿态模型同时运行的图片数量（不影响运行）， 默认为5
ntries为每轮测试的运行次数，次数越多结果越准，但耗时等比增长，默认为5，建议30以上

注：
1. 请保证当前卡上正在运行的其他进程所占用的显存不会发生变化，否则测试结果不准
2. 结果与pytroch版本以及显卡型号相关
3. 根据版本号调整Detect的import路径
4. 可基于结果选择合适的max_bz和max_bz_pose参数，若进行修改，对于v1.2版请删除connector_mp1.py的第783行

命令：
python compute_memconfig.py --imgpath [img_path] --device 0 --maxt 2.5 --bzsave 5 --ntries 5
"""