import yaml
import sys

sys.path.append(".")
import argparse
from multiprocessing import Lock, Manager
import multiprocessing as mp
from runs.processes import model_process, socket_process
from runs.processes_track import model_track_process


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_bz",type=int,default=10,help="batch_size of yolov5_model")
    parser.add_argument("--max_bz_pose",type=int,default=5,help="batch_size of pose_model")
    parser.add_argument("--max_num_processes", type=int, default=1)
    parser.add_argument("--logger_name_model", type=str, default="model")
    parser.add_argument("--logger_name_socket", type=str, default="socket")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--flip_test", action="store_true")
    parser.add_argument("--ip_port", type=int, default=8037)
    parser.add_argument("--mem_config", type=str, default="mem_config.yml")
    parser.add_argument("--gap", type=int, default=5)
    parser.add_argument("--process_imgs_thres", type=int, default=21)
    parser.add_argument("--window_size",type=int,default=7,help="the size of the window to aggregate scores")
    parser.add_argument("--track",action="store_true",help="using tracking module")
    parser.add_argument("--visualization",action="store_true",help="whether to visualize the results")
    parser.add_argument("--block",action="store_true",help="whether to split the image to blocks")
    parser.add_argument(
        "--log_rotator",
        type=str,
        default="D",
        choices=["S", "M", "H", "D", "W0", "W1", "W2", "W3", "W4", "W5", "W6"])
    parser.add_argument("--log_keep", type=int, default=7)
    return parser.parse_args()


if __name__ == "__main__":
    # os.chdir(os.path.dirname(sys.argv[0]))

    mp.set_start_method("spawn")
    args = parse_opt()
    question_list = []
    results_list = []
    locker_request = Lock()
    locker_result = Lock()
    result_queue = Manager().list()
    request_queue = Manager().list()
    model_pros = []
    Value_list = []
    with open(args.mem_config, "r", encoding="utf-8") as f:
        memory_config = yaml.load(f.read(), Loader=yaml.FullLoader)
    yolo_path = "model/checkpoint/last_noblock.pt"
    if args.block:
        yolo_path = "model/checkpoint/last_block.pt"
    for i in range(args.max_num_processes):
        if args.track:
            model_pros.append(
                model_track_process(
                    weights=[yolo_path, "model/checkpoint/hrnet.pt"],
                    configs=[
                        "model/configs/yolo/yolov5/yolov5m.py",
                        "model/configs/pose/hrnet/hrnet.py"
                    ],
                    device=args.device,
                    flip_test=args.flip_test,
                    logger_name=args.logger_name_model,
                    logger_cfg=(args.log_rotator, args.log_keep),
                    max_bz=args.max_bz,
                    max_bz_pose=args.max_bz_pose,
                    save_dir="results_json",
                    locker_request=locker_request,
                    locker_result=locker_result,
                    request_queue=request_queue,
                    result_queue=result_queue,
                    detector=None,
                    index=i,
                    memory_config=memory_config["mem_config"],
                    gap=args.gap,
                    process_imgs_thres=args.process_imgs_thres,
                    window_size=args.window_size,
                    block=args.block))
        else:
            model_pros.append(
                model_process(weights=[yolo_path, "model/checkpoint/hrnet.pt"],
                              configs=[
                                  "model/configs/yolo/yolov5/yolov5m.py",
                                  "model/configs/pose/hrnet/hrnet.py"
                              ],
                              device=args.device,
                              flip_test=args.flip_test,
                              logger_name=args.logger_name_model,
                              logger_cfg=(args.log_rotator, args.log_keep),
                              max_bz=args.max_bz,
                              max_bz_pose=args.max_bz_pose,
                              save_dir="results_json",
                              locker_request=locker_request,
                              locker_result=locker_result,
                              request_queue=request_queue,
                              result_queue=result_queue,
                              detector=None,
                              index=i,
                              memory_config=memory_config["mem_config"],
                              gap=args.gap,
                              block=args.block))

        model_pros[i].start()
    socket_pro = socket_process(port=args.ip_port,
                                logger_name=args.logger_name_socket,
                                logger_cfg=(args.log_rotator, args.log_keep),
                                locker_request=locker_request,
                                locker_result=locker_result,
                                request_queue=request_queue,
                                result_queue=result_queue)
    socket_pro.start()
    for pro in model_pros:
        pro.join()
    socket_pro.join()
