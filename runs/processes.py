from multiprocessing import Process
import torch
from socket import *
import traceback
import time
import json
from model.ops import set_logger
from runs.detect import Detect
from runs.detect_noblock import Detect as Detect_noblock


class socket_process(Process):

    def __init__(
        self,
        port=8080,
        logger_name="",
        logger_cfg=("D", 7),
        result_queue=None,
        request_queue=None,
        locker_request=None,
        locker_result=None,
    ):
        super().__init__()
        self.ip_port = (('localhost', port))
        self.locker_request = locker_request
        self.locker_result = locker_result
        self.result_queue = result_queue
        self.request_queue = request_queue
        self.logger_name = logger_name
        self.log_rotator, self.log_keep = logger_cfg
        self.logger = set_logger(self.logger_name, self.log_rotator,
                                 self.log_keep)

    def build_socket(self):
        while True:
            try:
                self.socket = socket(AF_INET, SOCK_STREAM)
                self.socket.setblocking(False)
                self.logger.info("success build socket")
                break
            except Exception as e:
                self.logger.warning("Failed to create socket. Error: %s" % e)
                continue
        while True:
            try:
                self.socket.bind(self.ip_port)
                self.logger.info(
                    f"success bind {self.ip_port[0]}:{self.ip_port[1]}")
                break
            except Exception:
                self.logger.warning(
                    f"failed to bind {self.ip_port[0]}:{self.ip_port[1]} \n {traceback.format_exc()}"
                )
                time.sleep(5)
                continue
        self.socket.listen(10)
        self.incomplete_str = ""

    def run(self):
        if not self.logger.hasHandlers():
            self.logger = set_logger(self.logger_name, self.log_rotator,
                                     self.log_keep)
        self.build_socket()
        while True:
            try:
                conn, addr = self.socket.accept()
                self.logger.info("connected to {}:{}".format(addr[0], addr[1]))
                print("connected to {}:{}".format(addr[0], addr[1]))
                conn.setblocking(False)
                while True:
                    try:
                        data = conn.recv(1024)
                        ######客户端断开，重新监听连接
                        if not data:
                            self.logger.info("connection interrupt !!!")
                            break
                        else:
                            data = str(data, encoding="utf-8")
                            self.locker_request.acquire()
                            data = self.incomplete_str + data
                            ########间隔标识符
                            data_split = data.split("inter")
                            self.incomplete_str = data_split[-1]
                            data_receive = data_split[:-1]
                            self.request_queue.extend(data_receive)
                            self.locker_request.release()
                            print(data_receive[0])
                    except:
                        pass
                    #####发送数据
                    self.locker_result.acquire()
                    t1 = time.time()
                    if len(self.result_queue):
                        self.logger.info(
                            "there are {} results waiting to sended".format(
                                len(self.result_queue)))
                        result_send = self.result_queue.pop(0)
                        try:
                            # print("\n\n\n\n\n")
                            # print(result_send)
                            conn.send(json.dumps(result_send).encode("utf-8"))
                            conn.send("inter".encode("utf-8"))
                            self.logger.info(f'发送数据花费时间： ({time.time() - t1:.3f}s)')
                            self.locker_result.release()
                            print(f'发送数据花费时间： ({time.time() - t1:.3f}s)')
                        ######客户端断开，重新监听连接
                        except:
                            # print("\n\n\n\n\n")
                            # print(result_send)
                            self.result_queue.insert(0, result_send)
                            self.locker_result.release()
                            self.logger.warning("failed to send data")
                            traceback.print_exc()
                            # exit(-1)
                            self.sleep(1)
                            pass
                    else:
                        self.locker_result.release()
            except:
                pass


class model_process(Process):

    def __init__(self,
                 weights=None,
                 configs=None,
                 device=0,
                 flip_test=False,
                 logger_name="connector",
                 logger_cfg=("D", 7),
                 max_bz=1,
                 max_bz_pose=5,
                 save_dir="",
                 result_queue=None,
                 request_queue=None,
                 locker_request=None,
                 locker_result=None,
                 detector=None,
                 index=None,
                 memory_config=None,
                 gap=100,
                 block=False):
        super().__init__()
        self.weights = weights
        self.configs = configs
        self.device = device
        self.flip_test = flip_test
        self.logger_name = logger_name
        self.max_bz = max_bz
        self.save_dir = save_dir
        self.locker_request = locker_request
        self.locker_result = locker_result
        self.result_queue = result_queue
        self.request_queue = request_queue
        self.detector = detector
        self.index = index
        self.max_bz_pose = max_bz_pose
        self.memory_config = memory_config
        self.gap = gap
        self.block = block
        # assert len(self.memory_config)==self.max_bz+1, "the length of mem_config is not right"

        if self.index is not None:
            self.logger_name = logger_name + "_{}".format(self.index + 1)
            self.save_dir = self.save_dir + "_{}".format(self.index + 1)
        self.log_rotator, self.log_keep = logger_cfg
        self.logger = set_logger(self.logger_name, self.log_rotator,
                                 self.log_keep)

    def build_model(self):
        while True:
            #########如果显存不足，回一直建立模型
            try:
                if self.block:
                    self.detector = Detect(weights=self.weights,
                                           configs=self.configs,
                                           device=self.device,
                                           flip_test=self.flip_test,
                                           logger=self.logger,
                                           max_bz_pose=self.max_bz_pose,
                                           save_dir=self.save_dir)
                    self.detector.load_model(self.detector.weights,self.detector.configs)
                else:
                    self.detector = Detect_noblock(
                        weights=self.weights,
                        configs=self.configs,
                        device=self.device,
                        flip_test=self.flip_test,
                        logger=self.logger,
                        max_bz_pose=self.max_bz_pose,
                        save_dir=self.save_dir)
                    self.detector.load_model(self.detector.weights,self.detector.configs)

                self.iteration = 0
                self.bz_now = 0
                break
            except RuntimeError as e:
                time.sleep(5)
                if "CUDA out of memory" in str(e):
                    self.logger.info(
                        "CUDA out of memory, try to build model again")

    def calculate_bz_now(self):
        import pynvml
        pynvml.nvmlInit()
        #########根据请求的大小确定最大的bz
        #######每隔一定的时间动态调整BZ，开始时刻也需要调整，防止显存不够
        if self.iteration % self.gap == 0:
            torch.cuda.empty_cache()
            # bz_now_request = max(1, min(self.max_bz, self.request_queue.qsize()))
            bz_now_request = max(1, min(self.max_bz, len(self.request_queue)))
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            ########获取当前剩余mem允许的最大bz
            bz_max_cal = 0
            ###########如果显存不够，则该进程一直等待足够的显存
            while True:
                #########查询剩余的显存，以便确定bz的上限
                meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                free_mem_now = meminfo.free / 1024 / 1024
                try:
                    for i in range(self.max_bz):
                        mem_need = self.memory_config[
                            i + 1] - self.memory_config[0] + 50
                        if free_mem_now > mem_need:
                            bz_max_cal = i + 1
                        else:
                            break
                    if bz_max_cal == 0:
                        raise Exception("free memory is not enough")
                    else:
                        break
                except Exception as e:
                    time.sleep(5)
                    self.logger.info(
                        str(e) + " and waiting for enough memory  ")

            self.bz_max_iter = bz_max_cal
            bz_now = min(self.bz_max_iter, bz_now_request)
            self.bz_max_iter = bz_now
            self.logger.info("change max_bz to: {}".format(self.bz_max_iter))
        else:
            bz_now_request = max(1, min(self.max_bz, len(self.request_queue)))
            bz_now = min(self.bz_max_iter, bz_now_request)
        return bz_now

    def run(self):
        if not self.logger.hasHandlers():
            self.logger = set_logger(self.logger_name, self.log_rotator,self.log_keep)
        while True:
            if self.detector is None:
                self.build_model()
            else:
                self.locker_request.acquire()
                if len(self.request_queue):
                    bz_now = self.calculate_bz_now()
                    self.locker_request.release()
                    while True:
                        self.locker_request.acquire()
                        if len(self.request_queue):
                            bz_now = min(bz_now, len(self.request_queue))
                        else:
                            self.locker_request.release()
                            continue
                        imgs_path = []
                        for i in range(bz_now):
                            imgs_path.append(self.request_queue.pop(0))
                        self.locker_request.release()
                        self.logger.info(
                            "there are {} images waiting to be processed".
                            format(len(self.request_queue)))
                        self.logger.info(
                            f"process {self.index} get inputs length: {len(imgs_path)}"
                        )
                        if len(imgs_path) < 1:
                            self.logger.info("img_path is empty")
                            continue
                        ##############如果显存不够报错，则将bz减小，并将self.bz_max_iter减小，继续进行预测
                        try:
                            results = self.detector.predict(imgs_path,logger=self.logger) ##调用推理代码
                            self.locker_result.acquire()
                            self.result_queue.append(results)
                            self.logger.info("ended result append {}".format(imgs_path))
                            self.locker_result.release()
                            self.iteration += 1
                            break
                        except RuntimeError as e:
                            self.logger.info("model error: {}".format(
                                e.__str__()))
                            if "CUDA out of memory" in str(e):
                                self.logger.info("CUDA out of memory, try to inference with smaller batch_sz")
                                self.locker_request.acquire()
                                #####将取出的图像放回请求列表
                                for i in range(len(imgs_path) - 1, -1, -1):
                                    self.request_queue.insert(0, imgs_path[i])
                                self.locker_request.release()
                                if bz_now > 1:
                                    bz_now -= 1
                                    self.bz_max_iter -= 1
                                    self.logger.info(
                                        "decrease bz_now and bz_max_iter")
                                else:
                                    import pynvml
                                    pynvml.nvmlInit()
                                    while True:
                                        handle = pynvml.nvmlDeviceGetHandleByIndex(
                                            0)
                                        meminfo = pynvml.nvmlDeviceGetMemoryInfo(
                                            handle)
                                        free_mem_now = meminfo.free / 1024 / 1024
                                        self.logger.info(
                                            "waiting for enough memory to inference one image"
                                        )
                                        #######当显存大于bz=1的显存需求量，则跳出循环
                                        if free_mem_now > self.memory_config[
                                                1] - self.memory_config[0] + 50:
                                            break
                                continue
                else:
                    self.locker_request.release()


#######根据显存和请求的数量动态管理模型进程
class processes_manager:

    def __init__(self):
        pass

    def run(self):
        pass