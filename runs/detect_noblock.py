import torch
import cv2
import os
import json
import yaml
import numpy as np
import sys
import torch
import numpy as np
import cv2
from pathlib import Path
import shutil
import time
import logging
import copy
import warnings

sys.path.append(".")
from tools.general import Iou,miniou,iou_nms,miniou_nms,clip_border, clip_cordinate,\
    clip_coords,merge_person,merge_person_phone,letterbox,xywh2xyxy\
    ,scale_coords,get_config,LoadImage,keypoints_from_heatmaps_subimg\
    ,non_max_suppression_block,DatasetInfo,flip_back

from tools.general import collate
from dataset.shared_transform import Compose
from model.detector import build_detector
from model.ops import fuse_model, iou_overlap


class Detect:

    def __init__(self,
                 logger,
                 iou_thres=0.45,
                 img_size=960,
                 conf_thres=0.25,
                 half=True,
                 device=0,
                 weights=None,
                 block_num=3,
                 factor=0.15,
                 configs=None,
                 flip_test=False,
                 max_bz_pose=5,
                 save_dir=None):
        self.iou_thres = iou_thres
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.half = half
        self.weights = weights
        self.device = device
        self.block_num = block_num
        self.factor = factor
        self.save_dir = save_dir if save_dir else "work_dir"
        self.configs = configs
        self.stride = 32
        self.flip_test = flip_test
        self.max_bz_pose = max_bz_pose
        self.init_device()

        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        else:
            shutil.rmtree(self.save_dir)
            os.mkdir(self.save_dir)
        self.logger = logger

    def init_device(self):
        cpu = self.device < 0
        if cpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        elif self.device:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.device)
        if cpu:
            self.half = False
        self.device = torch.device(f'cuda:{self.device}' if not cpu else 'cpu')

    def get_block_imgs(self, image):
        block_images_rect = []
        block_images = []
        block_cordinates = []
        shape = image.shape
        row_block = shape[0] // self.block_num
        col_block = shape[1] // self.block_num
        row_cordinates = [[
            row * row_block,
            (row + 1) * row_block + int(row_block * self.factor)
        ] for row in range(self.block_num)]
        col_cordinates = [[
            col * col_block,
            (col + 1) * col_block + int(col_block * self.factor)
        ] for col in range(self.block_num)]
        clip_cordinate(row_cordinates, shape[0])
        clip_cordinate(col_cordinates, shape[1])

        for row in range(self.block_num):
            for col in range(self.block_num):
                row_cordinate = row_cordinates[row]
                col_cordinate = col_cordinates[col]
                block_image = image[row_cordinate[0]:row_cordinate[1],
                                    col_cordinate[0]:col_cordinate[1], :]
                block_image_rect = letterbox(block_image,
                                             self.img_size,
                                             stride=self.stride)[0]
                block_image_rect = block_image_rect.transpose(
                    (2, 0, 1))[::-1][None]
                block_images_rect.append(
                    np.ascontiguousarray(block_image_rect))
                block_images.append(block_image)
                block_cordinates.append([row_cordinate, col_cordinate])
        image_rect = letterbox(image, self.img_size, stride=self.stride)[0]
        image_rect = np.ascontiguousarray(
            image_rect.transpose((2, 0, 1))[::-1][None])
        return block_images_rect, block_images, block_cordinates, image_rect, image

    def load_model(self, weights, configs):
        self.model_type_yolo = os.path.splitext(self.weights[0])[1][1:]
        self.model_type_pose = os.path.splitext(self.weights[1])[1][1:]
        if self.model_type_yolo == "onnx":
            self.pt_yolo = False
            import onnxruntime
            self.session_yolo = onnxruntime.InferenceSession(
                self.weights[0], None)
        else:
            self.pt_yolo = True
            self.config_yolo = get_config(self.configs[0])
            self.model_yolo = build_detector(self.config_yolo["model"])
            for name, para in self.model_yolo.named_parameters():
                para.requires_grad = False
            pretrained_yolo = torch.load(
                self.weights[0],
                map_location=self.device)["model"].state_dict()
            self.model_yolo.load_state_dict(pretrained_yolo)
            fuse_model(self.model_yolo)
            self.model_yolo.to(self.device).eval()
            if self.half:
                self.model_yolo.half()

        self.config_pose = get_config(self.configs[1])
        if self.model_type_pose == "onnx":
            self.pt_pose = False
            import onnxruntime
            self.session_pose = onnxruntime.InferenceSession(
                self.weights[1], None)
        else:
            self.pt_pose = True
            self.model_pose = build_detector(self.config_pose["model"])
            for name, para in self.model_pose.named_parameters():
                para.requires_grad = False
            pretrained_pose = torch.load(
                self.weights[1], map_location=self.device)["state_dict"]

            self.model_pose.load_state_dict(pretrained_pose)
            fuse_model(self.model_pose)
            self.model_pose.to(self.device).eval()
            #########pose_model不是混合精度训练的，半精度预测会降低准确度
            # if self.half:
            #     self.model_pose.half()

        self.dataset = self.config_pose.data['test']['type']
        self.dataset_info = self.config_pose.data['test'].get(
            'dataset_info', None)
        self.dataset_info = DatasetInfo(self.dataset_info)

        channel_order = self.config_pose.test_pipeline[0].get(
            'channel_order', 'rgb')
        test_pipeline = self.config_pose.test_pipeline[1:]
        self.test_pipeline = Compose(test_pipeline)

    def trans2half(self):
        if self.pt_yolo:
            self.model_yolo.half()
        # if self.pt_pose:
        #     self.model_hrnet.half()
        self.half = True

    def trans2float(self):
        if self.pt_yolo:
            self.model_yolo.float()
        # if self.pt_pose:
        #     self.model_hrnet.float()
        self.half = False

    def predict(self,
                img_or_path=None,
                logger=None,
                max_det=2000,
                w_factor=0.05,
                h_factor=0.05):
        t1 = time.time()
        valid_path = []
        invalid_path = []

        if not isinstance(img_or_path, list):
            img_or_path = [img_or_path]
        if isinstance(img_or_path[0], str):
            imgs = []
            info = ""
            for img_path in img_or_path:
                info += " " + img_path
            self.logger.info("get imgs" + info)
            for i, img_path in enumerate(img_or_path):
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        raise Exception(img_path + " is empty!")
                    else:
                        img = cv2.resize(img, dsize=(1920, 1080))
                        imgs.append(img)
                        valid_path.append(img_path)
                except Exception as e:
                    invalid_path.append(img_path)
                    self.logger.warning(img_path + " is invalid")

        else:
            imgs = img_or_path
        if len(imgs) == 0:
            results = dict(results=[])
            #####读取错误的路径
            for i, invalid_path_single in enumerate(invalid_path):
                results["results"].append(
                    dict(result=dict(person=[]),
                         img_path=invalid_path_single,
                         error="img_path is invalid"))
            return results

        images_rect = []
        preds_full = []
        bz = len(imgs)
        max_w, max_h = 0, 0

        for img in imgs:
            image_rect = letterbox(img, self.img_size, stride=self.stride)[0]
            image_rect = image_rect.transpose((2, 0, 1))[::-1][None]
            images_rect.append(np.ascontiguousarray(image_rect))
        images_rect = np.concatenate(images_rect, axis=0)

        #########基于全图预测的结果，需要去除手机预测(全图的手机预测效果差)，使用nms获取所有人的框
        if self.pt_yolo:
            images_rect = torch.from_numpy(images_rect).to(self.device)
            images_rect = images_rect.half(
            ) if self.half else images_rect.float()  # uint8 to fp16/32
        else:
            images_rect = images_rect.astype(np.float32)
        images_rect /= 255.0

        if len(images_rect.shape) == 3:
            images_rect = images_rect[None]
        if self.pt_yolo:
            try:
                pred_full = self.model_yolo(images_rect)[0]
                pred_full = pred_full.detach().float()
            except Exception as e:
                pass

        else:
            pred_full = torch.tensor(
                self.session_yolo.run(
                    [self.session_yolo.get_outputs()[0].name],
                    {self.session_yolo.get_inputs()[0].name: image_rect}))

        #####
        boxes_full = xywh2xyxy(pred_full[..., :4])
        pred_full = torch.cat((boxes_full, pred_full[..., 4:]), dim=-1)
        scale_coords(images_rect.shape[2:], pred_full[..., :4],
                     imgs[0].shape).round()
        pred_full = non_max_suppression_block(pred_full,
                                              self.conf_thres,
                                              self.iou_thres,
                                              max_det=2000)
        #######全图预测结果

        person_usingphone_index_all = []
        person_usingphone_all = []
        person_usingphone_index_img = []
        phone_usingphone_all = []
        pred_phone_all = []
        pred_person_all = []

        #####################[x, y, x, y, conf_cls，cls, 转向分数, 转向id，传递, 玩手机，吃东西，注释，起身，趴桌，阅读]
        for i, pred_single in enumerate(pred_full):
            pred_single_person = pred_single[pred_single[:, 5] == 0]
            pred_single_phone = pred_single[pred_single[:, 5] == 1]
            pred_single_person = pred_single_person[
                pred_single_person[:, 4] > 0.55]
            pred_single_phone = pred_single_phone[pred_single_phone[:,
                                                                    4] > 0.6]

            pred_person_all.append(pred_single_person)
            pred_phone_all.append(pred_single_phone)

            person_usingphone_indices = pred_single_person[:, 9] > 0.55
            person_usingphone_index = person_usingphone_indices.nonzero(
            ).squeeze()

            #########获取使用手机的人的索引
            if isinstance(person_usingphone_index.tolist(), list):
                person_usingphone_index = person_usingphone_index.tolist()
            else:
                person_usingphone_index = [person_usingphone_index.item()]
            #########属性判断无人使用手机，则跳出
            if len(person_usingphone_index
                   ) == 0 or pred_single_phone.shape[0] == 0:
                continue
            person_usingphone_single = pred_single_person[
                person_usingphone_index]
            person_usingphone_single, phone, person_indices = merge_person_phone(
                person_usingphone_single, pred_single_phone)
            person_index = person_indices.nonzero().squeeze()
            if isinstance(person_index.tolist(), list):
                person_index = person_index.tolist()
            else:
                person_index = [person_index.item()]
            person_usingphone_index = list(
                np.array(person_usingphone_index)[person_index])
            if len(person_usingphone_index) == 0:
                continue
            person_usingphone_index_all.extend(person_usingphone_index)
            person_usingphone_all.append(
                pred_single_person[person_usingphone_index])
            phone_usingphone_all.append(phone)
            ########添加图像索引
            for index in range(len(person_usingphone_index)):
                person_usingphone_index_img.append(i)

        if len(person_usingphone_all) > 0:
            person_usingphone_all = torch.cat(person_usingphone_all, dim=0)
            phone_usingphone_all = torch.cat(phone_usingphone_all, dim=0)
            factor_distance_w = 0.145
            factor_distance_h = 0.145
            bz_pose = self.max_bz_pose
            length = len(person_usingphone_index_img)
            batch_ids = np.arange(length) // bz_pose
            results_usingphone = []
            for batch_id in np.unique(batch_ids):
                batch_indices = batch_ids == batch_id
                person = person_usingphone_all[batch_indices]
                phone = phone_usingphone_all[batch_indices]
                image_ids = np.array(
                    person_usingphone_index_img)[batch_indices]
                boxs_person = person[:, :4]
                w, h = boxs_person[:,
                                   2] - boxs_person[:,
                                                    0], boxs_person[:,
                                                                    3] - boxs_person[:,
                                                                                     1]
                boxs_phone = phone[:, :4]
                centers_phone = torch.zeros_like(boxs_phone[:, :2])
                centers_phone[:, 0] = (boxs_phone[:, 0] + boxs_phone[:, 2]) / 2
                centers_phone[:, 1] = (boxs_phone[:, 1] + boxs_phone[:, 3]) / 2
                centers_phone.cpu().numpy()
                if w_factor or h_factor:
                    boxs_person[:, 0] = boxs_person[:, 0] - w * w_factor
                    boxs_person[:, 2] = boxs_person[:, 2] + w * w_factor
                    boxs_person[:, 1] = boxs_person[:, 1] - h * h_factor
                    boxs_person[:, 3] = boxs_person[:, 3] + h * h_factor

                for i, box in enumerate(boxs_person):
                    clip_border(box, imgs[image_ids[i]].shape)
                boxs_person = boxs_person.cpu().numpy()
                input_images = []
                for i in range(person.shape[0]):
                    input_images.append(imgs[image_ids[i]][
                        int(boxs_person[i, 1]):int(boxs_person[i, 3]),
                        int(boxs_person[i, 0]):int(boxs_person[i, 2]), :])

                if self.pt_pose:
                    pose_in_ori_img, pose_scores = self.inference_single_subimg(
                        self.model_pose,
                        input_images,
                        dataset=self.dataset,
                        dataset_info=self.dataset_info,
                        return_heatmap=True)
                else:
                    pose_in_ori_img, pose_scores = self.inference_single_subimg(
                        self.session_pose,
                        input_images,
                        dataset=self.dataset,
                        dataset_info=self.dataset_info,
                        return_heatmap=True)

                for i, pose_single in enumerate(pose_in_ori_img):
                    using_phone_flag = False
                    left_wrist, right_wrist = pose_single
                    box = boxs_person[i]
                    center_phone = centers_phone[i]
                    left_wrist = [
                        left_wrist[0] + box[0], left_wrist[1] + box[1]
                    ]
                    right_wrist = [
                        right_wrist[0] + box[0], right_wrist[1] + box[1]
                    ]

                    if abs(left_wrist[0] -
                           center_phone[0]) < factor_distance_w * w[i] and abs(
                               left_wrist[1] -
                               center_phone[1]) < factor_distance_h * h[i]:
                        using_phone_flag = True

                    elif abs(right_wrist[0] - center_phone[0]
                             ) < factor_distance_w * w[i] and abs(
                                 right_wrist[1] -
                                 center_phone[1]) < factor_distance_h * h[i]:
                        using_phone_flag = True
                    else:
                        using_phone_flag = False
                    if using_phone_flag:
                        results_usingphone.append(1)
                    else:
                        results_usingphone.append(0)

        self.logger.info(f'Done. ({time.time() - t1:.3f}s)')
        results = dict(results=[])
        #####读取错误的路径
        for i, invalid_path_single in enumerate(invalid_path):
            results["results"].append(
                dict(result=dict(person=[]),
                     img_path=invalid_path_single,
                     error="img_path is invalid"))

        for index, pred_single in enumerate(pred_person_all):
            result_single = dict(person=[])
            if pred_single.shape[0]:
                person_usingphone_index_single = []
                if index in person_usingphone_index_img:
                    img_indices = np.array(
                        person_usingphone_index_img) == index
                    result_usingphone = list(
                        np.array(results_usingphone)[img_indices])
                    person_usingphone_index_single = list(
                        np.array(person_usingphone_index_all)[img_indices])

                # ##################################调整分数    目前不引入,因为torch1.8以下版本的topk()会报错，之后将添加新的逻辑
                # pred_single_adjust=pred_single.clone()
                # box = pred_single_adjust[:, :4]
                # overlaps = iou_overlap(box, box)
                # ######取topk2的iou
                # iou_thres_exceed = 0.45
                # value, indices = overlaps.topk(2, dim=0, largest=True, sorted=True)
                # overlaps_topk2 = overlaps[range(overlaps.shape[0]), indices[1]]
                # indexs = (overlaps_topk2 > iou_thres_exceed).nonzero()
                # overlaps_exceed = overlaps_topk2[indexs]
                # ######保存iou超出规范的目标索引
                # weights = dict()
                #
                # ######使用一个非线性函数对iou输出一个衰减值
                # def get_exceed_index(weights=dict(), exceed_index=None, exceed_overlaps=None):
                #     import math
                #     for i in range(exceed_index.shape[0]):
                #         index = exceed_index[i][0].item()
                #         overlap = exceed_overlaps[i][0].item()
                #         weight = (1 - overlap)
                #         weights.setdefault(index, weight)
                #     return weights
                #
                # weights = get_exceed_index(weights, indexs, overlaps_exceed)
                #
                # #####获取所有目标框的面积，将面积处于面积序列0.6分位数以后的边界框进行衰减,暂定衰减系数为一个固定值0.9
                # #####每种不同的属性使用不同的参数进行惩罚
                # def get_area_factor(weights, bboxes):
                #     import math
                #     areas = (bboxes[..., 2] - bboxes[..., 0]) * (bboxes[..., 3] - bboxes[..., 1])
                #     # areas = bboxes[..., 1]
                #     value, indices = areas.sort(dim=0, descending=True)
                #     quantile_06 = max(1, math.floor(areas.shape[0] * 0.6))
                #     indexs = indices[quantile_06:]
                #     for index in indexs:
                #         index = index.item()
                #         if weights.get(index, None):
                #             weights[index] *= 0.9
                #         else:
                #             weights.setdefault(index, 0.9)
                #     return weights
                # weights = get_area_factor(weights, box)
                #
                # ######使用weights对对应目标框的分数进行衰减(玩手机,站立和转向的分数除外)
                # def adjust_score(weights, pred):
                #     for k, v in weights.items():
                #         pred[k][[13]] *= v
                #     return pred
                # predn_single = adjust_score(weights, pred_single)
                # ####################################调整分数

                pred_single = pred_single.cpu().numpy()

                for idx, pred in enumerate(pred_single):
                    result_person = dict()
                    result_person["id"] = idx
                    box = pred[:4]
                    box = [int(i) for i in box]
                    result_person["box"] = box
                    attr_dict = dict()
                    #[x, y, x, y, conf_cls，cls, 转向分数, 转向id，传递, 玩手机，吃东西，注释，起身，趴桌，阅读]
                    attr_dict["direction"] = [
                        pred[7].item() if pred[6] > 0.65 else 0,
                        pred[6].item()
                    ]
                    attr_dict["delivering"] = [
                        1 if pred[8] > 0.65 else 0, pred[8].item()
                    ]
                    if idx in person_usingphone_index_single:
                        if result_usingphone[
                                person_usingphone_index_single.index(idx)] > 0:
                            attr_dict["usingphone"] = [1, float(pred[9])]
                            attr_dict["reading"] = [0, float(pred[14])]
                        else:
                            attr_dict["usingphone"] = [0, float(pred[9])]
                            attr_dict["reading"] = [
                                1 if pred[14] > 0.75 else 0,
                                float(pred[14])
                            ]
                    else:
                        attr_dict["usingphone"] = [0, float(pred[9])]
                        attr_dict["reading"] = [
                            1 if pred[14] > 0.75 else 0,
                            float(pred[14])
                        ]

                    attr_dict["eating"] = [
                        1 if pred[10] > 0.65 else 0,
                        float(pred[10])
                    ]
                    attr_dict["watching"] = [
                        1 if pred[11] > 0.75 else 0,
                        float(pred[11])
                    ]
                    attr_dict["standing"] = [
                        1 if pred[12] > 0.6 else 0,
                        float(pred[12])
                    ]
                    attr_dict["sleeping"] = [
                        1 if pred[13] > 0.7 else 0,
                        float(pred[13])
                    ]
                    result_person["attr"] = attr_dict
                    result_single["person"].append(result_person)
                    result_single["num"] = len(pred_single)

                    if attr_dict["usingphone"][0] == 1:
                        cv2.rectangle(imgs[index],
                                      (int(pred[0]), int(pred[1])),
                                      (int(pred[2]), int(pred[3])),
                                      color=(0, 0, 255))
                        cv2.putText(imgs[index],
                                    "usingphone",
                                    (int(pred[0]) - 4, int(pred[1]) - 4),
                                    fontFace=cv2.FONT_ITALIC,
                                    fontScale=0.5,
                                    color=(0, 0, 255))
                    if attr_dict["standing"][0] == 1:
                        cv2.rectangle(imgs[index],
                                      (int(pred[0]), int(pred[1])),
                                      (int(pred[2]), int(pred[3])),
                                      color=(0, 0, 255))
                        cv2.putText(imgs[index],
                                    "standing",
                                    (int(pred[0]) - 4, int(pred[1]) - 4),
                                    fontFace=cv2.FONT_ITALIC,
                                    fontScale=0.5,
                                    color=(0, 0, 255))
                    if attr_dict["watching"][0] == 1:
                        cv2.rectangle(imgs[index],
                                      (int(pred[0]), int(pred[1])),
                                      (int(pred[2]), int(pred[3])),
                                      color=(0, 0, 255))
                        cv2.putText(imgs[index],
                                    "watching", (box[0] - 4, box[1] - 4),
                                    fontFace=cv2.FONT_ITALIC,
                                    fontScale=0.5,
                                    color=(0, 0, 255))
                    if attr_dict["sleeping"][0] == 1:
                        cv2.rectangle(imgs[index],
                                      (int(pred[0]), int(pred[1])),
                                      (int(pred[2]), int(pred[3])),
                                      color=(0, 0, 255))
                        cv2.putText(imgs[index],
                                    "sleeping",
                                    (int(pred[0]) - 4, int(pred[1]) - 4),
                                    fontFace=cv2.FONT_ITALIC,
                                    fontScale=0.5,
                                    color=(0, 0, 255))
                    if attr_dict["reading"][0] == 1:
                        cv2.rectangle(imgs[index],
                                      (int(pred[0]), int(pred[1])),
                                      (int(pred[2]), int(pred[3])),
                                      color=(0, 0, 255))
                        cv2.putText(imgs[index],
                                    "reading",
                                    (int(pred[0]) - 4, int(pred[1]) - 4),
                                    fontFace=cv2.FONT_ITALIC,
                                    fontScale=0.5,
                                    color=(0, 0, 255))
                    # cv2.rectangle(imgs[index], (int(pred[0]),int(pred[1])),(int(pred[2]),int(pred[3])), color=(0, 0, 255))
                    # cv2.putText(imgs[index], f"{pred[4]:.2f}", (int(pred[0])-4,int(pred[1])-4), fontFace=cv2.FONT_ITALIC,
                    #             fontScale=0.5, color=(0, 0, 255))

                cv2.imwrite(
                    os.path.join(self.save_dir,
                                 os.path.basename(valid_path[index])),
                    imgs[index])
                results["results"].append(
                    dict(result=result_single,
                         img_path=valid_path[index],
                         error=""))

            else:
                results["results"].append(
                    dict(result=result_single,
                         img_path=valid_path[index],
                         error="no person was detected"))

        self.logger.info("finish processing imgs" + info)

        return results

    def inference_single_subimg(self,
                                model,
                                img_or_path,
                                dataset='TopDownCocoDataset',
                                dataset_info=None,
                                return_heatmap=False):
        dataset_name = dataset_info.dataset_name
        #######可翻转的关键点的序号
        flip_pairs = dataset_info.flip_pairs
        batch_data = []
        for img in img_or_path:
            data = {
                'img': img,
                'dataset': dataset_name,
                'ann_info': {
                    'image_size':
                    np.array(self.config_pose.data_cfg['image_size']),
                    'num_joints': self.config_pose.data_cfg['num_joints'],
                    'flip_pairs': flip_pairs
                }
            }
            data = self.test_pipeline(data)
            batch_data.append(data)
        bz = len(img_or_path)
        batch_data = collate(batch_data, samples_per_gpu=bz)
        if self.pt_pose:
            batch_data['img'] = batch_data['img'].to(self.device)
        else:
            batch_data['img'] = batch_data['img'].numpy().astype(np.float32)
        batch_data['img_metas'] = [
            img_metas for img_metas in batch_data['img_metas'].data[0]
        ]
        flip_pairs = []
        pad = []
        shape = []
        for metas_single in batch_data["img_metas"]:
            flip_pairs.append(metas_single["flip_pairs"])
            pad.append(metas_single["pad"])
            shape.append(metas_single["shape"])
        batch_data["img_metas"] = dict(flip_pairs=flip_pairs,
                                       pad=pad,
                                       shape=shape)

        def get_flip(output, flip_pairs):
            #######测试翻转的图片后需要将关键点也进行翻转会原来的图像上
            if flip_pairs is not None:
                output_heatmap = flip_back(
                    output,
                    flip_pairs,
                )
                if self.config_pose.model.test_cfg.get('shift_heatmap', False):
                    output_heatmap[:, :, :, 1:] = output_heatmap[:, :, :, :-1]
            else:
                output_heatmap = output
            return output_heatmap

        if self.pt_pose:
            with torch.no_grad():
                result = model(img=batch_data['img'])
                if self.flip_test:
                    flip_img = batch_data['img'].flip(3)
                    result_flip = model(img=flip_img)
                    result_flip = get_flip(result_flip[0], flip_pairs)
                    result_final = (result + result_flip) * 0.5
                else:
                    result_final = result
                result_final = result_final.cpu().numpy()
        else:
            result = self.session_pose.run(
                [self.session_pose.get_outputs()[0].name],
                {self.session_pose.get_inputs()[0].name: batch_data['img']})
            if self.flip_test:
                flip_img = torch.from_numpy(batch_data['img']).flip(3)
                flip_img = flip_img.numpy().astype(np.float32)
                result_flip = self.session_pose.run(
                    [self.session_pose.get_outputs()[0].name],
                    {self.session_pose.get_inputs()[0].name: flip_img})

                result_flip = get_flip(result_flip[0], flip_pairs)
                result_final = (result[0] + result_flip[0]) * 0.5
            else:
                result_final = result[0]

        batch_size, _, img_height, img_width = batch_data["img"].shape

        keypoint_results = self.decode_subimg(batch_data['img_metas'],
                                              result_final,
                                              img_size=[img_width, img_height])

        preds = []
        scores = []
        for i in range(bz):
            pred = keypoint_results["preds"][i][9:11]
            h, w, _ = img_or_path[i].shape
            pred = [[int(px * w), int(py * h)] for px, py in pred]
            score = keypoint_results["maxvals"][i][9:11].tolist()
            preds.append(pred)
            scores.append(score)

        return preds, scores

    def decode_subimg(self, img_metas, output, **kwargs):
        preds, maxvals = keypoints_from_heatmaps_subimg(
            output,
            img_metas["shape"],
            img_metas["pad"],
            unbiased=self.config_pose.model.test_cfg.get(
                'unbiased_decoding', False),
            post_process=self.config_pose.model.test_cfg.get(
                'post_process', 'default'),
            kernel=self.config_pose.model.test_cfg.get('modulate_kernel', 11),
            valid_radius_factor=self.config_pose.model.test_cfg.get(
                'valid_radius_factor', 0.0546875),
            use_udp=self.config_pose.model.test_cfg.get('use_udp', False),
            target_type=self.config_pose.model.test_cfg.get(
                'target_type', 'GaussianHeatmap'))
        return {"preds": preds, "maxvals": maxvals}


if __name__ == "__main__":
    Detector = Detect(weights=[
        "../model/checkpoint/best.pt", "../model/checkpoint/hrnet.pt"
    ],
                      configs=[
                          "../model/configs/yolo/yolov5/yolov5m.py",
                          "../model/configs/pose/hrnet/hrnet.py"
                      ],
                      iou_thres=0.45,
                      img_size=960,
                      conf_thres=0.25,
                      half=True,
                      device=0,
                      block_num=3,
                      factor=0.15,
                      flip_test=False,
                      logger_name="connector",
                      max_bz_pose=5,
                      save_dir="../res/res_noblock")
    Detector.load_model(Detector.weights, Detector.configs)
    Detector.predict([
        "../imgs/test2/image_001950.jpg", "../imgs/test2/image_001951.jpg",
        "../imgs/test2/image_001952.jpg", "../imgs/test2/image_001953.jpg"
    ])
