import os
import torchvision
import glob
import time
from collections.abc import Mapping, Sequence
import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import default_collate
from dataset.builder import DataContainer

import torch
import numpy as np
import cv2
import warnings
from .config import Config
import functools

def get_config(config_file):
    config=Config.fromfile(config_file)
    return config

# 假设box1维度为[N,4]   box2维度为[M,4]
def Iou(box1, box2):
    N = box1.size(0)
    M = box2.size(0)

    lt = torch.max(  # 左上角的点
        box1[:, :2].unsqueeze(1).expand(N, M, 2),   # [N,2]->[N,1,2]->[N,M,2]
        box2[:, :2].unsqueeze(0).expand(N, M, 2),   # [M,2]->[1,M,2]->[N,M,2]
    )

    rb = torch.min(
        box1[:, 2:].unsqueeze(1).expand(N, M, 2),
        box2[:, 2:].unsqueeze(0).expand(N, M, 2),
    )

    wh = rb - lt  # [N,M,2]
    wh[wh < 0] = 0   # 两个box没有重叠区域
    inter = wh[:,:,0] * wh[:,:,1]   # [N,M]

    area1 = (box1[:,2]-box1[:,0]) * (box1[:,3]-box1[:,1])  # (N,)
    area2 = (box2[:,2]-box2[:,0]) * (box2[:,3]-box2[:,1])  # (M,)
    area1 = area1.unsqueeze(1).expand(N,M)  # (N,M)
    area2 = area2.unsqueeze(0).expand(N,M)  # (N,M)

    iou = inter / (area1+area2-inter)
    return iou

def merge_person(pred_person,full_person):
    if pred_person.shape[0]>0 and full_person.shape[0]>0:
        iou=Iou(full_person[:,:4],pred_person[:,:4])
        max_overlap,indices=iou.max(dim=1)
        add_indices=max_overlap<0.15
        pred_person=torch.cat([pred_person,full_person[add_indices]],dim=0)
    else:
        pred_person=torch.cat([pred_person,full_person],dim=0)
    return pred_person

def merge_person_phone(person,phone):
    iou=miniou(person[:,:4],phone[:,:4])
    max_over,phone_indices=iou.max(dim=1)
    indices=max_over>0.45
    phone_indices=phone_indices[indices]
    return person[indices],phone[phone_indices],indices

def miniou(box1,box2):
    N = box1.size(0)
    M = box2.size(0)

    lt = torch.max(  # 左上角的点
        box1[:, :2].unsqueeze(1).expand(N, M, 2),  # [N,2]->[N,1,2]->[N,M,2]
        box2[:, :2].unsqueeze(0).expand(N, M, 2),  # [M,2]->[1,M,2]->[N,M,2]
    )

    rb = torch.min(
        box1[:, 2:].unsqueeze(1).expand(N, M, 2),
        box2[:, 2:].unsqueeze(0).expand(N, M, 2),
    )

    wh = rb - lt  # [N,M,2]
    wh[wh < 0] = 0  # 两个box没有重叠区域
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # (N,)
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # (M,)
    area1 = area1.unsqueeze(1).expand(N, M)  # (N,M)
    area2 = area2.unsqueeze(0).expand(N, M)  # (N,M)
    #######求重叠区域与最小面积物体的重叠率
    iou = inter / torch.min(area1 ,area2)
    return iou


# NMS算法
# bboxes维度为[N,4]，scores维度为[N,], 均为tensor
def iou_nms(bboxes, scores, score_attr, threshold=0.5,image=None):
    x1 = bboxes[:,0]
    y1 = bboxes[:,1]
    x2 = bboxes[:,2]
    y2 = bboxes[:,3]
    areas = (x2-x1)*(y2-y1)   # [N,] 每个bbox的面积
    _, order = scores.sort(0, descending=True)    # 降序排列

    keep = []
    while order.numel() > 0:       # torch.numel()返回张量元素个数
        if order.numel() == 1:     # 保留框只剩一个
            i = order.item()
            keep.append(i)
            break
        else:
            i = order[0].item()    # 保留scores最大的那个框box[i]
            keep.append(i)

        # 计算box[i]与其余各框的IOU(思路很好)
        xx1 = x1[order[1:]].clamp(min=x1[i])   # [N-1,]
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])
        inter = (xx2-xx1).clamp(min=0) * (yy2-yy1).clamp(min=0)   # [N-1,]

        iou = inter / (areas[i]+areas[order[1:]]-inter)  # [N-1,]
        idx = (iou <= threshold).nonzero().squeeze() # 注意此时idx为[N-1,] 而order为[N,]
        if idx.numel() == 0:
            break
        ###########将与当前目标的iou大于阈值的box去除
        order = order[idx+1]  # 修补索引之间的差值
    return torch.cat([bboxes[keep], scores[keep][:, None], score_attr[keep]], dim=1)


##########去除重复样本的时候需要将去除样本的最大属性分数继承过来
def miniou_nms(bboxes, scores, score_attr, threshold=0.8, image=None, type="max"):

    x1 = bboxes[:,0]
    # print(x1)
    # print( x1.shape)
    # import sys
    # sys.exit()
    y1 = bboxes[:,1]
    x2 = bboxes[:,2]
    y2 = bboxes[:,3]
    areas = (x2-x1)*(y2-y1)   # [N,] 每个bbox的面积
    _, order = scores.sort(0, descending=True)    # 根据分数降序排列，保存原来的索引

    keep = []
    wi=0
    num_box=0
    while order.numel() > 0:       # torch.numel()返回张量元素个数
        if order.numel() == 1:     # 保留框只剩一个
            i = order.item()
            keep.append(i)
            break
        else:
            i = order[0].item()    # 保留scores最大的那个框box[i]
            keep.append(i)

        # image_per=image.copy()
        # 计算box[i]与其余各框的IOU(思路很好)
        xx1 = x1[order[1:]].clamp(min=x1[i])   # [N-1,]
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])
        inter = (xx2-xx1).clamp(min=0) * (yy2-yy1).clamp(min=0)   # [N-1,]

        miniou = inter / torch.min(areas[i],areas[order[1:]])  # [N-1,]
        ###########去除分块检测造成的高iou，因此使用较高的阈值
        idx = (miniou <= threshold).nonzero().squeeze() # 注意此时idx为[N-1,] 而order为[N,]
        ###########获取超过阈值的目标框与当前目标框中最大的目标框，并将其作为最后的框
        throw_idx= (miniou > threshold).nonzero().squeeze()

        if isinstance(throw_idx.tolist(), list):
            throw_idx=throw_idx.tolist()
        else:
            throw_idx=[throw_idx.tolist()]

        if len(throw_idx)>0:
            all_area=[areas[i],*[areas[order[id+1]] for id in throw_idx]]
            if type=="max":
                index_max=all_area.index(max(all_area))
            else:
                index_max = all_area.index(min(all_area))
            if index_max>0:
                save_index=order[(throw_idx[index_max-1])+1].item()
                keep.pop()
                keep.append(save_index)
            # del image_per
            num_box+=(len(throw_idx)+1)
            #########测试
        if idx.numel() == 0:
            break
        order = order[idx+1]  # 修补索引之间的差值
    return torch.cat([bboxes[keep],scores[keep][:,None],score_attr[keep]],dim=1)

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)   确定最小缩放比例
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle      用pad除以stride获得pad大小
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))

    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    #####添加pad
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

def clip_border(box,shape):
    box[0]=int(max(0,min(box[0],shape[1])))
    box[2] = int(max(0,min(box[2], shape[1])))
    box[1] = int(max(0,min(box[1], shape[0])))
    box[3] = int(max(0,min(box[3], shape[0])))

def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        #####求出pad的大小
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    #####将坐标减去pad
    coords[..., [0, 2]] -= pad[0]  # x padding
    coords[..., [1, 3]] -= pad[1]  # y padding
    coords[..., :4] /= gain
    ####限定坐标在图像范围内
    clip_coords(coords, img0_shape)
    return coords


def clip_cordinate(cordinates, shape):
    for sub_cor in cordinates:
        sub_cor[0] = max(0, min(sub_cor[0], shape))
        sub_cor[1] = max(0, min(sub_cor[1], shape))

def xywh2xyxy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

def xyxy2xywh(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2
    y[..., 2] = (x[..., 2] - x[..., 0])
    y[..., 3] = (x[..., 3] - x[..., 1])
    return y


def non_max_suppression_block(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=(), max_det=300):
    """Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    nc = 1 # number of classes
    ######(1,18900)   先用IOU筛选掉较差的目标框
    xc = prediction[..., 4] > conf_thres  # candidates
    # conf_class=0.7
    # xc = (prediction[..., 4] > conf_thres)&(prediction[...,14]>conf_class)  # candidates

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 15), device=prediction.device)] * prediction.shape[0]

    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # 分块pred合成的过程会转换为xyxy类型,此处不需要额外的转换
        box = x[:, :4]

        ################修改处[x,y,w,h,obj,person,phone,正，后，左，右，传递，玩手机，吃东西，注视，起身，趴桌，写作]
        conf=x[:,4:5]
        x[:,5:7]=x[:,5:7]*x[:,4:5]
        #########
        # x[:,5:6]=torch.tensor(1).float()
        # x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
        conf_direction, dir = x[:, 7:11].max(1, keepdim=True)
        conf_cls, cls = x[:, 5:7].max(1, keepdim=True)
        # [x, y, x, y, conf_cls，cls, 转向分数, 转向id，传递, 玩手机，吃东西，注释，起身，趴桌，阅读]
        x = torch.cat((box,conf_cls,cls,conf_direction,dir,x[:, 11:]), 1)[conf.view(-1) > conf_thres]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        #######如果送入nms的目标框数量过多，则排序按分数获取
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        boxes, scores = x[:, :4] , x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS

        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]

        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        # print(output[xi].shape)
        # sys.exit()
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output

class LoadImage:
    def __init__(self, color_type='color', channel_order='rgb'):
        self.color_type = color_type
        self.channel_order = channel_order

    def __call__(self, results):
        if isinstance(results['img_or_path'], str):
            results['image_file'] = results['img_or_path']
            img = cv2.imread(results['img_or_path'])
            if self.channel_order=="rgb":
                img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        elif isinstance(results['img_or_path'], np.ndarray):
            results['image_file'] = ''
            if self.color_type == 'color' and self.channel_order == 'rgb':
                img = cv2.cvtColor(results['img_or_path'], cv2.COLOR_BGR2RGB)
            else:
                img = results['img_or_path']
        else:
            raise TypeError('"img_or_path" must be a numpy array or a str or '
                            'a pathlib.Path object')
        results['img'] = img
        return results




def _gaussian_blur(heatmaps, kernel=11):
    """Modulate heatmap distribution with Gaussian.
     sigma = 0.3*((kernel_size-1)*0.5-1)+0.8
     sigma~=3 if k=17
     sigma=2 if k=11;
     sigma~=1.5 if k=7;
     sigma~=1 if k=3;

    Note:
        batch_size: N
        num_keypoints: K
        heatmap height: H
        heatmap width: W

    Args:
        heatmaps (np.ndarray[N, K, H, W]): model predicted heatmaps.
        kernel (int): Gaussian kernel size (K) for modulation, which should
            match the heatmap gaussian sigma when training.
            K=17 for sigma=3 and k=11 for sigma=2.

    Returns:
        np.ndarray[N, K, H, W]: Modulated heatmap distribution.
    """
    assert kernel % 2 == 1

    border = (kernel - 1) // 2
    batch_size = heatmaps.shape[0]
    num_joints = heatmaps.shape[1]
    height = heatmaps.shape[2]
    width = heatmaps.shape[3]
    for i in range(batch_size):
        for j in range(num_joints):
            origin_max = np.max(heatmaps[i, j])
            dr = np.zeros((height + 2 * border, width + 2 * border),
                          dtype=np.float32)
            dr[border:-border, border:-border] = heatmaps[i, j].copy()
            dr = cv2.GaussianBlur(dr, (kernel, kernel), 0)
            heatmaps[i, j] = dr[border:-border, border:-border].copy()
            heatmaps[i, j] *= origin_max / np.max(heatmaps[i, j])
    return heatmaps

def _get_max_preds(heatmaps):
    """Get keypoint predictions from score maps.

    Note:
        batch_size: N
        num_keypoints: K
        heatmap height: H
        heatmap width: W

    Args:
        heatmaps (np.ndarray[N, K, H, W]): model predicted heatmaps.

    Returns:
        tuple: A tuple containing aggregated results.

        - preds (np.ndarray[N, K, 2]): Predicted keypoint location.
        - maxvals (np.ndarray[N, K, 1]): Scores (confidence) of the keypoints.
    """
    assert isinstance(heatmaps,
                      np.ndarray), ('heatmaps should be numpy.ndarray')
    assert heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    N, K, _, W = heatmaps.shape
    heatmaps_reshaped = heatmaps.reshape((N, K, -1))
    #########获取最大值位置
    idx = np.argmax(heatmaps_reshaped, 2).reshape((N, K, 1))
    #########获取最大得分
    maxvals = np.amax(heatmaps_reshaped, 2).reshape((N, K, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)
    #######将最大值索引转换为坐标   x,y
    preds[:, :, 0] = preds[:, :, 0] % W
    preds[:, :, 1] = preds[:, :, 1] // W

    preds = np.where(np.tile(maxvals, (1, 1, 2)) > 0.0, preds, -1)
    return preds, maxvals


def post_dark_udp(coords, batch_heatmaps, kernel=3):
    """DARK post-pocessing. Implemented by udp. Paper ref: Huang et al. The
    Devil is in the Details: Delving into Unbiased Data Processing for Human
    Pose Estimation (CVPR 2020). Zhang et al. Distribution-Aware Coordinate
    Representation for Human Pose Estimation (CVPR 2020).

    Note:
        batch size: B
        num keypoints: K
        num persons: N
        hight of heatmaps: H
        width of heatmaps: W
        B=1 for bottom_up paradigm where all persons share the same heatmap.
        B=N for top_down paradigm where each person has its own heatmaps.

    Args:
        coords (np.ndarray[N, K, 2]): Initial coordinates of human pose.
        batch_heatmaps (np.ndarray[B, K, H, W]): batch_heatmaps
        kernel (int): Gaussian kernel size (K) for modulation.

    Returns:
        res (np.ndarray[N, K, 2]): Refined coordinates.
    """
    if not isinstance(batch_heatmaps, np.ndarray):
        batch_heatmaps = batch_heatmaps.cpu().numpy()
    B, K, H, W = batch_heatmaps.shape
    N = coords.shape[0]
    assert (B == 1 or B == N)
    for heatmaps in batch_heatmaps:
        for heatmap in heatmaps:
            cv2.GaussianBlur(heatmap, (kernel, kernel), 0, heatmap)
    np.clip(batch_heatmaps, 0.001, 50, batch_heatmaps)
    np.log(batch_heatmaps, batch_heatmaps)
    batch_heatmaps = np.transpose(batch_heatmaps,
                                  (2, 3, 0, 1)).reshape(H, W, -1)
    batch_heatmaps_pad = cv2.copyMakeBorder(
        batch_heatmaps, 1, 1, 1, 1, borderType=cv2.BORDER_REFLECT)
    batch_heatmaps_pad = np.transpose(
        batch_heatmaps_pad.reshape(H + 2, W + 2, B, K),
        (2, 3, 0, 1)).flatten()

    index = coords[..., 0] + 1 + (coords[..., 1] + 1) * (W + 2)
    index += (W + 2) * (H + 2) * np.arange(0, B * K).reshape(-1, K)
    index = index.astype(int).reshape(-1, 1)
    i_ = batch_heatmaps_pad[index]
    ix1 = batch_heatmaps_pad[index + 1]
    iy1 = batch_heatmaps_pad[index + W + 2]
    ix1y1 = batch_heatmaps_pad[index + W + 3]
    ix1_y1_ = batch_heatmaps_pad[index - W - 3]
    ix1_ = batch_heatmaps_pad[index - 1]
    iy1_ = batch_heatmaps_pad[index - 2 - W]

    dx = 0.5 * (ix1 - ix1_)
    dy = 0.5 * (iy1 - iy1_)
    derivative = np.concatenate([dx, dy], axis=1)
    derivative = derivative.reshape(N, K, 2, 1)
    dxx = ix1 - 2 * i_ + ix1_
    dyy = iy1 - 2 * i_ + iy1_
    dxy = 0.5 * (ix1y1 - ix1 - iy1 + i_ + i_ - ix1_ - iy1_ + ix1_y1_)
    hessian = np.concatenate([dxx, dxy, dxy, dyy], axis=1)
    hessian = hessian.reshape(N, K, 2, 2)
    hessian = np.linalg.inv(hessian + np.finfo(np.float32).eps * np.eye(2))
    coords -= np.einsum('ijmn,ijnk->ijmk', hessian, derivative).squeeze()
    return coords

def _taylor(heatmap, coord):
    """Distribution aware coordinate decoding method.

    Note:
        heatmap height: H
        heatmap width: W

    Args:
        heatmap (np.ndarray[H, W]): Heatmap of a particular joint type.
        coord (np.ndarray[2,]): Coordinates of the predicted keypoints.

    Returns:
        np.ndarray[2,]: Updated coordinates.
    """
    H, W = heatmap.shape[:2]
    px, py = int(coord[0]), int(coord[1])
    if 1 < px < W - 2 and 1 < py < H - 2:
        dx = 0.5 * (heatmap[py][px + 1] - heatmap[py][px - 1])
        dy = 0.5 * (heatmap[py + 1][px] - heatmap[py - 1][px])
        dxx = 0.25 * (
            heatmap[py][px + 2] - 2 * heatmap[py][px] + heatmap[py][px - 2])
        dxy = 0.25 * (
            heatmap[py + 1][px + 1] - heatmap[py - 1][px + 1] -
            heatmap[py + 1][px - 1] + heatmap[py - 1][px - 1])
        dyy = 0.25 * (
            heatmap[py + 2 * 1][px] - 2 * heatmap[py][px] +
            heatmap[py - 2 * 1][px])
        derivative = np.array([[dx], [dy]])
        hessian = np.array([[dxx, dxy], [dxy, dyy]])
        if dxx * dyy - dxy**2 != 0:
            hessianinv = np.linalg.inv(hessian)
            offset = -hessianinv @ derivative
            offset = np.squeeze(np.array(offset.T), axis=0)
            coord += offset
    return coord

def keypoints_from_heatmaps_subimg(heatmaps,
                                   shape=None,
                                   pad=None,
                                   unbiased=False,
                                   post_process='default',
                                   kernel=11,
                                   valid_radius_factor=0.0546875,
                                   use_udp=False,
                                   target_type='GaussianHeatmap'):

    heatmaps = heatmaps.copy()
    # print(post_process, use_udp, unbiased, heatmaps.shape, shape, pad)

    if unbiased:
        assert post_process not in [False, None, 'megvii']
    if post_process in ['megvii', 'unbiased']:
        assert kernel > 0
    if use_udp:
        assert not post_process == 'megvii'

    if post_process is False:
        warnings.warn(
            'post_process=False is deprecated, '
            'please use post_process=None instead', DeprecationWarning)
        post_process = None
    elif post_process is True:
        if unbiased is True:
            warnings.warn(
                'post_process=True, unbiased=True is deprecated,'
                " please use post_process='unbiased' instead",
                DeprecationWarning)
            post_process = 'unbiased'
        else:
            warnings.warn(
                'post_process=True, unbiased=False is deprecated, '
                "please use post_process='default' instead",
                DeprecationWarning)
            post_process = 'default'
    elif post_process == 'default':
        if unbiased is True:
            warnings.warn(
                'unbiased=True is deprecated, please use '
                "post_process='unbiased' instead", DeprecationWarning)
            post_process = 'unbiased'


    if post_process == 'megvii':
        heatmaps = _gaussian_blur(heatmaps, kernel=kernel)

    N, K, H, W = heatmaps.shape
    #### (default False False,..)
    # print(post_process, use_udp, unbiased, heatmaps.shape, shape, pad)
    # print(heatmaps.shape)
    # import sys
    # sys.exit()

    if use_udp:
        if target_type.lower() == 'GaussianHeatMap'.lower():
            preds, maxvals = _get_max_preds(heatmaps)

            preds = post_dark_udp(preds, heatmaps, kernel=kernel)
        elif target_type.lower() == 'CombinedTarget'.lower():
            for person_heatmaps in heatmaps:
                for i, heatmap in enumerate(person_heatmaps):
                    kt = 2 * kernel + 1 if i % 3 == 0 else kernel
                    cv2.GaussianBlur(heatmap, (kt, kt), 0, heatmap)
            # valid radius is in direct proportion to the height of heatmap.
            valid_radius = valid_radius_factor * H
            offset_x = heatmaps[:, 1::3, :].flatten() * valid_radius
            offset_y = heatmaps[:, 2::3, :].flatten() * valid_radius
            heatmaps = heatmaps[:, ::3, :]
            preds, maxvals = _get_max_preds(heatmaps)
            index = preds[..., 0] + preds[..., 1] * W
            index += W * H * np.arange(0, N * K / 3)
            index = index.astype(int).reshape(N, K // 3, 1)
            preds += np.concatenate((offset_x[index], offset_y[index]), axis=2)
        else:
            raise ValueError('target_type should be either '
                             "'GaussianHeatmap' or 'CombinedTarget'")
    else:
        preds, maxvals = _get_max_preds(heatmaps)
        if post_process == 'unbiased':  # alleviate biased coordinate
            # apply Gaussian distribution modulation.
            heatmaps = np.log(
                np.maximum(_gaussian_blur(heatmaps, kernel), 1e-10))
            for n in range(N):
                for k in range(K):
                    preds[n][k] = _taylor(heatmaps[n][k], preds[n][k])
        elif post_process is not None:
            # add +/-0.25 shift to the predicted locations for higher acc.
            for n in range(N):
                for k in range(K):
                    heatmap = heatmaps[n][k]
                    ###########获取每个关节最大值坐标,x,y
                    px = int(preds[n][k][0])
                    py = int(preds[n][k][1])
                    if 1 < px < W - 1 and 1 < py < H - 1:
                        ######最大值与周围点的差值
                        diff = np.array([
                            heatmap[py][px + 1] - heatmap[py][px - 1],
                            heatmap[py + 1][px] - heatmap[py - 1][px]
                        ])

                        ################注意该处会减去pad的大小，获得关键点在原图中的位置
                        ######增加一个偏移，根据左右，上下位置的差值
                        preds[n][k] += np.sign(diff) * .25
                        ######减去pad,注意stride=4
                        preds[n][k] -= [pad[n][0]/4,pad[n][1]/4]
                        preds[n][k] = [preds[n][k][0]*4/shape[n][0],preds[n][k][1]*4/shape[n][1]]
                        if post_process == 'megvii':
                            preds[n][k] += 0.5
    return preds, maxvals



class DatasetInfo:
    def __init__(self, dataset_info):
        self._dataset_info = dataset_info
        self.dataset_name = self._dataset_info['dataset_name']
        self.paper_info = self._dataset_info['paper_info']
        self.keypoint_info = self._dataset_info['keypoint_info']
        self.skeleton_info = self._dataset_info['skeleton_info']
        self.joint_weights = np.array(
            self._dataset_info['joint_weights'], dtype=np.float32)[:, None]

        self.sigmas = np.array(self._dataset_info['sigmas'])

        self._parse_keypoint_info()
        self._parse_skeleton_info()

    def _parse_skeleton_info(self):
        """Parse skeleton information.

        - link_num (int): number of links.
        - skeleton (list((2,))): list of links (id).
        - skeleton_name (list((2,))): list of links (name).
        - pose_link_color (np.ndarray): the color of the link for
            visualization.
        """
        self.link_num = len(self.skeleton_info.keys())
        self.pose_link_color = []

        self.skeleton_name = []
        self.skeleton = []
        for skid in self.skeleton_info.keys():
            link = self.skeleton_info[skid]['link']
            self.skeleton_name.append(link)
            self.skeleton.append([
                self.keypoint_name2id[link[0]], self.keypoint_name2id[link[1]]
            ])
            self.pose_link_color.append(self.skeleton_info[skid].get(
                'color', [255, 128, 0]))
        self.pose_link_color = np.array(self.pose_link_color)

    def _parse_keypoint_info(self):
        """Parse keypoint information.

        - keypoint_num (int): number of keypoints.
        - keypoint_id2name (dict): mapping keypoint id to keypoint name.
        - keypoint_name2id (dict): mapping keypoint name to keypoint id.
        - upper_body_ids (list): a list of keypoints that belong to the
            upper body.
        - lower_body_ids (list): a list of keypoints that belong to the
            lower body.
        - flip_index (list): list of flip index (id)
        - flip_pairs (list((2,))): list of flip pairs (id)
        - flip_index_name (list): list of flip index (name)
        - flip_pairs_name (list((2,))): list of flip pairs (name)
        - pose_kpt_color (np.ndarray): the color of the keypoint for
            visualization.
        """

        self.keypoint_num = len(self.keypoint_info.keys())
        self.keypoint_id2name = {}
        self.keypoint_name2id = {}

        self.pose_kpt_color = []
        self.upper_body_ids = []
        self.lower_body_ids = []

        self.flip_index_name = []
        self.flip_pairs_name = []

        for kid in self.keypoint_info.keys():

            keypoint_name = self.keypoint_info[kid]['name']
            self.keypoint_id2name[kid] = keypoint_name
            self.keypoint_name2id[keypoint_name] = kid
            self.pose_kpt_color.append(self.keypoint_info[kid].get(
                'color', [255, 128, 0]))

            type = self.keypoint_info[kid].get('type', '')
            if type == 'upper':
                self.upper_body_ids.append(kid)
            elif type == 'lower':
                self.lower_body_ids.append(kid)
            else:
                pass

            swap_keypoint = self.keypoint_info[kid].get('swap', '')
            if swap_keypoint == keypoint_name or swap_keypoint == '':
                self.flip_index_name.append(keypoint_name)
            else:
                self.flip_index_name.append(swap_keypoint)
                if [swap_keypoint, keypoint_name] not in self.flip_pairs_name:
                    self.flip_pairs_name.append([keypoint_name, swap_keypoint])

        self.flip_pairs = [[
            self.keypoint_name2id[pair[0]], self.keypoint_name2id[pair[1]]
        ] for pair in self.flip_pairs_name]
        self.flip_index = [
            self.keypoint_name2id[name] for name in self.flip_index_name
        ]
        self.pose_kpt_color = np.array(self.pose_kpt_color)




def flip_back(output_flipped, flip_pairs, target_type='GaussianHeatmap'):
    """Flip the flipped heatmaps back to the original form.

    Note:
        batch_size: N
        num_keypoints: K
        heatmap height: H
        heatmap width: W

    Args:
        output_flipped (np.ndarray[N, K, H, W]): The output heatmaps obtained
            from the flipped images.
        flip_pairs (list[tuple()): Pairs of keypoints which are mirrored
            (for example, left ear -- right ear).
        target_type (str): GaussianHeatmap or CombinedTarget

    Returns:
        np.ndarray: heatmaps that flipped back to the original image
    """
    assert output_flipped.ndim == 4, \
        'output_flipped should be [batch_size, num_keypoints, height, width]'
    shape_ori = output_flipped.shape
    channels = 1

    if target_type.lower() == 'CombinedTarget'.lower():
        channels = 3
        output_flipped[:,1::3, ...] = -output_flipped[:, 1::3, ...]
    output_flipped = output_flipped.reshape(shape_ori[0], -1, channels,
                                            shape_ori[2], shape_ori[3])
    # print(shape_ori,output_flipped.shape)

    output_flipped_back = output_flipped.copy()

    # Swap left-right parts
    for left, right in flip_pairs:
        output_flipped_back[:, left, ...] = output_flipped[:, right, ...]
        output_flipped_back[:, right, ...] = output_flipped[:, left, ...]
    output_flipped_back = output_flipped_back.reshape(shape_ori)
    # Flip horizontally  调换配对的顺序，然后水平翻转
    output_flipped_back = output_flipped_back[..., ::-1]
    return output_flipped_back

from dataset.builder import DataContainer


def collate(batch, samples_per_gpu=1):
    # # batch_size 2 2 dict_keys(['img_metas', 'img', 'gt_bboxes', 'gt_labels'])
    # print("batch_size",samples_per_gpu,len(batch),batch[0].keys())

    if not isinstance(batch, Sequence):
        raise TypeError(f'{batch.dtype} is not supported.')

    if isinstance(batch[0], DataContainer):
        stacked = []
        if batch[0].cpu_only:
            # print("cpu_only")
            for i in range(0, len(batch), samples_per_gpu):
                stacked.append(
                    [sample.data for sample in batch[i:i + samples_per_gpu]])
            return DataContainer(
                stacked, batch[0].stack, batch[0].padding_value, cpu_only=True)
        #########对于图像进行stack,需要对图像pad到固定的大小
        elif batch[0].stack:
            for i in range(0, len(batch), samples_per_gpu):
                assert isinstance(batch[i].data, torch.Tensor)

                if batch[i].pad_dims is not None:
                    ##########ndim=3
                    ndim = batch[i].dim()
                    assert ndim > batch[i].pad_dims
                    max_shape = [0 for _ in range(batch[i].pad_dims)]
                    for dim in range(1, batch[i].pad_dims + 1):
                        ######获取每张图片的[w,h]
                        max_shape[dim - 1] = batch[i].size(-dim)
                    # print("shape",max_shape)
                    #########对每个gpu采样的数据分别先组合
                    for sample in batch[i:i + samples_per_gpu]:
                        for dim in range(0, ndim - batch[i].pad_dims):
                            ######判断除了填充维度以外的其他维度的大小是否一致
                            assert batch[i].size(dim) == sample.size(dim)
                            # print(batch[i].size(dim),dim)
                        ##################遍历该gpu上所有采样的图像，获取最大的w,h,
                        ##################并根据最大的w,h,对每张图像padding
                        for dim in range(1, batch[i].pad_dims + 1):
                            max_shape[dim - 1] = max(max_shape[dim - 1],
                                                     sample.size(-dim))
                    ############根据最大的w,h,对每张图像padding
                    padded_samples = []
                    for sample in batch[i:i + samples_per_gpu]:
                        pad = [0 for _ in range(batch[i].pad_dims * 2)]
                        for dim in range(1, batch[i].pad_dims + 1):
                            pad[2 * dim -
                                1] = max_shape[dim - 1] - sample.size(-dim)
                        # print("pad",pad,sample.size(),max_shape)
                        ##############填充顺序为上下左右，因为将图像放在左上角，因此只对右，下填充(不影响bboxes)
                        padded_samples.append(
                            F.pad(
                                sample.data, pad, value=sample.padding_value))
                    #######使用pytorch默认函数对每个GPU的数据进行collect，然后将所有gpu汇总的数据放入列表中,
                    ##############列表中的每个元素为每个GPU的输入数据

                    stacked.append(default_collate(padded_samples))
                elif batch[i].pad_dims is None:
                    stacked.append(
                        default_collate([
                            sample.data
                            for sample in batch[i:i + samples_per_gpu]
                        ]))
                else:
                    raise ValueError(
                        'pad_dims should be either None or integers (1-3)')

        else:
            # print("cant collect")
            for i in range(0, len(batch), samples_per_gpu):
                stacked.append(
                    [sample.data for sample in batch[i:i + samples_per_gpu]])
        return DataContainer(stacked, batch[0].stack, batch[0].padding_value)
    elif isinstance(batch[0], Sequence):
        #######将不同图像的同一个测试尺度的图片放在一起
        transposed = zip(*batch)
        return [collate(samples, samples_per_gpu) for samples in transposed]
    elif isinstance(batch[0], Mapping):
        return {
            key: collate([d[key] for d in batch], samples_per_gpu)
            for key in batch[0]
        }
    else:
        # print(4)
        return default_collate(batch)
