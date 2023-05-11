# Model validation metrics

import warnings
from pathlib import Path

import math
import matplotlib.pyplot as plt
import numpy as np
import torch

import sys


def fitness(x):
    # Model fitness as a weighted combination of metrics
    w = [0.0, 0.0, 0.1, 0.9]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
    return (x[:, :4] * w).sum(1)


def ap_per_class_mine(tp, conf, pred_cls, target_cls, plot=False, save_dir='.', names=()):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
        plot:  Plot precision-recall curve at mAP@0.5
        save_dir:  Plot save directory
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness   降序排列
    i = np.argsort(-conf)
    ########按照得分的降序排列
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes    获取目标的类别
    unique_classes = np.unique(target_cls)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    # py=[]
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))

    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        #####每个类物体的个数
        # print(target_cls==c)
        # print(target_cls.sum(),type(target_cls[0]))
        # import sys
        # sys.exit()
        n_l = (target_cls == c).sum()  # number of labels
        ######每个类检测框的数目
        n_p = i.sum()  # number of predictions

        if n_p == 0 or n_l == 0:
            continue
        else:
            # Accumulate FPs and TPs    在不同的iou阈值上未检测成功的目标框个数
            fpc = (1 - tp[i]).cumsum(0)
            ##########计算在不同的iou阈值上，成功检测的目标框
            tpc = tp[i].cumsum(0)

            # Recall   召回率
            recall = tpc / (n_l + 1e-16)  # recall curve

            # length=tpc.shape[0]
            # px=np.linspace(0, 1, length)
            # print(px.shape,conf[i].shape,recall.shape)
            # print(px,conf[i])
            # print(recall)
            # sr= np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases
            r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases
            # Precision   准确率为
            precision = tpc / (tpc + fpc)  # precision curve
            ############计算
            p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

            # print(-conf[i])
            # print(precision[:,0])
            # import sys
            # sys.exit()

            plot=True
            # ap[ci, -1], mpre, mrec = compute_ap(recall[:, -1], precision[:, -1])
            # import sys
            # sys.exit()
            # AP from recall-precision curve   计算每个类别在每个iou阈值处的pr曲线的组成点，便可以得到由(recall,pre)组成的pr曲线
            for j in range(tp.shape[1]):
                ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
                if plot and j == 0:
                    ######获取IOU为0.5的pr曲线，更顺滑，因为插值点更多
                    py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + 1e-16)

    if plot:
        names=["no sleeping","sleeping","incertitude"]
        plot_pr_curve(px, py, ap, Path(save_dir) / 'PR_curve.png', names)
        ###############均为IOU0.5下的曲线
        plot_mc_curve(px, f1, Path(save_dir) / 'F1_curve.png', names, ylabel='F1')
        plot_mc_curve(px, p, Path(save_dir) / 'P_curve.png', names, ylabel='Precision')
        plot_mc_curve(px, r, Path(save_dir) / 'R_curve.png', names, ylabel='Recall')

    i = f1.mean(0).argmax()  # max F1 index   获取一个平衡pr的分数，由此确定用于分类的分数阈值
    return p[:, i], r[:, i], ap, f1[:, i], unique_classes.astype('int32')




def ap_per_class(tp, conf, pred_cls, target_cls, plot=False, save_dir='.', names=(),index=0):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
        plot:  Plot precision-recall curve at mAP@0.5
        save_dir:  Plot save directory
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness   降序排列
    i = np.argsort(-conf)
    ########按照得分的降序排列
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes    获取目标的类别
    unique_classes = np.unique(target_cls)
    # print(unique_classes)
    # import sys
    # sys.exit()
    # 如果是单个类别的话，unique_cls只有1
    if index>1:
        unique_classes = np.array([1])

    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    # py=[]
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    py_flag=0
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        #####每个类物体的个数
        # print(target_cls==c)
        # print(target_cls.sum(),type(target_cls[0]))
        # import sys
        # sys.exit()
        n_l = (target_cls == c).sum()  # number of labels
        ######每个类检测框的数目
        n_p = i.sum()  # number of predictions

        if n_p == 0 or n_l == 0:
            continue
        else:
            py_flag+=1
            # Accumulate FPs and TPs    在不同的iou阈值上未检测成功的目标框个数
            fpc = (1 - tp[i]).cumsum(0)
            ##########计算在不同的iou阈值上，成功检测的目标框
            tpc = tp[i].cumsum(0)
            # Recall   召回率
            recall = tpc / (n_l + 1e-16)  # recall curve

            # length=tpc.shape[0]
            # px=np.linspace(0, 1, length)
            # print(px.shape,conf[i].shape,recall.shape)
            # print(px,conf[i])
            # print(recall)
            # sr= np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases
            r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases
            # Precision   准确率为
            precision = tpc / (tpc + fpc)  # precision curve
            ############计算
            p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score


            plot=True
            # ap[ci, -1], mpre, mrec = compute_ap(recall[:, -1], precision[:, -1])
            # import sys
            # sys.exit()
            # AP from recall-precision curve   计算每个类别在每个iou阈值处的pr曲线的组成点，便可以得到由(recall,pre)组成的pr曲线
            for j in range(tp.shape[1]):
                ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
                if plot and j == 0:
                    ######获取IOU为0.5的pr曲线，更顺滑，因为插值点更多
                    py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + 1e-16)
    plot=True
    if plot and py_flag==nc:
        plot_pr_curve(px, py, ap, Path(save_dir) / 'PR_curve.png', names)
        ###############均为IOU0.5下的曲线
        plot_mc_curve(px, f1, Path(save_dir) / f'F1_curve{index}.png', names, ylabel='F1')
        plot_mc_curve(px, p, Path(save_dir) / f'P_curve{index}.png', names, ylabel='Precision')
        plot_mc_curve(px, r, Path(save_dir) / f'R_curve{index}.png', names, ylabel='Recall')

    i = f1.mean(0).argmax()  # max F1 index   获取一个平衡pr的分数，由此确定用于分类的分数阈值
    return p[:, i], r[:, i], ap, f1[:, i], unique_classes.astype('int32')


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
    # Returns
        Average precision, precision curve, recall curve
    """
    # print(recall)
    # print(precision)
    import sys

    # Append sentinel values to beginning and end
    ############mrec随着分数降序变化
    mrec = np.concatenate(([0.], recall, [recall[-1] + 0.01]))

    #######mpre是随着分数降序变化的
    mpre = np.concatenate(([1.], precision, [0.]))
    ######将分数倒序
    # print(np.flip(mpre))
    # print(np.maximum.accumulate(np.flip(mpre)))
    # print(np.flip(np.maximum.accumulate(np.flip(mpre))))
    # sys.exit()

    # Compute the precision envelope
    #######计算累积最大值
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        #######横坐标recall计算101个点，计算recall的插值[0,1]
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        # x = np.linspace(0, 1, 10)  # 101-point interp (COCO)
        # print(np.interp(x,mrec,mpre))
        # print(x)
        # print(mrec)
        # print(mpre)
        # sys.exit()
        ############根据recall,precision根据分数降序的对应关系，求出101个recall插值处precision的值，
        #####由此得到pr曲线
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec


class ConfusionMatrix:
    # Updated version of https://github.com/kaanakan/object_detection_confusion_matrix
    def __init__(self, nc, conf=0.25, iou_thres=0.45):
        self.matrix = np.zeros((nc + 1, nc + 1))
        self.nc = nc  # number of classes
        self.conf = conf
        self.iou_thres = iou_thres
        self.class_cont=[]
        if self.nc==1:
            self.boxs=[]

    def process_batch(self, detections, labels,image=None,index=None):
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            detections (Array[N, 6]), x1, y1, x2, y2, conf, class
            labels (Array[M, 5]), class, x1, y1, x2, y2
        Returns:
            None, updates confusion matrix accordingly
        """
        #######先通过分数筛选[x,y,x,y,分数,类别,转向分数,转向id，传递，玩手机，吃东西，注释，起身，趴桌，阅读]，需要修改
        detections = detections[detections[:, 4] > self.conf]
        gt_classes = labels[:, 0].int()
        self.class_cont.extend(list(labels[:,0].cpu().numpy().astype(np.int8)))
        if self.nc==1:
            boxs = [[], [], [], []]

        ######无需修改，[x,y,x,y,分数,类别,转向分数,转向id，传递，玩手机，吃东西，注释，起身，趴桌，阅读]
        detection_classes = detections[:, 5].int()
        ######[label_num,detect_num]
        iou = box_iou(labels[:, 1:], detections[:, :4])
        x = torch.where(iou > self.iou_thres)
        # print(len(detections))

        if x[0].shape[0]:
            ##########[label_id,detect_id,iou]
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            # print(len(detections))
            # print(len(labels))
            # print(x)
            # print(iou)
            # print(matches)
            # print(gt_classes)

            if x[0].shape[0] > 1:
                #######按照iou降序排列
                matches = matches[matches[:, 2].argsort()[::-1]]

                ##############剔除重复对应的detection，注意都是按照iou的顺序保留与GT最高iou的detection(存在多个box与同一个gt匹配)
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]

                matches = matches[matches[:, 2].argsort()[::-1]]
                ##########剔除重复的对应的gt,保留与detection最高iou的GT
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                # print(np.unique(matches[:, 0], return_index=True))
                # print(matches)
                # sys.exit()
        else:
            matches = np.zeros((0, 3))

        n = matches.shape[0] > 0
        #########m0为gt序号，m1为detection序号
        m0, m1, _ = matches.transpose().astype(np.int16)
        # if matches.shape[0]>0:
        #     print(m0,m1)
        #     print(matches)
            # sys.exit()

        for i, gc in enumerate(gt_classes):
            #######获取m0中对应的gt和标签
            j = m0 == i
            # if matches.shape[0] > 0:
            #     print(i,j,gc.item(),m0)

            # print("label",i,gc)
            if n and sum(j) == 1:
                #############将gc类别实例分类为m1[j]的,
                # print(detection_classes[m1[j]],gc)
                self.matrix[detection_classes[m1[j]], gc] += 1  # correct
                if detection_classes[m1[j]]==1:
                    if self.nc==1:
                        if gc==1:
                            boxs[3].append(labels[i][1:])
                        else:
                            boxs[2].append(labels[i][1:])

            else:
                ##########修改由于我们的标签是：未睡觉0，睡觉1，因此需要修改
                self.matrix[0, gc] += 1  # background FP
                if gc==1:
                    if self.nc==1:
                        boxs[1].append(labels[i][1:])


        if n:
            ########将背景加入FP
            for i, dc in enumerate(detection_classes):
                # print(i,dc,m1)
                # print(any(m1 == i))
                if not any(m1 == i):
                    ###################################修改,由于我们的标签是：未睡觉0，睡觉1，因此需要修改
                    ########被剔除的detection，因为其对应的gt与多个detection匹配，且其余的detection与gt的iou更高
                    self.matrix[dc, 0] += 1  # background FN
                    if dc==1:
                        if self.nc==1:
                            boxs[2].append(detections[i][:4])
        if self.nc==1:
            self.boxs.append(boxs)

    def process_batch_mine2(self, detections, labels,image=None):
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            detections (Array[N, 6]), x1, y1, x2, y2, conf, class
            labels (Array[M, 5]), class, x1, y1, x2, y2
        Returns:
            None, updates confusion matrix accordingly
        """
        #######先通过分数筛选
        detections = detections[detections[:, 4] > self.conf]
        gt_classes = labels[:, 0].int()
        self.class_cont.extend(list(labels[:,0].cpu().numpy().astype(np.int8)))
        boxs = [[], [], [], []]

        ########修改计算AP的类别，[x,y,x,y,分数,属性值]
        detection_classes = detections[:, 5].int()


        ######[label_num,detect_num]
        iou = box_iou(labels[:, 1:], detections[:, :4])

        x = torch.where(iou > self.iou_thres)

        if x[0].shape[0]:
            ##########[label_id,detect_id,iou]
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                #######按照iou降序排列
                matches = matches[matches[:, 2].argsort()[::-1]]
                ##############剔除重复对应的detection，注意都是按照iou的顺序保留与GT最高iou的detection
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                ##########剔除重复的对应的gt,保留与detection最高iou的GT
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                # print(np.unique(matches[:, 0], return_index=True))
                # print(matches)
                # sys.exit()
        else:
            matches = np.zeros((0, 3))


        n = matches.shape[0] > 0
        m0, m1, _ = matches.transpose().astype(np.int16)
        # if matches.shape[0]>0:
        #     print(m0,m1)
        #     print(matches)


        for i, gc in enumerate(gt_classes):
            #######获取m0中对应的gt和标签
            j = m0 == i
            # print("label",i,gc)
            if n and sum(j) == 1:
                #############将gc类别实例分类为m1[j]的,
                # print(detection_classes[m1[j]],gc)
                self.matrix[detection_classes[m1[j]], gc] += 1  # correct

            else:
                ##########修改由于我们的标签是：未睡觉0，睡觉1，因此需要修改
                self.matrix[self.nc, gc] += 1  # background FP


        if n:
            for i, dc in enumerate(detection_classes):
                if not any(m1 == i):
                    self.matrix[dc, self.nc] += 1  # background FN
        self.boxs.append(boxs)




    def matrix(self):
        return self.matrix

    def plot(self, normalize=True, save_dir='', names=()):
        try:
            import seaborn as sn

            array = self.matrix / ((self.matrix.sum(0).reshape(1, -1) + 1E-6) if normalize else 1)  # normalize columns
            array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)

            fig = plt.figure(figsize=(12, 9), tight_layout=True)
            sn.set(font_scale=1.0 if self.nc < 50 else 0.8)  # for label size
            labels = (0 < len(names) < 99) and len(names) == self.nc  # apply names to ticklabels
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')  # suppress empty matrix RuntimeWarning: All-NaN slice encountered
                sn.heatmap(array, annot=self.nc < 30, annot_kws={"size": 8}, cmap='Blues', fmt='.2f', square=True,
                           xticklabels=names + ['background FP'] if labels else "auto",
                           yticklabels=names + ['background FN'] if labels else "auto").set_facecolor((1, 1, 1))
            fig.axes[0].set_xlabel('True')
            fig.axes[0].set_ylabel('Predicted')
            fig.savefig(Path(save_dir) / 'confusion_matrix.png', dpi=250)
        except Exception as e:
            print(f'WARNING: ConfusionMatrix plot failure: {e}')

    def print(self):
        for i in range(self.nc + 1):
            print(' '.join(map(str, self.matrix[i])))


def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def bbox_ioa(box1, box2, eps=1E-7):
    """ Returns the intersection over box2 area given box1, box2. Boxes are x1y1x2y2
    box1:       np.array of shape(4)
    box2:       np.array of shape(nx4)
    returns:    np.array of shape(n)
    """

    box2 = box2.transpose()

    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

    # Intersection area
    inter_area = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * \
                 (np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)).clip(0)

    # box2 area
    box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + eps

    # Intersection over box2 area
    return inter_area / box2_area


def wh_iou(wh1, wh2):
    # Returns the nxm IoU matrix. wh1 is nx2, wh2 is mx2
    wh1 = wh1[:, None]  # [N,1,2]
    wh2 = wh2[None]  # [1,M,2]
    inter = torch.min(wh1, wh2).prod(2)  # [N,M]
    return inter / (wh1.prod(2) + wh2.prod(2) - inter)  # iou = inter / (area1 + area2 - inter)


# Plots ----------------------------------------------------------------------------------------------------------------

def plot_pr_curve(px, py, ap, save_dir='pr_curve.png', names=()):
    # Precision-recall curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = np.stack(py, axis=1)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py.T):
            ax.plot(px, y, linewidth=1, label=f'{names[i]} {ap[i, 0]:.3f}')  # plot(recall, precision)
    else:
        ax.plot(px, py, linewidth=1, color='grey')  # plot(recall, precision)

    ax.plot(px, py.mean(1), linewidth=3, color='blue', label='all classes %.3f mAP@0.5' % ap[:, 0].mean())
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(Path(save_dir), dpi=250)


def plot_mc_curve(px, py, save_dir='mc_curve.png', names=(), xlabel='Confidence', ylabel='Metric'):
    # Metric-confidence curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    #############如果类别少于21，则分别绘制给与不同的label
    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
    # if 0 < len(names) < 100:  # display per-class legend if < 21 classes
        for i, y in enumerate(py):
            ax.plot(px, y, linewidth=1, label=f'{names[i]}')  # plot(confidence, metric)
    else:
        ax.plot(px, py.T, linewidth=1, color='grey')  # plot(confidence, metric)
    y = py.mean(0)
    ax.plot(px, y, linewidth=3, color='blue', label=f'all classes {y.max():.2f} at {px[y.argmax()]:.3f}')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(Path(save_dir), dpi=250)
