import torch
import numpy as np
import cv2

# IOU计算
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
    iou=Iou(full_person[:,:4],pred_person[:,:4])
    max_overlap,indices=iou.max(dim=1)
    add_indices=max_overlap<0.2
    pred_person=torch.cat([pred_person,full_person[add_indices]],dim=0)
    return pred_person

def merge_person_phone(person,phone):
    print("merge",person.shape,phone.shape)
    phone=phone[phone[:,4]>0.5]
    print(phone.shape)
    if phone.shape[0]<1:
        return person, phone
    iou=miniou(person[:,:4],phone[:,:4])
    max_over,phone_indices=iou.max(dim=1)
    indices=max_over>0.45
    phone_indices=phone_indices[indices]
    return person[indices],phone[phone_indices]

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
    # return torch.LongTensor(keep)   # Pytorch的索引值为LongTensor


##########去除重复样本的时候需要将去除样本的最大属性分数继承过来
def miniou_nms(bboxes, scores, score_attr, threshold=0.8, image=None, type="max"):
    x1 = bboxes[:,0]
    y1 = bboxes[:,1]
    x2 = bboxes[:,2]
    y2 = bboxes[:,3]
    areas = (x2-x1)*(y2-y1)   # [N,] 每个bbox的面积
    _, order = scores.sort(0, descending=True)    # 根据分数降序排列，保存原来的索引

    win_name="test"
    keep = []
    wi=0
    num_box=0
    while order.numel() > 0:       # torch.numel()返回张量元素个数
        if order.numel() == 1:     # 保留框只剩一个
            i = order.item()
            keep.append(i)
            box=bboxes[i]
            image_per=image.copy()
            cv2.rectangle(image_per, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color=(0, 255, 255),
                          thickness=3)
            cv2.imwrite(f"workspace/test/{100}.jpg",image_per)

            break
        else:
            i = order[0].item()    # 保留scores最大的那个框box[i]
            keep.append(i)

        image_per=image.copy()
        # 计算box[i]与其余各框的IOU(思路很好)
        xx1 = x1[order[1:]].clamp(min=x1[i])   # [N-1,]
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])
        inter = (xx2-xx1).clamp(min=0) * (yy2-yy1).clamp(min=0)   # [N-1,]

        miniou = inter / torch.min(areas[i],areas[order[1:]])  # [N-1,]
        # print(wi,miniou)
        # print(torch.min(areas[i],areas[order[1:]]))
        # print(inter)
        ###########去除分块检测造成的高iou，因此使用较高的阈值
        idx = (miniou <= threshold).nonzero().squeeze() # 注意此时idx为[N-1,] 而order为[N,]
        ###########获取超过阈值的目标框与当前目标框中最大的目标框，并将其作为最后的框
        throw_idx= (miniou > threshold).nonzero().squeeze()

        #################测试查找输出为nan的数据
        # for i,iou_s in enumerate(miniou):
        #     print(iou_s)
        #
        #     if torch.isnan(iou_s):
        #
        #         image_per_s=image.copy()
        #         box=bboxes[order[i+1]]
        #         print(bboxes[order[0]])
        #         print(box)
        #         print(areas[order[0]])
        #         print(areas[order[i+1]])
        #         cv2.rectangle(image_per_s, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
        #                       color=(0, 255, 255),
        #                       thickness=3)
        #         cv2.putText(image_per_s, str(iou_s), (int(box[0]) - 4, int(box[1]) - 4), fontFace=cv2.FONT_ITALIC,
        #                     fontScale=0.5, color=(0, 255, 0))
        #         cv2.imwrite(f"workspace/test/{wi}_iou_nan.jpg", image_per_s)
        #         import sys
        #         sys.exit()
        #################测试查找输出为nan的数据

        # print(wi,throw_idx,iou_thr)
        if isinstance(throw_idx.tolist(), list):
            throw_idx=throw_idx.tolist()
        else:
            throw_idx=[throw_idx.tolist()]
        #
        # print(throw_idx)
        if len(throw_idx)>0:
            all_area=[areas[i],*[areas[order[id+1]] for id in throw_idx]]
            if type=="max":
                index_max=all_area.index(max(all_area))
            else:
                index_max = all_area.index(min(all_area))
            # print("wi",wi)
            # print(throw_idx)
            # print(all_area)
            # print(iou_thr)
            # print(index_max)
            if index_max>0:
                save_index=order[(throw_idx[index_max-1])+1].item()
                # print("save",save_index)
                keep.pop()
                keep.append(save_index)
            #####################测试
            # box_now=bboxes[i]
            # cv2.rectangle(image_per,(int(box_now[0]),int(box_now[1])),(int(box_now[2]),int(box_now[3])),color=(0,0,255),thickness=3)
            # for i,index in enumerate(throw_idx):
            #     index_ori=order[index+1]
            #     box=bboxes[index_ori]
            #     cv2.rectangle(image_per,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),color=(0,255,0),thickness=3)
            #     cv2.putText(image_per, str(i), (int(box[0]) - 4, int(box[1]) - 4), fontFace=cv2.FONT_ITALIC,fontScale=0.5, color=(0, 255, 0))
            # cv2.imwrite(f"workspace/test/{wi}.jpg", image_per)
            # # cv2.destroyWindow(win_name)
            del image_per
            num_box+=(len(throw_idx)+1)
        else:
           # image_per=image.copy()
           # box_now = bboxes[i]
           # cv2.rectangle(image_per, (int(box_now[0]), int(box_now[1])), (int(box_now[2]), int(box_now[3])),
           #               color=(0, 255, 255), thickness=3)
           # cv2.imwrite(f"workspace/test/{wi}.jpg", image_per)
           num_box+=1
            #########测试
        if idx.numel() == 0:
            break
        order = order[idx+1]  # 修补索引之间的差值
        wi+=1
    # print("left",order.numel())
    # print(num_box)
    # print(bboxes.shape[0],len(keep))

    return torch.cat([bboxes[keep],scores[keep][:,None],score_attr[keep]],dim=1)
    # return torch.LongTensor(keep)   # Pytorch的索引值为LongTensor


