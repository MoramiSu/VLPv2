from __future__ import division
import math
from random import sample
import time
import ipdb
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from terminaltables import AsciiTable


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,
                                          0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,
                                          0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * \
        torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


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
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) -
             torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    # iou = inter / (area1 + area2 - inter)
    return inter / (area1[:, None] + area2 - inter)


def non_max_suppression(prediction, num_classes, conf_thres=0.5, nms_thres=0.4):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """

    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    box_corner = prediction.new(prediction.shape)  # 随机初始化一个与prediction形状、设备、类型相同的张量
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2  # 左上角横坐标
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2  # 左上角纵坐标
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2  # 右下角横坐标
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2  # 右下角纵坐标
    prediction[:, :, :4] = box_corner[:, :, :4]  # 用上述四个坐标覆盖prediction的位置坐标

    output = [None for _ in range(len(prediction))]  # bs个None
    for image_i, image_pred in enumerate(prediction):  # 遍历每个batch
        # Filter out confidence scores below threshold
        conf_mask = (image_pred[:, 4] >= conf_thres).squeeze()  # 每个bbox的置信度是否大于等于阈值
        image_pred = image_pred[conf_mask]  # 去掉置信度小于阈值的bbox
        # If none are remaining => process next image
        if not image_pred.size(0):  # 所有bbox的置信度都小于阈值
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(  # 每个bbox的类别及其类别置信度（得分）
            image_pred[:, 5:5 + num_classes], 1,  keepdim=True)
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat(
            (image_pred[:, :5], class_conf.float(), class_pred.float()), 1)  # 去掉后面的各类得分，拼接上类别置信度和类别
        # Iterate through all predicted classes
        unique_labels = detections[:, -1].cpu().unique()  # 预测结果中出现的所有类别。unique：去除张量中的重复元素，每个元素返回一次
        if prediction.is_cuda:
            unique_labels = unique_labels.cuda()
        for c in unique_labels:  # 对每个类别的bbox使用非极大值抑制去重
            # Get the detections with the particular class
            detections_class = detections[detections[:, -1] == c]  # 该类别的所有bbox
            # Sort the detections by maximum objectness confidence
            _, conf_sort_index = torch.sort(
                detections_class[:, 4], descending=True)  # 对该类别的所有bbox根据置信度从高到低排序
            detections_class = detections_class[conf_sort_index]  # 调序
            # Perform non-maximum suppression
            max_detections = []
            while detections_class.size(0):
                # Get detection with highest confidence and save as max detection
                max_detections.append(detections_class[0].unsqueeze(0))
                # Stop if we're at the last detection
                if len(detections_class) == 1:  # 若只剩一个bbox
                    break
                # Get the IOUs for all boxes with lower confidence
                ious = bbox_iou(max_detections[-1], detections_class[1:]) # 计算所有bbox与已保留的最后一个bbox的iou
                # Remove detections with IoU >= NMS threshold
                detections_class = detections_class[1:][ious < nms_thres]  # 若某些bbox与已保留的最后一个bbox的iou大于一个阈值，说明这个bbox已经有相似的bbox被保留下来，故将其去除

            max_detections = torch.cat(max_detections).data  # 组装成一个张量
            # Add max detections to outputs
            output[image_i] = max_detections if output[image_i] is None else torch.cat(
                (output[image_i], max_detections))  # 保存非最大值抑制后的bbox

    return output


def get_batch_statistics(outputs, targets, iou_threshold, imsize=224):
    ''' Compute true positives, predicted scores and predicted labels per sample 
        :param outputs: List
        :param targets: bz, 10, 5
        :output batch_metrics: List
    '''

    batch_metrics = []
    for sample_i in range(len(outputs)):
        if outputs[sample_i] is None:
            continue
        output = outputs[sample_i]
        pred_boxes = output[:, :4]
        pred_scores = output[:, 4]
        pred_labels = output[:, -1]
        target_sample = targets[sample_i, targets[sample_i, :, 3] != 0]
        n = len(target_sample)
        if n > 0:
            target_boxes = torch.zeros(n, 4).type_as(target_sample)
            target_boxes[:, 0] = imsize * \
                (target_sample[:, 1] - target_sample[:, 3] / 2)
            target_boxes[:, 1] = imsize * \
                (target_sample[:, 2] - target_sample[:, 4] / 2)
            target_boxes[:, 2] = imsize * \
                (target_sample[:, 1] + target_sample[:, 3] / 2)
            target_boxes[:, 3] = imsize * \
                (target_sample[:, 2] + target_sample[:, 4] / 2)
            # target_boxes = target_sample[:, 1:]
            # TODO: if we don't consider labels
            ious = box_iou(pred_boxes, target_boxes)
            corrects = (ious > iou_threshold).sum(dim=1)
            # mask = pred_labels == target_sample[:, 0].unsqueeze(1)
            # ious = torch.masked_select(ious, mask)
            true_positives = (corrects > 0).float().cpu().numpy()

            batch_metrics.append(
                [true_positives, pred_scores.cpu().numpy(), pred_labels.cpu().numpy()])

    return batch_metrics


def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in unique_classes:
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects
        # ipdb.set_trace()
        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum()
            tpc = (tp[i]).cumsum()

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1])

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype("int32")


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def print_eval_stats(metrics_output, class_names, verbose):
    if metrics_output is not None:
        precision, recall, AP, f1, ap_class = metrics_output
        if verbose:
            # Prints class AP and mean AP
            ap_table = [["Index", "Class", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            print(AsciiTable(ap_table).table)
        print(f"---- mAP {AP.mean():.5f} ----")
    else:
        print("---- mAP not measured (no detections found by model) ----")


if __name__ == "__main__":
    batch_preds = []
    for i in range(16):
        pred = torch.rand(10, 7)
        pred[:, -1] = 0
        batch_preds.append(pred)

    targets = torch.stack(batch_preds)[..., :4]
    targets = torch.cat([torch.zeros(16, 10, 1), targets], dim=2)
    sample_metrics = get_batch_statistics(batch_preds, targets, 0.5)
    true_positives, pred_scores, pred_labels = [
        np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    # labels = [0] * 16 * 100
    # metrics_output = ap_per_class(
    #     true_positives, pred_scores, pred_labels, labels)

    # precision, recall, AP, f1, ap_class = metrics_output
    # print(AP.mean())
    ipdb.set_trace()
    # print_eval_stats(metrics_output, ["Pneumonia"], True)