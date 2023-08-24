import torch
import torch.nn as nn
import numpy as np
import math


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
                                          0], box1[:, 1], box1[:, 2], box1[:, 3]  # 左上角横坐标，左上角纵坐标，右下角横坐标，右下角纵坐标
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,
                                          0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)  # 将b1_x1元素复制补充到anchor数，然后与anchor逐个元素对比，输出最大值
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area  # 相交面积
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * \
        torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)  # clamp表示将得到的长宽用0截断
    # Union Area  # 总面积
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def non_max_suppression(prediction, num_classes, conf_thres=0.5, nms_thres=0.4):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """

    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        conf_mask = (image_pred[:, 4] >= conf_thres).squeeze()
        image_pred = image_pred[conf_mask]
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(
            image_pred[:, 5:5 + num_classes], 1,  keepdim=True)
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat(
            (image_pred[:, :5], class_conf.float(), class_pred.float()), 1)
        # Iterate through all predicted classes
        unique_labels = detections[:, -1].cpu().unique()
        if prediction.is_cuda:
            unique_labels = unique_labels.cuda()
        for c in unique_labels:
            # Get the detections with the particular class
            detections_class = detections[detections[:, -1] == c]
            # Sort the detections by maximum objectness confidence
            _, conf_sort_index = torch.sort(
                detections_class[:, 4], descending=True)
            detections_class = detections_class[conf_sort_index]
            # Perform non-maximum suppression
            max_detections = []
            while detections_class.size(0):
                # Get detection with highest confidence and save as max detection
                max_detections.append(detections_class[0].unsqueeze(0))
                # Stop if we're at the last detection
                if len(detections_class) == 1:
                    break
                # Get the IOUs for all boxes with lower confidence
                ious = bbox_iou(max_detections[-1], detections_class[1:])
                # Remove detections with IoU >= NMS threshold
                detections_class = detections_class[1:][ious < nms_thres]

            max_detections = torch.cat(max_detections).data
            # Add max detections to outputs
            output[image_i] = max_detections if output[image_i] is None else torch.cat(
                (output[image_i], max_detections))

    return output


class YOLOLoss(nn.Module):
    def __init__(self, anchors, num_classes, img_size):
        super(YOLOLoss, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes  # 4个位置参数，一个置信度，一个类别
        self.img_size = img_size

        self.ignore_threshold = 0.5  # IOU阈值
        # 各损失函数权重
        self.lambda_xy = 2.5
        self.lambda_wh = 2.5
        self.lambda_conf = 1.0
        self.lambda_cls = 1.0

        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()

    def forward(self, input, targets=None):
        bs = input.size(0)
        in_h = input.size(2)  # feature map大小
        in_w = input.size(3)
        stride_h = self.img_size[1] / in_h  # 图像下采样倍数
        stride_w = self.img_size[0] / in_w
        scaled_anchors = [(a_w / stride_w, a_h / stride_h)  # 对anchor也进行下采样
                          for a_w, a_h in self.anchors]

        prediction = input.view(bs,  self.num_anchors,
                                self.bbox_attrs, in_h, in_w).permute(0, 1, 3, 4, 2).contiguous()  # [bs, anchors数, feature map高, feature map宽, 预测结果]

        # Get outputs
        x = torch.sigmoid(prediction[..., 0])          # Center x  # [..., 0]=[:, :, :, :, 0]
        y = torch.sigmoid(prediction[..., 1])          # Center y
        w = prediction[..., 2]                         # Width
        h = prediction[..., 3]                         # Height
        conf = torch.sigmoid(prediction[..., 4])       # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.

        if targets is not None:
            #  build target  # 计算得到label和mask。
            mask, noobj_mask, tx, ty, tw, th, tconf, tcls = self.get_target(targets, scaled_anchors,
                                                                            in_w, in_h,
                                                                            self.ignore_threshold)  # mask：只有负责预测的anchor为1；noobj_mask：iou低于阈值的负样本为1；tx、ty、tw、th、tconf、tcls：负责预测的anchor的label
            # mask, noobj_mask = mask.cuda(), noobj_mask.cuda()
            # tx, ty, tw, th = tx.cuda(), ty.cuda(), tw.cuda(), th.cuda()
            # tconf, tcls = tconf.cuda(), tcls.cuda()
            mask, noobj_mask = mask.type_as(targets), noobj_mask.type_as(targets)
            tx, ty, tw, th = tx.type_as(targets), ty.type_as(targets), tw.type_as(targets), th.type_as(targets)
            tconf, tcls = tconf.type_as(targets), tcls.type_as(targets)
            loss_x = self.bce_loss(x * mask, tx * mask)  # 为什么用bce？
            loss_y = self.bce_loss(y * mask, ty * mask)
            loss_w = self.mse_loss(w * mask, tw * mask)
            loss_h = self.mse_loss(h * mask, th * mask)
            loss_conf = self.bce_loss(conf * mask, mask) + \
                0.5 * self.bce_loss(conf * noobj_mask, noobj_mask * 0.0)
            # if no bboxes in this batch
            loss_cls = self.bce_loss((pred_cls * mask.unsqueeze(-1)).float(), (tcls * mask.unsqueeze(-1)).float())
            #  total loss = losses * weight
            loss = loss_x * self.lambda_xy + loss_y * self.lambda_xy + \
                loss_w * self.lambda_wh + loss_h * self.lambda_wh + \
                loss_conf * self.lambda_conf + loss_cls * self.lambda_cls

            return loss, loss_x.item(), loss_y.item(), loss_w.item(),\
                loss_h.item(), loss_conf.item(), loss_cls.item()
        else:
            FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
            LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
            # Calculate offsets for each grid
            grid_x = torch.linspace(0, in_w-1, in_w).repeat(in_w, 1).repeat(
                bs * self.num_anchors, 1, 1).view(x.shape).type(FloatTensor)  # 一个bs*num_anchor*in_w*in_w的张量，每一行为从0~in_w-1的数字
            grid_y = torch.linspace(0, in_h-1, in_h).repeat(in_h, 1).t().repeat(  # 同上，唯一不同是进行了转置，每一列为从0~in_h-1的数字
                bs * self.num_anchors, 1, 1).view(y.shape).type(FloatTensor)
            # Calculate anchor w, h
            anchor_w = FloatTensor(
                scaled_anchors).index_select(1, LongTensor([0]))  # 该feature map三个anchor的宽度
            anchor_h = FloatTensor(
                scaled_anchors).index_select(1, LongTensor([1]))  # 该feature map三个anchor的高度
            anchor_w = anchor_w.repeat(bs, 1).repeat(
                1, 1, in_h * in_w).view(w.shape)  # 将其进行复制，得到bs*num_anchor*in_h*in_w的张量
            anchor_h = anchor_h.repeat(bs, 1).repeat(
                1, 1, in_h * in_w).view(h.shape)
            # Add offset and scale with anchors
            # predictions: bz, 3, 7, 7, 6  # bbox预测结果
            pred_boxes = FloatTensor(prediction[..., :4].shape)  # bs*num_anchor*in_h*in_w的张量，其初始值是无意义的，会被覆盖掉，用于存放每个grid cell三个bbox的预测结果
            pred_boxes[..., 0] = x.data + grid_x
            pred_boxes[..., 1] = y.data + grid_y
            pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
            pred_boxes[..., 3] = torch.exp(h.data) * anchor_h
            # pred_boxes: bz, 3, 7, 7, 4
            # Results
            _scale = torch.Tensor([stride_w, stride_h] * 2).type(FloatTensor)  # [stride_w, stride_h, stride_w, stride_h]
            output = torch.cat((pred_boxes.view(bs, -1, 4) * _scale,
                                conf.view(bs, -1, 1), pred_cls.view(bs, -1, self.num_classes)), -1) # 恢复下采样
            return output.data  # data：相当于detach

    def get_target(self, target, anchors, in_w, in_h, ignore_threshold):
        bs = target.size(0)

        mask = torch.zeros(bs, self.num_anchors, in_h, in_w,
                           requires_grad=False)  # anchor mask矩阵。只有与gt iou最大的anchor为1
        noobj_mask = torch.ones(
            bs, self.num_anchors, in_h, in_w, requires_grad=False)  # 负样本mask矩阵，所有正样本均为0
        # gt，负责预测之的anchor处为gt的tx，ty，th，tw，tconf，tcls，即该anchor的label
        tx = torch.zeros(bs, self.num_anchors, in_h, in_w,
                         requires_grad=False)
        ty = torch.zeros(bs, self.num_anchors, in_h, in_w,
                         requires_grad=False)
        tw = torch.zeros(bs, self.num_anchors, in_h, in_w,
                         requires_grad=False)
        th = torch.zeros(bs, self.num_anchors, in_h, in_w,
                         requires_grad=False)
        tconf = torch.zeros(bs, self.num_anchors, in_h, in_w,
                            requires_grad=False)
        tcls = torch.zeros(bs, self.num_anchors, in_h, in_w,
                           self.num_classes, requires_grad=False)

        for b in range(bs):
            for t in range(target.shape[1]):  # 遍历每个bbox
                if target[b, t].sum() == 0:  # 空bbox
                    continue
                # Convert to position relative to box  # 得到bbox在当前feature map中的大小和位置
                gx = target[b, t, 1].item() * in_w
                gy = target[b, t, 2].item() * in_h
                gw = target[b, t, 3].item() * in_w
                gh = target[b, t, 4].item() * in_h
                # Get grid box indices  # 得到bbox位于哪个grid cell中
                gi = int(gx)
                gj = int(gy)
                # Get shape of gt box
                gt_box = torch.FloatTensor(
                    np.array([0, 0, gw, gh])).unsqueeze(0)  # 转化为左上角位于grid cell左上角的bbox，四个值分别左上角横坐标，左上角纵坐标，右下角横坐标，右下角纵坐标
                # Get shape of anchor box
                anchor_shapes = torch.FloatTensor(np.concatenate((np.zeros((self.num_anchors, 2)),
                                                                  np.array(anchors)), 1))  # 同上，将所有anchor box都转化为左上角位于grad cell左上角的bbox
                # Calculate iou between gt and anchor shapes
                anch_ious = bbox_iou(gt_box, anchor_shapes)  # 忽略中心位置，只考虑bbox大小的iou值
                # Where the overlap is larger than threshold set mask to zero (ignore)
                noobj_mask[b, anch_ious > ignore_threshold, gj, gi] = 0  # 将iou大于阈值的anchor box置零，剩余的anchor box均为负样本
                # Find the best matching anchor box
                best_n = np.argmax(anch_ious)  # iou最大的anchor

                # Masks
                mask[b, best_n, gj, gi] = 1
                # Coordinates  # label
                tx[b, best_n, gj, gi] = gx - gi
                ty[b, best_n, gj, gi] = gy - gj
                # Width and height
                tw[b, best_n, gj, gi] = math.log(gw/anchors[best_n][0] + 1e-16)
                th[b, best_n, gj, gi] = math.log(gh/anchors[best_n][1] + 1e-16)
                # object
                tconf[b, best_n, gj, gi] = 1
                # One-hot encoding of label
                tcls[b, best_n, gj, gi, int(target[b, t, 0])] = 1

        return mask, noobj_mask, tx, ty, tw, th, tconf, tcls


if __name__ == "__main__":
    pass