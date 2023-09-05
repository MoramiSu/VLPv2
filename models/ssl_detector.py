from typing import List
import torch
import torch.nn as nn
from collections import OrderedDict
from MGCA.utils.yolo_loss import YOLOLoss
from MGCA.datasets.data_module import DataModule
from MGCA.datasets.transforms import DataTransforms
from MGCA.datasets.detection_dataset import RSNADetectionDataset, BUSIDetectionDataset
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from MGCA.utils.detection_utils import non_max_suppression
from pytorch_lightning import LightningModule


class SSLDetector(LightningModule):
    def __init__(self,
                 img_encoder: nn.Module,
                 learning_rate: float = 5e-4,
                 weight_decay: float = 1e-6,
                 imsize: int = 224,
                 conf_thres: float = 0.5,
                 iou_thres: List = [0.4, 0.45, 0.5,
                                    0.55, 0.6, 0.65, 0.7, 0.75],
                 nms_thres: float = 0.5,
                 *args,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters(ignore=['img_encoder'])
        self.model = ModelMain(img_encoder)  # yolov3架构
        self.yolo_losses = []
        # 对于每个尺度，初始化YOLO损失
        for i in range(3):
            self.yolo_losses.append(YOLOLoss(self.model.anchors[i], self.model.classes,
                                             (imsize, imsize)))
        self.val_map = MeanAveragePrecision(
            iou_thresholds=self.hparams.iou_thres) # mAP@0.5
        self.test_map = MeanAveragePrecision(
            iou_thresholds=self.hparams.iou_thres)

    def shared_step(self, batch, batch_idx, split):
        outputs = self.model(batch["imgs"])  # 三种feature map的bbox预测结果
        losses_name = ["total_loss", "x", "y", "w", "h", "conf", "cls"]
        losses = []
        for _ in range(len(losses_name)):
            losses.append([])
        for i in range(3):  # 不同粒度的预测损失
            _loss_item = self.yolo_losses[i](outputs[i], batch["labels"])  # 得到的损失分别对应losses_name里的各种损失
            for j, l in enumerate(_loss_item):
                losses[j].append(l)
        losses = [sum(l) for l in losses]
        loss = losses[0]  # 所有粒度的total loss之和

        self.log(f"{split}_loss", loss, on_epoch=True,
                 prog_bar=True, sync_dist=True)

        if split != "train":
            output_list = []
            for i in range(3):  # 模型计算出的bbox
                output_list.append(self.yolo_losses[i](outputs[i]))
            output = torch.cat(output_list, 1)  # 模型计算出的所有bbox
            output = non_max_suppression(output, self.model.classes,  # 非极大值抑制
                                         conf_thres=self.hparams.conf_thres,
                                         nms_thres=self.hparams.nms_thres)  # 得到非最大值抑制后各个batch的所有bbox

            targets = batch["labels"].clone()
            # cxcywh -> xyxy
            h, w = batch["imgs"].shape[2:]  # 图像高宽
            targets[:, :, 1] = (batch["labels"][..., 1] -
                                batch["labels"][..., 3] / 2) * w  # bbox在原图中左上角横坐标
            targets[:, :, 2] = (batch["labels"][..., 2] -
                                batch["labels"][..., 4] / 2) * h  # bbox在原图中左上角纵坐标
            targets[:, :, 3] = (batch["labels"][..., 1] +
                                batch["labels"][..., 3] / 2) * w  # bbox在原图中右下角横坐标
            targets[:, :, 4] = (batch["labels"][..., 2] +
                                batch["labels"][..., 4] / 2) * h  # bbox在原图中右下角纵坐标

            sample_preds, sample_targets = [], []
            for i in range(targets.shape[0]):  # 遍历每个batch
                target = targets[i]
                out = output[i]
                if out is None:  # 没有预测出bbox
                    continue
                filtered_target = target[target[:, 3] > 0]  # 非空gt
                if filtered_target.shape[0] > 0:  # 存在非空gt
                    sample_target = dict(
                        boxes=filtered_target[:, 1:],  # bbox位置大小
                        labels=filtered_target[:, 0]  # bbox类别
                    )
                    sample_targets.append(sample_target)

                    out = output[i]
                    sample_pred = dict(
                        boxes=out[:, :4],  # bbox位置大小
                        scores=out[:, 4],  # 置信度
                        labels=out[:, 6]  # 类别与置信度
                    )

                    sample_preds.append(sample_pred)

            if split == "val":  # 更新mAP
                self.val_map.update(sample_preds, sample_targets)
            elif split == "test":
                self.test_map.update(sample_preds, sample_targets)

        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, "test")

    def validation_epoch_end(self, validation_step_outputs):
        torch.use_deterministic_algorithms(False)
        map = self.val_map.compute()["map"]  # 计算mAP
        self.log("val_mAP", map, prog_bar=True,
                 on_epoch=True, sync_dist=True)
        self.val_map.reset()

    def test_epoch_end(self, test_step_outputs):
        torch.use_deterministic_algorithms(False)
        map = self.test_map.compute()["map"]
        self.log("test_mAP", map, prog_bar=True,
                 on_epoch=True, sync_dist=True)
        self.test_map.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.hparams.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=self.hparams.weight_decay
        )

        return optimizer

    @staticmethod
    def num_training_steps(trainer, dm) -> int:
        """Total training steps inferred from datamodule and devices."""
        dataset = dm.train_dataloader()
        dataset_size = len(dataset)
        num_devices = max(1, trainer.num_gpus, trainer.num_processes)
        if trainer.tpu_cores:
            num_devices = max(num_devices, trainer.tpu_cores)
        effective_batch_size = trainer.accumulate_grad_batches * num_devices

        return (dataset_size // effective_batch_size) * trainer.max_epochs


class ModelMain(nn.Module):
    def __init__(self, backbone, is_training=True):
        super(ModelMain, self).__init__()
        self.training = is_training
        self.backbone = backbone
        self.anchors = torch.tensor([  # yolov2的先验框
            [[116, 90], [156, 198], [373, 326]],  # 大尺度先验框，用于感受野较大的feature map，即resnet最后一个block的输出，用于预测大物体
            [[30, 61], [62, 45], [59, 119]],  # 中尺度先验框，用于感受野中等的feature map，即resnet倒数第二个block的输出，用于预测中等物体
            [[10, 13], [16, 30], [33, 23]]  #   小尺度先验框，用于感受野较小的feature map，即resnet倒数第三个block的输出，用于预测小物体
        ]) * 224 / 416  # 使框适应224×224的图片
        self.classes = 1  # 检测类别数。检测目标只有一种

        _out_filters = self.backbone.filters  # 三种feature map的通道数
        #  embedding0  # # 生成bbox预测结果和与更大feature map拼接信息的变换
        final_out_filter0 = len(self.anchors[0]) * (5 + self.classes)  # 模型输出通道数，每个感受野的框个数×（5+类别数），其中5表示bbox的中心偏移量、长宽系数和置信度
        self.embedding0 = self._make_embedding(
            [512, 1024], _out_filters[-1], final_out_filter0)  # 生成bbox预测结果和与更大feature map拼接信息的变换
        #  embedding1
        final_out_filter1 = len(self.anchors[1]) * (5 + self.classes)
        self.embedding1_cbl = self._make_cbl(512, 256, 1)  # 通道数减半
        self.embedding1_upsample = nn.Upsample(
            scale_factor=2, mode='nearest')
        self.embedding1 = self._make_embedding(
            [256, 512], _out_filters[-2] + 256, final_out_filter1)  # _out_filters[-2] + 256：当前feature map的通道数+下一个更小feature map变换后的通道数
        #  embedding2
        final_out_filter2 = len(self.anchors[2]) * (5 + self.classes)
        self.embedding2_cbl = self._make_cbl(256, 128, 1)
        self.embedding2_upsample = nn.Upsample(
            scale_factor=2, mode='nearest')
        self.embedding2 = self._make_embedding(
            [128, 256], _out_filters[-3] + 128, final_out_filter2)

    def _make_cbl(self, _in, _out, ks):
        ''' cbl = conv + batch_norm + leaky_relu
        '''
        pad = (ks - 1) // 2 if ks else 0  # 卷积四周的padding数，保持卷积前后图像大小不变
        return nn.Sequential(OrderedDict([
            ("conv", nn.Conv2d(_in, _out, kernel_size=ks,
             stride=1, padding=pad, bias=False)),
            ("bn", nn.BatchNorm2d(_out)),
            ("relu", nn.LeakyReLU(0.1)),  # ReLU的变种，自变量小于0时梯度不等于0
        ]))

    def _make_embedding(self, filters_list, in_filters, out_filter):
        m = nn.ModuleList([
            self._make_cbl(in_filters, filters_list[0], 1),  # 1×1卷积+BN+AF，通道数减半
            self._make_cbl(filters_list[0], filters_list[1], 3),  # 3×3卷积+BN+AF，通道数翻倍
            self._make_cbl(filters_list[1], filters_list[0], 1),
            self._make_cbl(filters_list[0], filters_list[1], 3),
            self._make_cbl(filters_list[1], filters_list[0], 1),
            self._make_cbl(filters_list[0], filters_list[1], 3)])
        m.add_module("conv_out", nn.Conv2d(filters_list[1], out_filter, kernel_size=1,
                                           stride=1, padding=0, bias=True))  # 将通道数降至输出通道数
        return m

    def forward(self, x):
        def _branch(_embedding, _in):
            for i, e in enumerate(_embedding):
                _in = e(_in)
                if i == 4:  # 保留第五个embedding层，此时通道数为512，后面两层用于生成bbox预测
                    out_branch = _in
            return _in, out_branch  # 该fature map的bbox预测结果，第五个embedding层的输出（通道数512）
        #  backbone
        x2, x1, x0 = self.backbone(x)  # 从细到粗三种feature map的

        # x2: bz, 512, 28, 28
        # x1: bz, 1024, 14, 14
        # x0: bz, 2048, 7, 7
        #  yolo branch 0  # 感受野最大的feature map的bbox预测结果
        out0, out0_branch = _branch(self.embedding0, x0)  # 感受野最大的feature map的bbox预测结果，第五个embedding层的输出（通道数512）
        #  yolo branch 1  # 感受野中等的feature map的bbox预测结果
        x1_in = self.embedding1_cbl(out0_branch)  # 通道数减半
        x1_in = self.embedding1_upsample(x1_in)  # 上采样，feature map尺寸翻倍
        x1_in = torch.cat([x1_in, x1], 1)  # 通道维度拼接
        out1, out1_branch = _branch(self.embedding1, x1_in)
        #  yolo branch 2  感受野最小的feature map的bbox预测结果
        x2_in = self.embedding2_cbl(out1_branch)
        x2_in = self.embedding2_upsample(x2_in)
        x2_in = torch.cat([x2_in, x2], 1)
        out2, out2_branch = _branch(self.embedding2, x2_in)

        # out0: bz, 18, 7, 7
        # out1: bz, 18, 14, 14
        # out2: bz, 18, 28, 28
        return out0, out1, out2  # 三种feature map的bbox预测结果


if __name__ == "__main__":
    model = ModelMain()

    datamodule = DataModule(BUSIDetectionDataset, None, DataTransforms,
                            0.1, 32, 1, 224)

    for batch in datamodule.train_dataloader():
        break
