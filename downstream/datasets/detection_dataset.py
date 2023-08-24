import pickle
import random

import cv2
import numpy as np
from constants import *
from datasets.classification_dataset import BaseImageDataset
from datasets.transforms import *
from datasets.utils import read_from_dicom, resize_img
from PIL import Image
import torch
import pandas as pd

random.seed(42)
np.random.seed(42)


class RSNADetectionDataset(BaseImageDataset):
    def __init__(self, split="train", transform=None, data_pct=1., imsize=224, max_objects=10):
        super().__init__(split, transform)
        if not os.path.exists(RSNA_DATA_DIR):
            raise RuntimeError(f"{RSNA_DATA_DIR} does not exist!")

        if self.split == "train":
            with open(RSNA_DETECTION_TRAIN_PKL, "rb") as f:
                filenames, bboxs = pickle.load(f)
        elif self.split == "valid":
            with open(RSNA_DETECTION_VALID_PKL, "rb") as f:
                filenames, bboxs = pickle.load(f)
        elif self.split == "test":
            with open(RSNA_DETECTION_TEST_PKL, "rb") as f:
                filenames, bboxs = pickle.load(f)
        else:
            raise ValueError(f"split {split} does not exist!")

        self.imsize = imsize

        self.filenames_list, self.bboxs_list = [], []
        for i in range(len(filenames)):
            bbox = bboxs[i]
            new_bbox = bbox.copy()
            new_bbox[:, 0] = (bbox[:, 0] + bbox[:, 2]) / 2.  # 水平中心
            new_bbox[:, 1] = (bbox[:, 1] + bbox[:, 3]) / 2.  # 竖直中心
            new_bbox[:, 2] = bbox[:, 2] - bbox[:, 0]  # bbox总宽
            new_bbox[:, 3] = bbox[:, 3] - bbox[:, 1]  # bbox总高
            n = new_bbox.shape[0]  # bbox数
            new_bbox = np.hstack([np.zeros((n, 1)), new_bbox])  # 类别标签。只有一个类，故均为0
            pad = np.zeros((max_objects - n, 5))  # padding至最大bbox数以进行并行传输
            new_bbox = np.vstack([new_bbox, pad])
            self.filenames_list.append(filenames[i])
            self.bboxs_list.append(new_bbox)

        self.filenames_list = np.array(self.filenames_list)
        self.bboxs_list = np.array(self.bboxs_list)
        n = len(self.filenames_list)  # 数据量
        if split == "train":
            indices = np.random.choice(n, int(data_pct * n), replace=False)  # 随机采样
            self.filenames_list = self.filenames_list[indices]
            self.bboxs_list = self.bboxs_list[indices]

    def __len__(self):
        return len(self.filenames_list)  # 随机采样后的样本数

    def __getitem__(self, index):
        filename = self.filenames_list[index]
        img_path = RSNA_IMG_DIR / filename
        x = read_from_dicom(
            img_path, None, None)

        x = cv2.cvtColor(np.asarray(x), cv2.COLOR_BGR2RGB)  # 改变通道顺序
        h, w, _ = x.shape
        x = cv2.resize(x, (self.imsize, self.imsize),
                       interpolation=cv2.INTER_LINEAR)
        x = Image.fromarray(x, "RGB")

        if self.transform:
            x = self.transform(x)

        # bbox归一化
        y = self.bboxs_list[index]
        y[:, 1] /= w
        y[:, 3] /= w
        y[:, 2] /= h
        y[:, 4] /= h

        sample = {
            "imgs": x,
            "labels": y
        }

        return sample


class OBJCXRDetectionDataset(BaseImageDataset):
    def __init__(self, split="train", transform=None, data_pct=1., imsize=224, max_objects=20):
        # TODO: resize in detection is different from that in classification.
        super().__init__(split, transform)
        if not os.path.exists(OBJ_DATA_DIR):
            raise RuntimeError(f"{OBJ_DATA_DIR} does not exist!")

        if self.split == "train":
            with open(OBJ_TRAIN_PKL, "rb") as f:
                filenames, bboxs = pickle.load(f)
        elif self.split == "valid":
            with open(OBJ_VALID_PKL, "rb") as f:
                filenames, bboxs = pickle.load(f)
        elif self.split == "test":
            with open(OBJ_TEST_PKL, "rb") as f:
                filenames, bboxs = pickle.load(f)
        else:
            raise ValueError(f"split {split} does not exist!")

        self.filenames_list = []
        self.bboxs_list = []
        for i in range(len(bboxs)):
            filename = filenames[i]
            bbox = bboxs[i]
            bbox = np.array(bbox)

            padding = np.zeros((max_objects - len(bbox), 4))
            bbox = np.vstack((bbox, padding))
            bbox = np.hstack((np.zeros((max_objects, 1)), bbox))

            new_bbox = bbox.copy()
            # xminyminwh -> xywh
            new_bbox[:, 1] = bbox[:, 1] + bbox[:, 3] / 2
            new_bbox[:, 2] = bbox[:, 2] + bbox[:, 4] / 2

            self.filenames_list.append(filename)
            self.bboxs_list.append(new_bbox)

        n = len(self.filenames_list)
        self.filenames_list = np.array(self.filenames_list)
        self.bboxs_list = np.array(self.bboxs_list)
        if split == "train":
            indices = np.random.choice(n, int(data_pct * n), replace=False)
            self.filenames_list = self.filenames_list[indices]
            self.bboxs_list = self.bboxs_list[indices]

        self.imsize = imsize

    def __len__(self):
        return len(self.filenames_list)

    def __getitem__(self, index):
        filename = self.filenames_list[index]
        if self.split == "train":
            img_path = OBJ_TRAIN_IMG_PATH / filename
        elif self.split == "valid":
            img_path = OBJ_VALID_IMG_PATH / filename
        elif self.split == "test":
            img_path = OBJ_TEST_IMG_PATH / filename
        else:
            raise RuntimeError()

        x = cv2.imread(str(img_path))
        h, w, _ = x.shape
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        x = cv2.resize(x, (self.imsize, self.imsize),
                       interpolation=cv2.INTER_LINEAR)

        if self.transform:
            x = self.transform(x)

        y = self.bboxs_list[index]
        y[:, 1] /= w
        y[:, 3] /= w
        y[:, 2] /= h
        y[:, 4] /= h

        sample = {
            "imgs": x,
            "labels": y
        }

        return sample

class BUSIDetectionDataset(BaseImageDataset):
    def __init__(self, split="train", transform=None, data_pct=1., imsize=224, max_objects=10):
        super().__init__(split, transform)
        if not os.path.exists(RSNA_DATA_DIR):
            raise RuntimeError(f"{BUSI_DATA_DIR} does not exist!")

        if self.split == "train":
            with open(BUSI_DETECTION_TRAIN_PKL, "rb") as f:
                filenames, bboxs = pickle.load(f)
        elif self.split == "valid":
            with open(BUSI_DETECTION_VALID_PKL, "rb") as f:
                filenames, bboxs = pickle.load(f)
        elif self.split == "test":
            with open(BUSI_DETECTION_TEST_PKL, "rb") as f:
                filenames, bboxs = pickle.load(f)
        else:
            raise ValueError(f"split {split} does not exist!")

        self.imsize = imsize

        self.filenames_list, self.bboxs_list = [], []
        for i in range(len(filenames)):
            bbox = np.array(bboxs[i])
            new_bbox = bbox.copy()
            new_bbox[:, 0] = (bbox[:, 0] + bbox[:, 2]) / 2.  # 水平中心
            new_bbox[:, 1] = (bbox[:, 1] + bbox[:, 3]) / 2.  # 竖直中心
            new_bbox[:, 2] = bbox[:, 2] - bbox[:, 0]  # bbox总宽
            new_bbox[:, 3] = bbox[:, 3] - bbox[:, 1]  # bbox总高
            n = new_bbox.shape[0]  # bbox数
            new_bbox = np.hstack([np.zeros((n, 1)), new_bbox])  # 类别标签。只有一个类，故均为0
            pad = np.zeros((max_objects - n, 5))  # padding至最大bbox数以进行并行传输
            new_bbox = np.vstack([new_bbox, pad])
            self.filenames_list.append(filenames[i])
            self.bboxs_list.append(new_bbox)

        self.filenames_list = np.array(self.filenames_list)
        self.bboxs_list = np.array(self.bboxs_list)
        n = len(self.filenames_list)  # 数据量
        if split == "train":
            indices = np.random.choice(n, int(data_pct * n), replace=False)  # 随机采样
            self.filenames_list = self.filenames_list[indices]
            self.bboxs_list = self.bboxs_list[indices]

    def __len__(self):
        return len(self.filenames_list)  # 随机采样后的样本数

    def __getitem__(self, index):
        filename = self.filenames_list[index]
        img_path = BUSI_DATA_DIR / ('files'+'/'+filename)
        x = cv2.imread(str(img_path), 0)  # 0表示返回灰度图
        x = cv2.convertScaleAbs(x, alpha=(255.0 / x.max()))
        h, w = x.shape
        x = cv2.resize(x, (self.imsize, self.imsize),
                       interpolation=cv2.INTER_LINEAR)
        img = Image.fromarray(x).convert("RGB")  # 转化为RGB三通道PIL图像格式
        if self.transform is not None:  # 图像变换
            img = self.transform(img)

        # bbox归一化
        y = self.bboxs_list[index]
        y[:, 1] /= w
        y[:, 3] /= w
        y[:, 2] /= h
        y[:, 4] /= h

        # sample = {
        #     "imgs": img,
        #     "labels": y,
        #     'path': img_path
        # }

        return img, torch.tensor(y), filename

class DDTIDetectionDataset(BaseImageDataset):
    def __init__(self, split="train", transform=None, data_pct=1., imsize=224, max_objects=10):
        super().__init__(split, transform)
        if not os.path.exists(DDTI_DATA_DIR):
            raise RuntimeError(f"{DDTI_DATA_DIR} does not exist!")

        if self.split == "train":
            with open(DDTI_DETECTION_TRAIN_PKL, "rb") as f:
                filenames, bboxs = pickle.load(f)
        elif self.split == "valid":
            with open(DDTI_DETECTION_VALID_PKL, "rb") as f:
                filenames, bboxs = pickle.load(f)
        elif self.split == "test":
            with open(DDTI_DETECTION_TEST_PKL, "rb") as f:
                filenames, bboxs = pickle.load(f)
        else:
            raise ValueError(f"split {split} does not exist!")

        self.imsize = imsize

        self.filenames_list, self.bboxs_list = [], []
        for i in range(len(filenames)):
            bbox = np.array(bboxs[i])
            new_bbox = bbox.copy()
            new_bbox[:, 0] = (bbox[:, 0] + bbox[:, 2]) / 2.  # 水平中心
            new_bbox[:, 1] = (bbox[:, 1] + bbox[:, 3]) / 2.  # 竖直中心
            new_bbox[:, 2] = bbox[:, 2] - bbox[:, 0]  # bbox总宽
            new_bbox[:, 3] = bbox[:, 3] - bbox[:, 1]  # bbox总高
            n = new_bbox.shape[0]  # bbox数
            new_bbox = np.hstack([np.zeros((n, 1)), new_bbox])  # 类别标签。只有一个类，故均为0
            pad = np.zeros((max_objects - n, 5))  # padding至最大bbox数以进行并行传输
            new_bbox = np.vstack([new_bbox, pad])
            self.filenames_list.append(filenames[i])
            self.bboxs_list.append(new_bbox)

        self.filenames_list = np.array(self.filenames_list)
        self.bboxs_list = np.array(self.bboxs_list)
        n = len(self.filenames_list)  # 数据量
        if split == "train":
            indices = np.random.choice(n, int(data_pct * n), replace=False)  # 随机采样
            self.filenames_list = self.filenames_list[indices]
            self.bboxs_list = self.bboxs_list[indices]

    def __len__(self):
        return len(self.filenames_list)  # 随机采样后的样本数

    def __getitem__(self, index):
        filename = self.filenames_list[index]
        img_path = DDTI_DATA_DIR / ('files'+'/'+filename)
        x = cv2.imread(str(img_path), 0)  # 0表示返回灰度图
        x = cv2.convertScaleAbs(x, alpha=(255.0 / x.max()))
        h, w = x.shape
        x = cv2.resize(x, (self.imsize, self.imsize),
                       interpolation=cv2.INTER_LINEAR)
        img = Image.fromarray(x).convert("RGB")  # 转化为RGB三通道PIL图像格式
        if self.transform is not None:  # 图像变换
            img = self.transform(img)

        # bbox归一化
        y = self.bboxs_list[index]
        y[:, 1] /= w
        y[:, 3] /= w
        y[:, 2] /= h
        y[:, 4] /= h

        sample = {
            "imgs": img,
            "labels": y
        }

        return sample

def det_collate_fn(batch):
    """sort sequence"""
    imgs, bboxes, filenames = [], [], []
    # imgs, mask_imgs, cap_len, ids, tokens, attention, position = [], [], [], [], [], [], []
    for b in batch:
        img, bbox, filename = b
        # img, mask_img, cap, cap_l, p = b
        imgs.append(img)
        bboxes.append(bbox)
        filenames.append(filename)

    # stack
    imgs = torch.stack(imgs)
    bboxes = torch.stack(bboxes)

    return_dict = {
        "imgs": imgs,
        'labels': bboxes,
        'filenames': filenames
    }
    return return_dict

if __name__ == "__main__":
    dataset = DDTIDetectionDataset(split='valid')
    print(len(dataset))
    # print(dataset[5])
