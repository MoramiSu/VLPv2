import pickle
import os

import cv2
import numpy as np
import pandas as pd
import pydicom
import torch
from albumentations import Compose, Normalize, Resize, ShiftScaleRotate
from albumentations.pytorch import ToTensorV2
from constants import *
from downstream.datasets.classification_dataset import BaseImageDataset
from downstream.datasets.utils import resize_img, get_imgs
from PIL import Image

np.random.seed(42)


class SIIMImageDataset(BaseImageDataset):
    def __init__(self, split="train", transform=None,
                 data_pct=0.01, phase="segmentation", imsize=224):
        super().__init__(split, transform)

        self.phase = phase
        self.imsize = imsize
        if self.phase == "segmentation":
            self.seg_transform = self.get_transforms()
        else:
            raise NotImplementedError(f"{self.phase} not implemented")

        # read in csv file
        if split == "train":
            self.df = pd.read_csv(PNEUMOTHORAX_TRAIN_CSV)
        elif split == "valid":
            self.df = pd.read_csv(PNEUMOTHORAX_VALID_CSV)
        else:
            self.df = pd.read_csv(PNEUMOTHORAX_TEST_CSV)

        self.df["Path"] = self.df["Path"].apply(
            lambda x: os.path.join(PNEUMOTHORAX_IMG_DIR, x))

        # only keep positive samples for segmentation
        self.df["class"] = self.df[" EncodedPixels"].apply(lambda x: x != "-1")
        if self.phase == "segmentation" and split == "train":
            self.df_neg = self.df[self.df["class"] == False]
            self.df_pos = self.df[self.df["class"] == True]
            n_pos = self.df_pos["ImageId"].nunique()
            neg_series = self.df_neg["ImageId"].unique()
            neg_series_selected = np.random.choice(
                neg_series, size=n_pos, replace=False
            )
            self.df_neg = self.df_neg[self.df_neg["ImageId"].isin(
                neg_series_selected)]
            self.df = pd.concat([self.df_pos, self.df_neg])

        # sample data
        if data_pct != 1 and split == "train":
            ids = self.df["ImageId"].unique()
            n_samples = int(len(ids) * data_pct)
            series_selected = np.random.choice(
                ids, size=n_samples, replace=False)
            self.df = self.df[self.df["ImageId"].isin(series_selected)]

        self.imgids = self.df.ImageId.unique().tolist()

    def __getitem__(self, index):
        imgid = self.imgids[index]
        imgid_df = self.df.groupby("ImageId").get_group(imgid)

        # get image
        img_path = imgid_df.iloc[0]["Path"]
        x = self.read_from_dicom(img_path)

        # get labels
        if self.phase == "segmentation":
            rle_list = imgid_df[" EncodedPixels"].tolist()
            mask = np.zeros([1024, 1024])
            if rle_list[0] != "-1":
                for rle in rle_list:
                    mask += self.rle2mask(
                        rle, PNEUMOTHORAX_IMG_SIZE, PNEUMOTHORAX_IMG_SIZE
                    )
            mask = (mask >= 1).astype("float32")
            mask = resize_img(mask, self.imsize)

            augmented = self.seg_transform(image=x, mask=mask)
            x = augmented["image"]
            y = augmented["mask"].squeeze()
        else:
            y = imgid_df.iloc[0]["Label"]
            y = torch.tensor([y])

        return x, y

    def read_from_dicom(self, img_path):

        dcm = pydicom.read_file(img_path)
        x = dcm.pixel_array
        x = cv2.convertScaleAbs(x, alpha=(255.0 / x.max()))

        if dcm.PhotometricInterpretation == "MONOCHROME1":
            x = cv2.bitwise_not(x)

        img = Image.fromarray(x).convert("RGB")
        return np.asarray(img)

    def __len__(self):
        return len(self.imgids)

    def read_from_dicom(self, img_path):

        dcm = pydicom.read_file(img_path)
        x = dcm.pixel_array
        x = cv2.convertScaleAbs(x, alpha=(255.0 / x.max()))

        if dcm.PhotometricInterpretation == "MONOCHROME1":
            x = cv2.bitwise_not(x)

        img = Image.fromarray(x).convert("RGB")
        return np.asarray(img)

    def rle2mask(self, rle, width, height):
        """Run length encoding to segmentation mask"""

        mask = np.zeros(width * height)
        array = np.asarray([int(x) for x in rle.split()])
        starts = array[0::2]
        lengths = array[1::2]
        current_position = 0
        for index, start in enumerate(starts):
            current_position += start
            mask[current_position: current_position + lengths[index]] = 1
            current_position += lengths[index]

        return mask.reshape(width, height).T

    def get_transforms(self, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        list_transforms = []
        if self.split == "train":
            list_transforms.extend(
                [
                    ShiftScaleRotate(
                        shift_limit=0,  # no resizing
                        scale_limit=0.1,
                        rotate_limit=10,  # rotate
                        p=0.5,
                        border_mode=cv2.BORDER_CONSTANT,
                    )
                ]
            )
        list_transforms.extend(
            [
                Resize(self.imsize, self.imsize),
                Normalize(mean=mean, std=std, p=1),
                ToTensorV2(),
            ]
        )

        list_trfms = Compose(list_transforms)
        return list_trfms


class RSNASegmentDataset(BaseImageDataset):
    def __init__(self, split="train", transform=None, data_pct=1., imsize=224) -> None:
        super().__init__(split, transform)

        if not os.path.exists(RSNA_DATA_DIR):
            raise RuntimeError(f"{RSNA_DATA_DIR} does not exist!")

        if self.split == "train":
            with open(RSNA_DETECTION_TRAIN_PKL, "rb") as f:
                self.filenames, self.bboxs = pickle.load(f)
        elif self.split == "valid":
            with open(RSNA_DETECTION_VALID_PKL, "rb") as f:
                self.filenames, self.bboxs = pickle.load(f)
        elif self.split == "test":
            with open(RSNA_DETECTION_TEST_PKL, "rb") as f:
                self.filenames, self.bboxs = pickle.load(f)
        else:
            raise ValueError(f"split {split} does not exist!")

        # self.df["Path"] = self.df["patientId"].apply(
        #     lambda x: RSNA_IMG_DIR / (x + ".dcm"))

        n = len(self.filenames)
        if split == "train":
            indices = np.random.choice(n, int(data_pct * n), replace=False)
            self.filenames = self.filenames[indices]
            self.bboxs = self.bboxs[indices]

        self.imsize = imsize
        self.seg_transform = self.get_transforms()  # 图像变换

    def __len__(self):
        return len(self.filenames)

    def read_from_dicom(self, img_path):

        dcm = pydicom.read_file(img_path)
        x = dcm.pixel_array  # 转化为array
        x = cv2.convertScaleAbs(x, alpha=(255.0 / x.max()))  # 自动亮度均衡

        # 若存在反色图片，则恢复正常
        if dcm.PhotometricInterpretation == "MONOCHROME1":  # 负片
            x = cv2.bitwise_not(x)  # 负片变换

        img = Image.fromarray(x).convert("RGB")  # 转化为三通道
        return np.asarray(img)

    def __getitem__(self, index):
        filename = self.filenames[index]
        img_path = RSNA_IMG_DIR / filename
        x = self.read_from_dicom(img_path)

        mask = np.zeros([1024, 1024])  # 与图片大小相同

        bbox = self.bboxs[index]
        new_bbox = bbox[bbox[:, 3] > 0].astype(np.int64)  # 非空bbox
        if len(new_bbox) > 0:  # 存在bbox
            for i in range(len(new_bbox)):  # 遍历bbox
                try:
                    mask[new_bbox[i, 1]:new_bbox[i, 3],
                         new_bbox[i, 0]:new_bbox[i, 2]] += 1
                except:
                    import ipdb
                    ipdb.set_trace()
        mask = (mask >= 1).astype("float32")
        mask = resize_img(mask, self.imsize)  # mask为什么要resize两次
        augmented = self.seg_transform(image=x, mask=mask)

        x = augmented["image"]
        y = augmented["mask"].squeeze()

        return x, y

    def get_transforms(self, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        list_transforms = []
        if self.split == "train":
            list_transforms.extend(  # 类似于append，但是可以一次性增加多个值
                [
                    ShiftScaleRotate(  # 随机平移、缩放、旋转
                        shift_limit=0,  # no resizing  # 不平移
                        scale_limit=0.1,
                        rotate_limit=10,  # rotate
                        p=0.5,
                        border_mode=cv2.BORDER_CONSTANT,
                    )
                ]
            )
        list_transforms.extend(
            [
                Resize(self.imsize, self.imsize),
                Normalize(mean=mean, std=std, p=1),
                ToTensorV2(),  # 将HWC的array转化为CHW的tensor
            ]
        )

        list_trfms = Compose(list_transforms)
        return list_trfms

class BUSISegmentDataset(BaseImageDataset):
    def __init__(self, split="train", transform=None, data_pct=1., imsize=224) -> None:
        super().__init__(split, transform)

        if not os.path.exists(BUSI_DATA):
            raise RuntimeError(f"{BUSI_DATA} does not exist!")

        if self.split == "train":
            with open(BUSI_SEG_TRAIN, "rb") as f:
                self.filenames, self.bboxs = pickle.load(f)
        elif self.split == "valid":
            with open(BUSI_SEG_VAL, "rb") as f:
                self.filenames, self.bboxs = pickle.load(f)
        elif self.split == "test":
            with open(BUSI_SEG_TEST, "rb") as f:
                self.filenames, self.bboxs = pickle.load(f)
        else:
            raise ValueError(f"split {split} does not exist!")

        # self.df["Path"] = self.df["patientId"].apply(
        #     lambda x: RSNA_IMG_DIR / (x + ".dcm"))

        n = len(self.filenames)
        if split == "train":
            indices = np.random.choice(n, int(data_pct * n), replace=False)
            self.filenames = self.filenames[indices]
            self.bboxs = self.bboxs[indices]

        self.imsize = imsize
        self.seg_transform = self.get_transforms()  # 图像变换

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index]
        box = self.bboxs[index]
        img_path = BUSI_IMG + '/' + filename
        x = cv2.imread(str(img_path), 0)
        x = cv2.convertScaleAbs(x, alpha=(255.0 / x.max()))
        x = np.asarray(Image.fromarray(x).convert('RGB'))
        mask_path = BUSI_IMG + '/' + filename.replace('.png', '_mask.png')
        mask = cv2.imread(str(mask_path), 0)
        if len(box) > 1:
            for i in range(1, len(box)):
                mask_path = BUSI_IMG + '/' + filename.replace('.png', '_mask_'+str(i)+'.png')
                mask += cv2.imread(str(mask_path), 0)
        mask = (mask >= 1).astype("float32")
        mask = resize_img(mask, self.imsize)
        augmented = self.seg_transform(image=x, mask=mask)

        x = augmented["image"]
        y = augmented["mask"]

        return x, y, filename

    def get_transforms(self, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        list_transforms = []
        if self.split == "train":
            list_transforms.extend(  # extend类似于append，但是可以一次性增加多个值
                [
                    ShiftScaleRotate(  # 随机平移、缩放、旋转
                        shift_limit=0,  # no resizing  # 不平移
                        scale_limit=0.1,
                        rotate_limit=10,  # rotate
                        p=0.5,
                        border_mode=cv2.BORDER_CONSTANT,
                    )
                ]
            )
        list_transforms.extend(
            [
                Resize(self.imsize, self.imsize),
                Normalize(mean=mean, std=std, p=1),
                ToTensorV2(),  # 将HWC的array转化为CHW的tensor
            ]
        )

        list_trfms = Compose(list_transforms)
        return list_trfms

class DDTISegmentDataset(BaseImageDataset):
    def __init__(self, split="train", transform=None, data_pct=1., imsize=224) -> None:
        super().__init__(split, transform)

        if not os.path.exists(DDTI_DATA_DIR):
            raise RuntimeError(f"{DDTI_DATA_DIR} does not exist!")

        if self.split == "train":
            with open(DDTI_SEG_TRAIN_PKL, "rb") as f:
                self.filenames, self.bboxs = pickle.load(f)
        elif self.split == "valid":
            with open(DDTI_SEG_VALID_PKL, "rb") as f:
                self.filenames, self.bboxs = pickle.load(f)
        elif self.split == "test":
            with open(DDTI_SEG_TEST_PKL, "rb") as f:
                self.filenames, self.bboxs = pickle.load(f)
        else:
            raise ValueError(f"split {split} does not exist!")

        # self.df["Path"] = self.df["patientId"].apply(
        #     lambda x: RSNA_IMG_DIR / (x + ".dcm"))

        n = len(self.filenames)
        if split == "train":
            indices = np.random.choice(n, int(data_pct * n), replace=False)
            self.filenames = self.filenames[indices]
            self.bboxs = self.bboxs[indices]

        self.imsize = imsize
        self.seg_transform = self.get_transforms()  # 图像变换

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index]
        box = self.bboxs[index]
        img_path = DDTI_IMG_DIR / filename
        x = cv2.imread(str(img_path), 0)
        x = cv2.convertScaleAbs(x, alpha=(255.0 / x.max()))
        x = np.asarray(Image.fromarray(x).convert('RGB'))
        mask_path = DDTI_IMG_DIR / filename.replace('.PNG', '_mask.PNG')
        mask = cv2.imread(str(mask_path), 0)
        # if len(box) > 1:
            # for i in range(1, len(box)):
                # mask_path = DDTI_IMG_DIR / filename.replace('.png', '_mask_'+str(i)+'.png')
                # mask += cv2.imread(str(mask_path), 0)
        mask = (mask >= 1).astype("float32")
        mask = resize_img(mask, self.imsize)
        augmented = self.seg_transform(image=x, mask=mask)

        x = augmented["image"]
        y = augmented["mask"]

        return x, y, filename

    def get_transforms(self, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        list_transforms = []
        if self.split == "train":
            list_transforms.extend(  # extend类似于append，但是可以一次性增加多个值
                [
                    ShiftScaleRotate(  # 随机平移、缩放、旋转
                        shift_limit=0,  # no resizing  # 不平移
                        scale_limit=0.1,
                        rotate_limit=10,  # rotate
                        p=0.5,
                        border_mode=cv2.BORDER_CONSTANT,
                    )
                ]
            )
        list_transforms.extend(
            [
                Resize(self.imsize, self.imsize),
                Normalize(mean=mean, std=std, p=1),
                ToTensorV2(),  # 将HWC的array转化为CHW的tensor
            ]
        )

        list_trfms = Compose(list_transforms)
        return list_trfms

def seg_collate_fn(batch):
    """sort sequence"""
    imgs, masks, filenames = [], [], []
    # imgs, mask_imgs, cap_len, ids, tokens, attention, position = [], [], [], [], [], [], []
    for b in batch:
        img, mask, filename = b
        # img, mask_img, cap, cap_l, p = b
        imgs.append(img)
        masks.append(mask)
        filenames.append(filename)

    # stack
    imgs = torch.stack(imgs)
    masks = torch.stack(masks)



    return_dict = {
        "imgs": imgs,
        'masks': masks,
        'filenames': filenames
    }
    return return_dict

if __name__ == "__main__":
    dataset = DDTISegmentDataset()
    for data in dataset:
        pass
