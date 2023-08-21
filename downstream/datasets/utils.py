import cv2
import numpy as np
import pydicom
import torch
from PIL import Image
from constants import *
import torchvision.transforms as transforms
import random
import matplotlib.pyplot as plt
from downstream.datasets.transforms import DataTransforms
import time

PATCH_LIST = [((i//14)*16, (i//14)*16+16, (i%14)*16, (i%14)*16+16) for i in range(196)]


def read_from_dicom(img_path, imsize=None, transform=None):
    dcm = pydicom.read_file(img_path)
    x = dcm.pixel_array

    x = cv2.convertScaleAbs(x, alpha=(255.0 / x.max()))  # 自动亮度均衡

    # 若图片反色，则将其恢复正常
    if dcm.PhotometricInterpretation == "MONOCHROME1":
        x = cv2.bitwise_not(x)

    # transform images
    if imsize is not None:
        x = resize_img(x, imsize)

    img = Image.fromarray(x).convert("RGB")

    if transform is not None:
        img = transform(img)

    return img


def resize_img(img, scale):
    """
    Args:
        img - image as numpy array (cv2)
        scale - desired output image-size as scale x scale
    Return:
        image resized to scale x scale with shortest dimension 0-padded
    """
    size = img.shape
    max_dim = max(size)
    max_ind = size.index(max_dim)  # 图像最大边是高还是宽

    # Resizing
    if max_ind == 0:  # 高长
        # image is heigher
        wpercent = scale / float(size[0])  # 缩放比例
        hsize = int((float(size[1]) * float(wpercent)))  # 缩放后的宽度
        desireable_size = (scale, hsize)  # 缩放后图像大小
    else:  # 宽长
        # image is wider
        hpercent = scale / float(size[1])
        wsize = int((float(size[0]) * float(hpercent)))
        desireable_size = (wsize, scale)
    resized_img = cv2.resize(
        img, desireable_size[::-1], interpolation=cv2.INTER_AREA
    )  # 缩放  # this flips the desireable_size vector

    # Padding
    if max_ind == 0:  # 高长，计算宽度方向填充量
        # height fixed at scale, pad the width
        pad_size = scale - resized_img.shape[1]
        left = int(np.floor(pad_size / 2))
        right = int(np.ceil(pad_size / 2))
        top = int(0)
        bottom = int(0)
    else:  # 宽长，计算高度方向填充量
        # width fixed at scale, pad the height
        pad_size = scale - resized_img.shape[0]
        top = int(np.floor(pad_size / 2))
        bottom = int(np.ceil(pad_size / 2))
        left = int(0)
        right = int(0)
    resized_img = np.pad(
        resized_img, [(top, bottom), (left, right)], "constant", constant_values=0
    )  # 填充

    return resized_img


def get_imgs(img_path, scale, transform=None, multiscale=False):
    img_path = str(img_path).replace('_k', '_1')
    x = cv2.imread(str(img_path), 0)  # 0表示返回灰度图
    # tranform images
    x = resize_img(x, scale)  # 图像伸缩
    img = Image.fromarray(x).convert("RGB")  # 转化为RGB三通道PIL图像格式
    if transform is not None:  # 图像变换
        img = transform(img)

    return img

def get_imgs_sample(img_path, scale, transform=None, multiscale=False):
    i = 1
    path_list = []
    while True:
        if os.path.isfile(str(img_path).replace('_k', '_'+str(i))):
            path_list.append(str(img_path).replace('_k', '_'+str(i)))
            i += 1
        else:
            break
    img_path = random.sample(path_list, 1)[0]
    x = cv2.imread(img_path, 0)  # 0表示返回灰度图
    # tranform images
    x = resize_img(x, scale)  # 图像伸缩
    img = Image.fromarray(x).convert("RGB")  # 转化为RGB三通道PIL图像格式
    if transform is not None:  # 图像变换
        img = transform(img)

    return img

def get_imgs_pretrain(img_path, split, scale=256, transform=None):
    img = pretrain_image_process(img_path, split)

    img = cv2.resize(img, (scale, scale))
    img = Image.fromarray(img)
    if transform is not None:
        img = transform(img)

    return img

def pretrain_image_process(img_path, split):
    img_list = []
    if split == 'train':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomResizedCrop(size=600, scale=(0.8, 1)),
            transforms.ColorJitter(brightness=(0.5, 1.5), contrast=(0.5, 1.5), saturation=(0.5, 1.5))
            # transforms.Resize(600)
        ])
        for i in range(6):
            path = str(img_path).replace('_k', '_'+str(i))
            if os.path.isfile(path):
                img = cv2.imread(path)
                img = transform(img)*255
            else:
                img = torch.zeros((3, 600, 600))
            img_list.append(img)

        a = random.randint(450, 550)
        b = random.randint(450, 550)
        c = random.randint(950, 1050)
        x = torch.zeros(3, 1000, 1500)
        x[:, :a, :b] = img_list[0][:, :a, :b]
        x[:, a:, :b] = img_list[1][:, a-400:, :b]
        x[:, :a, b:c] = img_list[2][:, :a, int(300-(c-b)/2):int(300+(c-b)/2)]
        x[:, a:, b:c] = img_list[3][:, a-400:, int(300 - (c - b) / 2):int(300 + (c - b) / 2)]
        x[:, :a, c:] = img_list[4][:, :a, c-900:]
        x[:, a:, c:] = img_list[5][:, a-400:, c-900:]

    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop(450),
            transforms.Resize(500)
        ])
        for i in range(6):
            path = str(img_path).replace('_k', '_' + str(i))
            if os.path.isfile(path):
                img = cv2.imread(path)
                img = transform(img) * 255
            else:
                img = torch.zeros((3, 500, 500))
            img_list.append(img)

        x = torch.zeros(3, 1000, 1500)
        x[:, :500, :500] = img_list[0]
        x[:, 500:, :500] = img_list[1]
        x[:, :500, 500:1000] = img_list[2]
        x[:, 500:, 500:1000] = img_list[3]
        x[:, :500, 1000:] = img_list[4]
        x[:, 500:, 1000:] = img_list[5]

    return x.permute(1, 2, 0).numpy().astype(np.uint8)

def get_mask_img(img):
    global PATCH_LIST

    mask_img = img.clone()

    mask_patch_list = random.sample([i for i in range(196)], 29)

    for i in mask_patch_list:
        j = random.choice([k for k in range(196)])
        mask_img[:, PATCH_LIST[i][0]:PATCH_LIST[i][1], PATCH_LIST[i][2]:PATCH_LIST[i][3]] = img[:, PATCH_LIST[j][0]:PATCH_LIST[j][1], PATCH_LIST[j][2]:PATCH_LIST[j][3]]
        mask_img[:, PATCH_LIST[j][0]:PATCH_LIST[j][1], PATCH_LIST[j][2]:PATCH_LIST[j][3]] = img[:, PATCH_LIST[i][0]:PATCH_LIST[i][1], PATCH_LIST[i][2]:PATCH_LIST[i][3]]

    '''
    mask_patch_list = random.sample([i for i in range(196)], 59)

    for i in mask_patch_list:
        mask_img[:, PATCH_LIST[i][0]:PATCH_LIST[i][1], PATCH_LIST[i][2]:PATCH_LIST[i][3]] = -torch.ones([3, 16, 16])
    '''
    return mask_img

if __name__ == '__main__':
    img_path = ULTRA_DATA_DIR / 'files/229901_k.jpg'
    transform = DataTransforms(is_train=True)
    img = get_imgs_sample(img_path, scale=256, transform=transform)
    plt.imshow(transforms.ToPILImage()(img))
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    # plt.savefig('./img_train.png', dpi=1000, bbox_inches='tight', pad_inches=0)
    plt.show()


