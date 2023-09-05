from .text_generation import TextGenerationPipeline
import cv2
import os
from datasets.transforms import DataTransforms
from models.mgca.mgca_module import MGCA

MODEL_PATH = '/home/sutongkun/VLPv2/data/ckpts/'
IMG_PATH = '/home/sutongkun/VLPv2/data/ultrasound/files/229145_k.jpeg'
PROMPT = '生成中文超声报告：'

def read_img(path, transform):
    img0_path = path.replace('_k', '_1')
    img1_path = path.replace('_k', '_2')
    img0 = cv2.imread(img0_path)
    img0 = cv2.resize(img0, (256, 256))
    img0 = transform(img0)
    if os.path.isfile(img1_path):
        img1 = cv2.imread(img1_path)
        img1 = cv2.resize(img1, (256, 256))
        img1 = transform(img1)
    else:
        img1 = img0
    return img0, img1

def main():
    model = MGCA.load_from_checkpoint(MODEL_PATH, strict=True)
    transform = DataTransforms(is_train=False)
    imgs = [read_img(IMG_PATH, transform)]



