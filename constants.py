import os
from pathlib import Path


DATA_BASE_DIR = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "./data")
DATA_BASE_DIR = Path(DATA_BASE_DIR)
# DATA_BASE_DIR = Path('/home/sutongkun/Pretrain_VLP_Project/MGCA/Datasets')
# #############################################


ULTRA_DATA_DIR = DATA_BASE_DIR / 'ultrasound'
ULTRA_MASTER_CSV = ULTRA_DATA_DIR / 'master.csv'
ULTRA_COARSE = ULTRA_DATA_DIR / 'coarse.json'
ULTRA_MIDDLE = ULTRA_DATA_DIR / 'middle.json'
ULTRA_FINE = ULTRA_DATA_DIR / 'fine.json'
ULTRA_IMG_DIR = ULTRA_DATA_DIR / 'files'
ULTRA_CAPTION_CSV = ULTRA_DATA_DIR / 'caption_master.csv'
ULTRA_PATH_COL = 'Path'
ULTRA_SPLIT_COL = 'split'

BUSI_DATA_DIR = DATA_BASE_DIR / 'Dataset_BUSI_with_GT'
BUSI_IMG_DIR = BUSI_DATA_DIR / 'files'
BUSI_CLASSIFICATION_TRAIN_CSV = BUSI_DATA_DIR / 'classification/train.csv'
BUSI_CLASSIFICATION_VALID_CSV = BUSI_DATA_DIR / 'classification/val.csv'
BUSI_CLASSIFICATION_TEST_CSV = BUSI_DATA_DIR / 'classification/test.csv'
BUSI_DETECTION_TRAIN_PKL = BUSI_DATA_DIR / 'detection/train.pkl'
BUSI_DETECTION_TEST_PKL = BUSI_DATA_DIR / 'detection/test.pkl'
BUSI_DETECTION_VALID_PKL = BUSI_DATA_DIR / 'detection/val.pkl'
BUSI_SEG_TRAIN_PKL = BUSI_DATA_DIR / 'detection/train.pkl'
BUSI_SEG_TEST_PKL = BUSI_DATA_DIR / 'detection/test.pkl'
BUSI_SEG_VALID_PKL = BUSI_DATA_DIR / 'detection/val.pkl'

AUIDT_DATA_DIR = DATA_BASE_DIR / 'dataset thyroid'
AUIDT_IMG_DIR = AUIDT_DATA_DIR / 'files'
AUIDT_TRAIN_CSV = AUIDT_DATA_DIR / 'train.csv'
AUIDT_VALID_CSV = AUIDT_DATA_DIR / 'val.csv'
AUIDT_TEST_CSV = AUIDT_DATA_DIR / 'test.csv'

DDTI_DATA_DIR = DATA_BASE_DIR / 'DDTI'
DDTI_IMG_DIR = DDTI_DATA_DIR / 'files'
DDTI_DETECTION_TRAIN_PKL = DDTI_DATA_DIR / 'detection/train.pkl'
DDTI_DETECTION_TEST_PKL = DDTI_DATA_DIR / 'detection/test.pkl'
DDTI_DETECTION_VALID_PKL = DDTI_DATA_DIR / 'detection/val.pkl'
DDTI_SEG_TRAIN_PKL = DDTI_DATA_DIR / 'detection/train.pkl'
DDTI_SEG_TEST_PKL = DDTI_DATA_DIR / 'detection/test.pkl'
DDTI_SEG_VALID_PKL = DDTI_DATA_DIR / 'detection/val.pkl'



