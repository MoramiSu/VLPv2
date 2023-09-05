import json
import pandas as pd
from constants import *


def coarse_preprocess():
    data = pd.read_csv(ULTRA_MASTER_CSV)

    corase_data = {}
    for _, row in data.iterrows():
        if '乳腺' in row['findings'] or '甲状腺' in row['findings']:
            filepath = row['Path'].replace('Ultrasonic_datasets/files', str(ULTRA_IMG_DIR)).replace('jpg', 'jpeg')
            corase_data[filepath] = [['生成中文超声报告：', row['findings']]]

    with open(ULTRA_COARSE, 'w', encoding='utf-8') as f:
        json.dump(corase_data, f, ensure_ascii=False)


if __name__ == "__main__":
    coarse_preprocess()
