import pandas as pd
import json

def coarse_preprocess():
    master_path = '../data/Ultrasound_data/master.csv'
    json_path = '../data/Ultrasound_data/coarse_grained_data2.json'
    json_data = []
    data = pd.read_csv(master_path)
    for _, row in data.iterrows():
        if '乳腺' in row['findings']:
            json_data.append({'img': row['Path'].replace('Ultrasonic_datasets/files', 'data/Ultrasound_data/files2').replace('jpg', 'jpeg'), 'prompt': '生成中文乳腺超声报告：', 'label': row['findings']})
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False)

def middle_preprocess():
    data_path = '../data/Ultrasound_data/ultrasound_middle_grained.csv'
    json_path = '../data/Ultrasound_data/middle_grained_data2.json'
    json_data = []
    data = pd.read_csv(data_path)
    for _, row in data.iterrows():
        if '乳腺' in row['findings']:
            for i in range(1, 4):
                label_idx = 'label' + str(i)
                prompt_idx = 'prompt' + str(i)
                a = row[prompt_idx]
                b = row[label_idx]
                if row[label_idx] != ' ':
                    json_data.append({'img': row['Path'].replace('Ultrasonic_datasets/files',
                                                         'data/Ultrasound_data/files2').replace('jpg', 'jpeg'),
                              'prompt': row[prompt_idx] + '：', 'label': row[label_idx]})
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False)

def data_mix():
    json1 = '../data/Ultrasound_data/coarse_grained_data2.json'
    json2 = '../data/Ultrasound_data/middle_grained_data2.json'
    new_json = '../data/Ultrasound_data/coarse_middle_11.json'

    with open(json1,'r', encoding='utf-8') as f:
        data1 = json.load(f)
    with open(json2,'r', encoding='utf-8') as f:
        data2 = json.load(f)

    data = data1 + data2[:3500]

    with open(new_json, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)

if __name__ == '__main__':
    data_mix()

