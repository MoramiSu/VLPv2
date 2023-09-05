import os
import pickle
import re
import json

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from nltk.tokenize import RegexpTokenizer
from MGCA.constants import *
from MGCA.datasets.utils import get_imgs
from tqdm import tqdm
from transformers import BertTokenizer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class UltrasonicCaptioningDataset(data.Dataset):
    # def __init__(self, split="train", transform=None, data_pct=1.0,
                 # imsize=256, max_words=112, sent_num=3):
    def __init__(self, split="train", transform=None, data_pct=1.0,
                 imsize=256, max_words=200, sent_num=3):
        super().__init__()
        # if not os.path.exists(MIMIC_CXR_DATA_DIR):
            # raise RuntimeError(f"{MIMIC_CXR_DATA_DIR} does not exist!")
        if not os.path.exists(ULTRA_DATA_DIR):
            raise RuntimeError(f"{ULTRA_DATA_DIR} does not exist!")

        self.transform = transform
        self.imsize = imsize
        # self.df = pd.read_csv(MIMIC_CXR_MASTER_CSV)
        self.df = pd.read_csv(ULTRA_CAPTION_CSV)
        # self.df = self.df[self.df["ViewPosition"].isin(["PA", "AP"])]  # 去掉侧身，只保留PA和AP
        # self.df[MIMIC_CXR_PATH_COL] = self.df[MIMIC_CXR_PATH_COL].apply(
            # lambda x: os.path.join(MIMIC_CXR_DATA_DIR, "/".join(x.split("/")[1:])))  # 转化为绝对路径
        self.df[ULTRA_PATH_COL] = self.df[ULTRA_PATH_COL].apply(
            lambda x: os.path.join(ULTRA_DATA_DIR, "/".join(x.split("/")[1:])))

        # self.df = self.df[self.df[MIMIC_CXR_SPLIT_COL] == split]  # 保留特定split的数据
        if data_pct != 1.0 and split == "train":
            self.df = self.df.sample(frac=data_pct, random_state=42)
        self.df.reset_index(drop=True, inplace=True)  # 重置行索引

        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-chinese")

        # load studies and study to text mapping
        self.filenames, self.path2sent = self.load_text_data(split)  # 加载文本，得到数据路径、路径2文本字典

        # self.tokenizer = BertTokenizer.from_pretrained(
            # "emilyalsentzer/Bio_ClinicalBERT")  # 预训练模型分词器

        self.max_words = max_words  # 最大词数

    def load_text_data(self, split):
        # get study to captions mapping
        # TODO: check this
        # filepath = os.path.join(
            # BASE_DIR, "../data/captions.pickle")
        filepath = os.path.join(
            BASE_DIR, "../data/captions for captioning.json")
        if not os.path.isfile(filepath):
            print(
                f"Caption file {filepath} does not exit. Creating captions...")
            path2sent = self.create_path_2_sent_mapping()  # 生成路径：文本dict
            # with open(filepath, "wb") as f:  # 保存
                # pickle.dump(path2sent, f, protocol=2)
                # print("Save to: ", filepath)
            with open(filepath, "w", encoding='utf-8') as f:  # 保存
                json.dump(path2sent, f)
                print("Save to: ", filepath)
        else:
            # with open(filepath, "rb") as f:
                # path2sent = pickle.load(f)
            with open(filepath, "r", encoding='utf-8') as f:
                path2sent = json.load(f)

        # filter studies to use for current split
        filenames = []  # 数据路径
        for row in self.df.itertuples():  # 按行遍历，返回一个元组，每一列为一个元素，第一个元素是index
            # cur_split = getattr(row, MIMIC_CXR_SPLIT_COL)  # 数据划分
            # path = getattr(row, MIMIC_CXR_PATH_COL)  # 路径
            cur_split = getattr(row, ULTRA_SPLIT_COL)  # 数据划分
            path = getattr(row, ULTRA_PATH_COL)  # 路径
            if cur_split == split and path in path2sent:  # 若数据划分符合要求且path2sent中有相应路径
                filenames.append(path)

        return filenames, path2sent

    def create_path_2_sent_mapping(self):
        sent_lens, num_sents = [], []  # 各id词元数和各id句数
        path2sent = {}  # 路径：文本（单句列表）
        # iterrows is not faster than itertuples ...  but it is ok
        for _, row in tqdm(self.df.iterrows(), total=self.df.shape[0]):  # 逐行遍历，返回(index,row data)
            # pick impression, findings, last_paragraph
            captions = ""
            # captions += row["impression"]
            # captions += " "
            captions += row["findings"]

            # use space instead of newline
            captions = captions.replace("\n", " ")  # 去掉换行符

            # split sentences
            # splitter = re.compile("[0-9]+\.")  # 以数字+.作为pattern
            # captions = splitter.split(captions)  # 用pattern分句
            # captions = [point.split(".") for point in captions]  # 用.分句
            # captions = [sent for point in captions for sent in point]  # 无用，去掉外层的list，得到一个由句子构成的list，每个句子是一个元素
            spliter = re.compile(r'[,.，。]+')
            captions = spliter.split(captions)

            cnt = 0  # 统计该id的词元数
            study_sent = []
            # create tokens from captions
            for cap in captions:  # 遍历每一个句子
                if len(cap) == 0:  # 舍弃字符数为0的句子
                    continue

                cap = cap.replace("\ufffd\ufffd", " ")  # 去除解码失败的字符，\ufffd是替换字符，即某些解码失败的字符
                # picks out sequences of alphanumeric characters as tokens
                # and drops everything else
                # tokenizer = RegexpTokenizer(r"\w+")  # 初始化分词器
                # tokens = tokenizer.tokenize(cap.lower())  # 分词
                tokens = self.tokenizer.tokenize(cap.lower())
                # TODO: < 3 has instances of ['no', 'pneumothorax'], ['clear', 'lung']
                if len(tokens) <= 1:  # 去除少于2个次元的句子
                    continue

                # filter tokens for current sentence
                # included_tokens = []
                # for t in tokens:
                    # t = t.encode("ascii", "ignore").decode("ascii")
                    # if len(t) > 0:
                        # included_tokens.append(t)

                # if len(included_tokens) > 0:
                    # study_sent.append("".join(included_tokens))  # 将词元还原成句子

                # cnt += len(included_tokens)

                if len(tokens) > 0:
                    study_sent.append("".join(tokens)+',')  # 将词元还原成句子

                cnt += len(tokens)

            if cnt >= 3:  # 若该id词元数大于等于3
                sent_lens.append(cnt)
                num_sents.append(len(study_sent))
                path2sent[row[ULTRA_PATH_COL]] = study_sent

        # get report word/setence statistics
        sent_lens = np.array(sent_lens)
        num_sents = np.array(num_sents)

        print(
            f"sent lens: {sent_lens.min()},{sent_lens.mean()},{sent_lens.max()} [{np.percentile(sent_lens, 5)}, {np.percentile(sent_lens, 95)}]"
        )  # np.percentile：sent_lens的5%分位数和95%分位数
        print(
            f"num sents: {num_sents.min()},{num_sents.mean()},{num_sents.max()} [{np.percentile(num_sents, 5)}, {np.percentile(num_sents, 95)}]"
        )

        return path2sent

    def __len__(self):
        return len(self.filenames)

    def get_caption(self, path):
        series_sents = self.path2sent[path]  # 单句序列

        if len(series_sents) == 0:  # 句子数为0
            raise Exception("no sentence for path")

        # separate different sentences
        series_sents = list(filter(lambda x: x != "", series_sents))  # 过滤掉空句子
        sent = " ".join(series_sents)  # 拼接成完整文本

        tokens = self.tokenizer(  # 分词，生成padding mask
            sent,
            return_tensors="pt",  # 返回pytorch tensor
            truncation=True,  # 将过长的句子截断到最大长度
            padding="max_length",  # 将过短的句子填充到最大长度
            max_length=self.max_words,
        )
        # 得到三个张量：input_ids——文本词元id；token_type_ids——文本句子位置张量，每个句子一个id，从0开始增加；attention_mask——padding mask
        x_len = len([t for t in tokens["input_ids"][0] if t != 0])  # 非padding词元长度

        tokens['output_ids'] = torch.cat((tokens['input_ids'][0][1:], torch.tensor([0])), dim=0).unsqueeze(0)
        tokens['input_ids'][0][x_len-1] = 0
        tokens['attention_mask'][0][x_len-1] = 0

        return tokens, x_len

    def __getitem__(self, index):
        key = self.filenames[index]  # 路径
        caps, cap_len = self.get_caption(key)
        imgs = get_imgs(key, self.imsize, self.transform, multiscale=False)
        return imgs, caps, cap_len, key  # 图像，词元序列，非padding词元长度，路径


def multimodal_collate_fn(batch):
    """sort sequence"""
    imgs, cap_len, in_ids, tokens, attention, out_ids = [], [], [], [], [], []
    path = []
    for b in batch:
        img, cap, cap_l, p = b
        imgs.append(img)
        cap_len.append(cap_l)
        in_ids.append(cap["input_ids"])
        tokens.append(cap["token_type_ids"])
        attention.append(cap["attention_mask"])
        out_ids.append(cap['output_ids'])

        path.append(p)

    # stack
    imgs = torch.stack(imgs)
    in_ids = torch.stack(in_ids).squeeze()
    tokens = torch.stack(tokens).squeeze()
    attention = torch.stack(attention).squeeze()
    out_ids = torch.stack(out_ids).squeeze()

    # sort and add to dictionary
    sorted_cap_lens, sorted_cap_indices = torch.sort(  # 根据非padding词元数降序排序，得到排序后的词元数和对应索引
        torch.tensor(cap_len), 0, True)

    path = np.array(path)

    return_dict = {
        "input_caption_ids": in_ids[sorted_cap_indices],
        "token_type_ids": tokens[sorted_cap_indices],
        "attention_mask": attention[sorted_cap_indices],
        "imgs": imgs[sorted_cap_indices],
        "cap_lens": sorted_cap_lens,
        "path": path[sorted_cap_indices],
        'output_caption_ids': out_ids[sorted_cap_indices]
    }
    return return_dict


if __name__ == "__main__":
    from MGCA.datasets.transforms import DataTransforms
    transform = DataTransforms(is_train=True)
    dataset = UltrasonicCaptioningDataset(split="train", transform=transform)
    print(len(dataset))
    data = dataset[0]
    print(data)
