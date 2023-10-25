import sys
sys.path.append('..')
import pandas as pd
from constants import *
from preprocess.utils import extract_mimic_text
import numpy as np


extract_text = False

np.random.seed(42)


def main():
    if extract_text:
        extract_mimic_text()
    metadata_df = pd.read_csv(MIMIC_CXR_META_CSV)  # 一些id和照片方位
    metadata_df = metadata_df[["dicom_id", "subject_id",
                               "study_id", "ViewPosition"]].astype(str)  # 抽取出这几列并转成str
    metadata_df["study_id"] = metadata_df["study_id"].apply(lambda x: "s"+x)
    # Only keep frontal images
    metadata_df = metadata_df[metadata_df["ViewPosition"].isin(["PA", "AP"])]  # 过滤掉侧身照片，只保留PA和AP

    text_df = pd.read_csv(MIMIC_CXR_TEXT_CSV)  # 报告
    text_df.dropna(subset=["impression", "findings"], how="all", inplace=True)  # 将impressing和findings列均为nan的行删除
    text_df = text_df[["study", "impression", "findings"]]
    text_df.rename(columns={"study": "study_id"}, inplace=True)

    split_df = pd.read_csv(MIMIC_CXR_SPLIT_CSV)  # 数据集划分
    split_df = split_df.astype(str)
    split_df["study_id"] = split_df["study_id"].apply(lambda x: "s"+x)
    # TODO: merge validate and test into test.
    split_df["split"] = split_df["split"].apply(
        lambda x: "valid" if x == "validate" or x == "test" else x)  # 合并验证数据和测试数据

    chexpert_df = pd.read_csv(MIMIC_CXR_CHEXPERT_CSV)  # 疾病类别
    chexpert_df[["subject_id", "study_id"]] = chexpert_df[[
        "subject_id", "study_id"]].astype(str)
    chexpert_df["study_id"] = chexpert_df["study_id"].apply(lambda x: "s"+x)

    master_df = pd.merge(metadata_df, text_df, on="study_id", how="left")  # 数据合并。取出metadata的每一行，匹配text的对应键值（on），若数据不存在则填充nan
    master_df = pd.merge(master_df, split_df, on=["dicom_id", "subject_id", "study_id"], how="inner")  # inner：连接交集，不相交舍弃
    master_df.dropna(subset=["impression", "findings"], how="all", inplace=True)  # inplace：True在原文件中进行删除，False返回新文件
    
    n = len(master_df)
    master_data = master_df.values  # 去掉第一行的索引，用数字替代，即用数字代替key

    root_dir = str(MIMIC_CXR_DATA_DIR).split("/")[-1] + "/files"
    path_list = []
    for i in range(n):
        row = master_data[i]
        file_path = "%s/p%s/p%s/%s/%s.jpg" % (root_dir, str(
            row[1])[:2], str(row[1]), str(row[2]), str(row[0]))
        path_list.append(file_path)
        
    master_df.insert(loc=0, column="Path", value=path_list)  # 加入文件位置列

    # Create labeled data df
    labeled_data_df = pd.merge(master_df, chexpert_df, on=[
                               "subject_id", "study_id"], how="inner")
    labeled_data_df.drop(["dicom_id", "subject_id", "study_id",
                          "impression", "findings"], axis=1, inplace=True)  # 只保留图像位置、图像方位、训练/验证、疾病类别

    train_df = labeled_data_df.loc[labeled_data_df["split"] == "train"]
    train_df.to_csv(MIMIC_CXR_TRAIN_CSV, index=False)  # index：是否保留行索引
    valid_df = labeled_data_df.loc[labeled_data_df["split"] == "valid"]
    valid_df.to_csv(MIMIC_CXR_TEST_CSV, index=False)

    # master_df.drop(["dicom_id", "subject_id", "study_id"],
    #                axis=1, inplace=True)

    # Fill nan in text
    master_df[["impression"]] = master_df[["impression"]].fillna(" ")  # 将nan的impression置空
    master_df[["findings"]] = master_df[["findings"]].fillna(" ")
    master_df.to_csv(MIMIC_CXR_MASTER_CSV, index=False)


if __name__ == "__main__":
    main()