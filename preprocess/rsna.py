import pickle
import numpy as np
import pandas as pd
from MGCA.constants import *
from sklearn.model_selection import train_test_split

np.random.seed(0)


# create bounding boxes
def create_bbox(row):  # 生成bounding box
    if row["Target"] == 0:  # 不存在检测目标
        return 0
    else:
        x1 = row["x"]  # 左侧边界
        y1 = row["y"]  # 上侧边界
        x2 = x1 + row["width"]  # 右侧边界
        y2 = y1 + row["height"]  # 下侧边界
        return [x1, y1, x2, y2]


def preprocess_rsna_data(test_fac=0.15):
    try:
        df = pd.read_csv(RSNA_ORIGINAL_TRAIN_CSV)
    except:
        raise Exception(
            "Please make sure the the RSNA_Pneumonia RSNA_Pneumonia dataset is \
            stored at {RSNA_DATA_DIR}"
        )

    # class_df = pd.read_csv(RSNA_CLASSINFO_CSV)
    # all_df = pd.merge()

    df["bbox"] = df.apply(lambda x: create_bbox(x), axis=1)

    # aggregate multiple boxes
    df = df[["patientId", "bbox"]]
    df = df.groupby("patientId").agg(list)
    df = df.reset_index()
    df["bbox"] = df["bbox"].apply(lambda x: None if x == [0] else x)

    # create labels
    df["Target"] = df["bbox"].apply(lambda x: 0 if x == None else 1)

    # no encoded pixels mean healthy
    # df["Path"] = df["patientId"].apply(
    #     lambda x: RSNA_IMG_DIR / (x + ".dcm"))

    # split data
    train_df, test_val_df = train_test_split(
        df, test_size=test_fac * 2, random_state=0)
    test_df, valid_df = train_test_split(
        test_val_df, test_size=0.5, random_state=0)

    print(f"Number of train samples: {len(train_df)}")
    print(train_df["Target"].value_counts())
    print(f"Number of valid samples: {len(valid_df)}")
    print(valid_df["Target"].value_counts())
    print(f"Number of test samples: {len(test_df)}")
    print(test_df["Target"].value_counts())

    train_df.to_csv(RSNA_TRAIN_CSV, index=False)
    valid_df.to_csv(RSNA_VALID_CSV, index=False)
    test_df.to_csv(RSNA_TEST_CSV, index=False)


def prepare_detection_pkl(df, path):
    filenames = []
    bboxs = []  # 存放bounding box四元组
    for row in df.itertuples():
        filename = row.patientId + ".dcm"  # 文件位置加后缀
        filenames.append(filename)
        if row.Target == 0:  # 有无疾病
            bboxs.append(np.zeros((1, 4)))
        else:
            y = np.array(row.bbox)
            bboxs.append(y)

    filenames = np.array(filenames)
    bboxs = np.array(bboxs)

    with open(path, "wb") as f:
        pickle.dump([filenames, bboxs], f)


def prepare_detection_data():
    try:
        df = pd.read_csv(RSNA_ORIGINAL_TRAIN_CSV)
    except:
        raise Exception(
            "Please make sure the the RSNA_Pneumonia RSNA_Pneumonia dataset is \
            stored at {RSNA_DATA_DIR}"
        )

    # class_df = pd.read_csv(RSNA_CLASSINFO_CSV)
    # all_df = pd.merge()

    df["bbox"] = df.apply(lambda x: create_bbox(x), axis=1)  # 生成bounding box

    # aggregate multiple boxes
    df = df[["patientId", "bbox"]]  # 保留id和box四元组
    df = df.groupby("patientId").agg(list)  # 将有多个box的id进行组合，groupby：根据id进行组合，将相同id合并；agg：合并方式为列举，即将所有box作为该id的box
    df = df.reset_index()  # 上一步将'patientId'作为index，失去了该列的名字，通过reset_index()恢复索引，从而恢复'patientId'的名字，也可直接在groupby中加上参数as_index=False
    df["bbox"] = df["bbox"].apply(lambda x: None if x == [0] else x)  # 将没有box的id置none

    # create labels
    df["Target"] = df["bbox"].apply(lambda x: 0 if x == None else 1) # 疾病标签

    # split data
    train_df, test_val_df = train_test_split(
        df, test_size=5337 * 2, random_state=0)
    test_df, valid_df = train_test_split(
        test_val_df, test_size=0.5, random_state=0)

    print(f"Number of train samples: {len(train_df)}")
    print(train_df["Target"].value_counts())
    print(f"Number of valid samples: {len(valid_df)}")
    print(valid_df["Target"].value_counts())
    print(f"Number of test samples: {len(test_df)}")
    print(test_df["Target"].value_counts())

     # 保存
    prepare_detection_pkl(
        train_df, RSNA_DETECTION_TRAIN_PKL)
    prepare_detection_pkl(
        valid_df, RSNA_DETECTION_VALID_PKL)
    prepare_detection_pkl(test_df, RSNA_DETECTION_TEST_PKL)



if __name__ == "__main__":
    # preprocess_rsna_data()
    prepare_detection_data()