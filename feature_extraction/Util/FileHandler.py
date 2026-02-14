import csv
import os
import pandas as pd

from feature_extraction.Config import testActiv


def validate_path(dir):
    if not os.path.exists(dir):
        raise FileNotFoundError("File Not Found! {}".format(dir))

def isFileExist(dir):
    if os.path.exists(dir):
        return True
    return False
def read_csv(file_path: str):
    df = None
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print("Read {} Failed.".format(file_path))
        print(e)
    return df

def write_activ_dict_to_csv(file_dir,test_activ_feat_dict):
    all_features = set()
    for neurons in test_activ_feat_dict.values():
        for features in neurons.values():
            all_features.update(features.keys())
    file_name=testActiv+".csv"
    file_path=os.path.join(file_dir,file_name)
    with open(file_path, mode='w', newline='') as csvfile:

        writer = csv.writer(csvfile)

        # 写入表头
        header = ["layer_id", "neuron_idx"] + sorted(all_features)
        writer.writerow(header)
        # 遍历嵌套字典并写入数据
        for layer_id, neurons in test_activ_feat_dict.items():
            for neuron_idx, features in neurons.items():
                row = [layer_id, neuron_idx] + [features.get(feature, '') for feature in sorted(all_features)]
                writer.writerow(row)
    print(f"Write into {file_path} successfully")


