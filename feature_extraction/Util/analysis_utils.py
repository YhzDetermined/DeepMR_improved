import pandas as pd

from feature_extraction.Config import OPERATORS, Neural_Common_Feature


def convert_bool2int(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        # 如果列的数据类型为 object，尝试进行转换
        if df[col].dtype == 'object':
            # 尝试将布尔字符串转换为数值
            df[col] = df[col].replace({'True': 1, 'False': 0})
            # 使用 pd.to_numeric 进行转换，无法转换的设置为 NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')
        elif df[col].dtype == 'bool':
            # 布尔类型直接转换为 int
            df[col] = df[col].astype('float32')
        else:
            # 对于其他数值类型，不做处理
            pass
    return df

def extract_feature(df: pd.DataFrame):
    feature_dict = {}
    features = {k: OPERATORS for k in df.columns}
    extracted_feat = df.agg(features).to_dict()
    for para, values in extracted_feat.items():
        for p, v in values.items():
            key = "ft_{}_{}".format(para, p)

            # handle exceptional value
            if type(v) == str and (v == "0" or v == v == "False"):
                v = 0.0

            if type(v) == str and (v == "1" or v == "True"):
                v = 1.0

            if type(v) != float:
                print("Type", type(v), v)
            feature_dict[key] = v
    return feature_dict

def extract_neural_feature(df: pd.DataFrame):
    oper=OPERATORS[:]
    oper.append("last")
    grouped = df.groupby(Neural_Common_Feature).agg(oper)

    # 重命名列，按照 nft_原特征名_运算符名 的格式

    grouped.columns = [f"nft_{col[0]}_{col[1]}" for col in grouped.columns]

    # 重置索引，使分组键恢复为普通列

    grouped.reset_index(inplace=True)
    return grouped
