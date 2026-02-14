import csv
import os.path

from keras.layers import Dense

from config.Config import ExeStatsRows
from entity.ExecutionStrategyStatistic import ExecutionStatistic


def MutantNameParser(name:str):
    subname=name.split("-")
    if len(subname)!=3:
        return -1,-1,""
    else:
        layer_pos_str,neural_pos_str,mutant_oper=subname[0],subname[1],subname[2]
        if layer_pos_str.startswith("L") and neural_pos_str.startswith("N"):
            layer_idx=int(layer_pos_str[1:])
            neural_idx=int(neural_pos_str[1:])
            return layer_idx,neural_idx,mutant_oper
        else:
            return -1,-1,""

def write_to_file(basedir,nested_dict,passDict,loss_func):
    # file_name = 'result_dir/mutant_sus.csv'
    # # file_name = 'mutant_sus.csv'
    # csv_file=basedir+file_name
    # # 写入 CSV 文件
    # header=['layer_idx', 'neural_idx', 'mutant_oper', 'ochiai']
    # header.extend(passDict.keys())
    # with open(csv_file, mode='w', newline='') as file:
    #     writer = csv.writer(file)
    #
    #     # 写入表头
    #     writer.writerow(header)
    #
    #     # 遍历字典，将键值对写入 CSV 文件
    #     for (a, b, c), [d] in dic.items():
    #         row=[a,b,c,d]
    #         extra_values = [passDict[key] for key in passDict.keys()]
    #         # 合并原始数据和额外的特征值
    #         writer.writerow(row + extra_values)
    # print(f"数据已成功写入 {csv_file}")
    file_name="mut_summary.csv"
    mut_csv=os.path.join(basedir,file_name)
    all_fields = set()
    for layer_id in nested_dict:
        for neuron_idx in nested_dict[layer_id]:
            for mutant_oper in nested_dict[layer_id][neuron_idx]:
                all_fields.update(nested_dict[layer_id][neuron_idx][mutant_oper].keys())

    # 固定的前几个字段
    fieldnames = ['layer_id', 'neuron_idx', 'mutant_oper', 'pass', 'fail','loss_func'] + sorted(all_fields)

    # 打开 CSV 文件写入
    with open(mut_csv, mode='w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        # 遍历嵌套字典并展平数据
        for layer_id, neurons in nested_dict.items():
            for neuron_idx, mutants in neurons.items():
                for mutant_oper, features in mutants.items():
                    row = {
                        'layer_id': layer_id,
                        'neuron_idx': neuron_idx,
                        'mutant_oper': mutant_oper,
                        'pass': passDict['pass'],  # 添加 pass 值
                        'fail': passDict['fail'],  # 添加 fail 值
                        'loss_func':loss_func
                    }
                    row.update(features)  # 添加特征
                    writer.writerow(row)

def convert_keys_to_int(d):
    if isinstance(d, dict):
        new_dict = {}
        for key, value in d.items():
            # 如果键是字符串，并且符合整数的格式，则转换为整数
            if isinstance(key, str) and key.isdigit():
                key = int(key)
            # 对字典进行递归调用
            new_dict[key] = convert_keys_to_int(value)
        return new_dict
    elif isinstance(d, str) and d.isdigit():
        return int(d)
    else:
        return d

def get_dense_weights(model):
    """
    提取模型中 Dense 层的权重和偏置。

    参数：
    - model: Keras 的 Sequential 或 Functional 模型对象。

    返回：
    - dense_weights: 一个列表，包含所有 Dense 层的权重和偏置。
      格式：[weights_1, biases_1, weights_2, biases_2, ...]
    """
    dense_weights = []
    for layer in model.layers:
        if isinstance(layer, Dense):  # 检查是否是 Dense 层
            dense_weights.extend(layer.get_weights())  # 获取权重和偏置
    return dense_weights


def append_dict_to_csv(data_dict):
    # 检查文件是否存在且不为空
    file_name='./runtime_record.csv'
    file_exists = os.path.exists(file_name) and os.path.getsize(file_name) > 0
    # 打开 CSV 文件
    with open(file_name, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=data_dict.keys())
        # 如果文件不存在或为空，写入表头
        if not file_exists:
            writer.writeheader()
        # 写入字典数据
        writer.writerow(data_dict)

def append_execution_statistic_to_csv(model_name, statistic: ExecutionStatistic, csv_path):
    # CSV 表头（字段名）
    # 一行数据
    row = [
        model_name,
        statistic.caseCount,
        statistic.totalMutantNum,
        statistic.generateTime,
        statistic.selected_ratio,
        statistic.selectedMutantNum,
        statistic.executeTime,
        statistic.caseNotExecuteNum,
        statistic.mutantNotExecuteAllNum,
    ]

    # 判断文件是否存在，用于是否写表头
    file_exists = os.path.exists(csv_path)

    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(ExeStatsRows)
        writer.writerow(row)
def append_scores_to_csv(max_ochiai_scores, model_name):
    """
    将字典中的每个键值对 (key, value) 作为一行数据添加到 CSV 文件中。
    如果 CSV 文件不存在或为空，先创建文件并写入表头。

    参数:
    - max_ochiai_scores (dict): 包含 layer_id 和 sus 分数的字典。
    - model_name (int): 模型名称的标识符。
    - csv_file (str): CSV 文件的路径。
    """
    csv_file='./model_layer_sus.csv'
    # 定义 CSV 文件的表头
    fieldnames = ['model_name', 'layer_id', 'sus']

    # 检查文件是否存在且不为空
    file_exists = os.path.exists(csv_file) and os.path.getsize(csv_file) > 0

    # 以追加模式打开文件
    with open(csv_file, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        # 如果文件不存在或为空，写入表头
        if not file_exists:
            writer.writeheader()

        # 遍历字典并写入每一行
        for layer_id, sus in max_ochiai_scores.items():
            writer.writerow({
                'model_name': model_name,
                'layer_id': layer_id,
                'sus': sus
            })

import csv
import os

def exists_model_name(csv_file_path: str, target_value: int) -> bool:
    """
    判断 csv 文件中 model_name 列是否存在等于 target_value 的记录
    文件不存在时直接返回 False
    """
    # 1. 判断文件是否存在
    if not os.path.isfile(csv_file_path):
        return False

    target_str = str(target_value)
    try:
        with open(csv_file_path, mode='r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            # 2. 判断列是否存在
            if not reader.fieldnames or 'model_name' not in reader.fieldnames:
                return False
            # 3. 逐行检查
            for row in reader:
                value = row.get('model_name')
                if value is None:
                    continue
                if value.strip() == target_str:
                    return True
    except Exception:
        # 文件损坏、编码错误等情况，统一认为不存在
        return False
    return False

def getIsClass(kind):
    if kind == 'regression' or kind == 'reg':
        return 1
    elif kind == 'classification' or kind == 'class':
        return 0
    return -1
def get_network_type(model):
    def check_layer(layer):
        lstm = rnn = cnn = False
        cls_name = layer.__class__.__name__
        if cls_name == 'LSTM':
            lstm = True
        elif cls_name in ['SimpleRNN', 'GRU']:
            rnn = True
        elif cls_name in ['Conv1D', 'Conv2D', 'Conv3D']:
            cnn = True
        # 检查包装层如Bidirectional
        if cls_name in ['Bidirectional', 'TimeDistributed']:
            sub = layer.layer
            sub_lstm, sub_rnn, sub_cnn = check_layer(sub)
            lstm |= sub_lstm
            rnn |= sub_rnn
            cnn |= sub_cnn
        return lstm, rnn, cnn

    has_lstm = has_rnn = has_cnn = False
    for layer in model.layers:
        l, r, c = check_layer(layer)
        has_lstm |= l
        has_rnn |= r
        has_cnn |= c

    return ('LSTM' if has_lstm else
            'RNN' if has_rnn else
            'CNN' if has_cnn else
            'FCNN')