import csv
from collections import defaultdict

from feature_extraction import Config
import os
import json
from feature_extraction.Util.FileHandler import validate_path, read_csv, isFileExist
from feature_extraction.Util.analysis_utils import convert_bool2int, extract_feature, extract_neural_feature
import pandas as pd
'''
后续进行修改，目前仅限于对于单个神经网络模型提取特征
'''
def summaryNeural(neural_feat_dir):
    validate_path(neural_feat_dir)
    df=read_csv(neural_feat_dir)
    df=extract_neural_feature(df)
    return df

'''
后续进行修改，目前仅限于对于单个神经网络模型提取特征
'''
def summary(parent_dir):
    # for model_dir in os.listdir(parent_dir):
    #     # model_dir should not be a file
    #     if os.path.isfile(os.path.join(parent_dir, model_dir)):
    #         continue
    #
    #     print("\nModel dir: ", model_dir)
    #     acc_dict = defaultdict(list)
    #     # list all mutants
    #     for mutant_dir in os.listdir(os.path.join(parent_dir, model_dir)):
    #         # if mutant dir == "raw_data" or not a directory, then skip
    #         if mutant_dir == "raw_data" or os.path.isfile(os.path.join(parent_dir, model_dir, mutant_dir)):
    #             continue
    #
    #         print("\nMutant dir: ", mutant_dir)
    #         # validate sufficient iter
    #         existing_iterations = [f for f in os.listdir(os.path.join(parent_dir, model_dir, mutant_dir)) if
    #                                f.startswith("iter_")]
    #         # collect accuracy of 20 iters
    df_to_concat=[]
    cnt=0
    for model_dir in os.listdir(parent_dir):
        dataset_summary_dict = defaultdict()
        # model_dir should not be a file
        if os.path.isfile(os.path.join(parent_dir, model_dir)):
            continue

        print("\nModel dir: ", model_dir)

        acc_list = []
            # print("iter: ", iter_num)
        log_dir = os.path.join(parent_dir, model_dir,"log_dir",
                               "log.csv")
        feature_dir = os.path.join(parent_dir,model_dir,"result_dir",
                                   "monitor_features.csv")
        autotrainer_dir = os.path.join(parent_dir,model_dir,
                                   "result_dir", "monitor_detection.log")
        neuron_dir=os.path.join(parent_dir,model_dir,"result_dir","monitor_neural_features.csv")
        testActive_dir=os.path.join(parent_dir,model_dir,"result_dir","testActiv.csv")
        mutants_summary_dir=os.path.join(parent_dir,model_dir,"mut_summary.csv")
        flag=True
        flag=flag and isFileExist(log_dir)
        flag = flag and isFileExist(feature_dir)
        flag = flag and isFileExist(autotrainer_dir)
        flag = flag and isFileExist(neuron_dir)
        flag = flag and isFileExist(testActive_dir)
        flag = flag and isFileExist(mutants_summary_dir)
        if flag == False:
            print(f"Insufficent file to extract feature for {model_dir}")
            continue
        else:
            cnt+=1

    # validate file exist
    #     validate_path(log_dir)
    #     validate_path(feature_dir)
    #     validate_path(autotrainer_dir)

        # get fault number, get faults and labels

    # ################################################
    #   log.csv
    # val_loss,val_accuracy,loss,accuracy
        with open(log_dir, "r") as f:
            lines = [line for line in f.readlines() if line.strip()]
            heads = lines[0].strip('\n').split(',')
            values_last_line = lines[-1].strip("\n").split(",")

            # update dataset_summary_dict
            for head, value in zip(heads, values_last_line):
                dataset_summary_dict["ft_{}".format(head)] = value
            # get val_accuracy and add into acc_list
            var_acc = values_last_line[3]
            acc_list.append(float(var_acc))

    # ################################################
    # monitor_detection.csv
    # checktype,current_epoch,issue_list,time_usage,Describe
        with open(autotrainer_dir, "r") as f:
            lines = f.readlines()
            # print(lines)
            total_time = float(lines[-1].strip("\n").split(",")[-2])
            ave_time = total_time / (len(lines) - 1)
            autoTrainer_identified = lines[-1].strip("\n").split(",")[-1] != "No Issue now"

        # print("last line: {}".format(lines[-1]))
        # print("Average time = {} / {} = {}, autoTrainer: {}".format(total_time, len(lines) - 1, ave_time,
        #                                                             autoTrainer_identified))

            dataset_summary_dict["time"] = ave_time
            dataset_summary_dict["autoTrainer"] = "1" if autoTrainer_identified else "0"

    # ################################################
    # get features
        df = read_csv(feature_dir)
        df = df.fillna(0.0)

    # if has_enough_feature(df, min_feature=10) and has_enough_sample(df, min_sample=5):
    # preprocess, convert bool dtype ot int if necessary
        df = convert_bool2int(df)
    # print(df['not_converge'])
    # print(df['vanish_gradient'])
        feature_dict = extract_feature(df)
        for feat_key, feat_val in feature_dict.items():
            dataset_summary_dict[feat_key] = feat_val
        neuron_df = summaryNeural(neuron_dir)
        mut_summary_df = pd.read_csv(mutants_summary_dir)
        test_activ_df = pd.read_csv(testActive_dir)
        merged_df = mut_summary_df.merge(neuron_df, on=['layer_id', 'neuron_idx'], how='outer')
        merged_df = merged_df.merge(test_activ_df, on=['layer_id', 'neuron_idx'], how='outer')
        tot_summary_df = pd.DataFrame([feature_dict] * len(merged_df), index=merged_df.index)
        merged_df_optimized = pd.concat([merged_df, tot_summary_df], axis=1)
        merged_df_optimized.insert(0, 'model_name', model_dir)
        df_to_concat.append(merged_df_optimized)
    print(f"There are {cnt} models in total")
    all_summary = pd.concat(df_to_concat, axis=0, ignore_index=True)
    all_summary_path = os.path.join(parent_dir,"all_summary.csv")
    model_type_path=os.path.join(parent_dir,"task_type.csv")
    all_summary["model_name"] = all_summary["model_name"].astype(str)
    model_type_df = pd.read_csv(model_type_path)
    # 将 df2 的 model_name 转换为字符串
    model_type_df["model_name"] = model_type_df["model_name"].astype(str)
    df_merged = pd.merge(all_summary, model_type_df, on="model_name", how="inner")
    df_merged.to_csv(all_summary_path,index=False)


def summary_Single(model_dir,model_type="FCNN",isclass=1):
    df_to_concat = []
    dataset_summary_dict = defaultdict()
    parent_dir=os.path.abspath(os.path.join(model_dir, ".."))
    # model_dir should not be a file
    if os.path.isfile(model_dir):
        return

    print("\nModel dir: ", model_dir)

    acc_list = []
    # print("iter: ", iter_num)
    log_dir = os.path.join(model_dir, "log_dir",
                           "log.csv")
    feature_dir = os.path.join( model_dir, "result_dir",
                               "monitor_features.csv")
    autotrainer_dir = os.path.join(model_dir,
                                   "result_dir", "monitor_detection.log")
    neuron_dir = os.path.join(model_dir, "result_dir", "monitor_neural_features.csv")
    testActive_dir = os.path.join(model_dir, "result_dir", "testActiv.csv")
    mutants_summary_dir = os.path.join(model_dir, "mut_summary.csv")
    flag = True
    flag = flag and isFileExist(log_dir)
    flag = flag and isFileExist(feature_dir)
    flag = flag and isFileExist(autotrainer_dir)
    flag = flag and isFileExist(neuron_dir)
    flag = flag and isFileExist(testActive_dir)
    flag = flag and isFileExist(mutants_summary_dir)
    if flag == False:
        print(f"Insufficent file to extract feature for {model_dir}")
        return

    # validate file exist
    #     validate_path(log_dir)
    #     validate_path(feature_dir)
    #     validate_path(autotrainer_dir)

    # get fault number, get faults and labels

    # ################################################
    #   log.csv
    # val_loss,val_accuracy,loss,accuracy
    with open(log_dir, "r") as f:
        lines = [line for line in f.readlines() if line.strip()]
        heads = lines[0].strip('\n').split(',')
        values_last_line = lines[-1].strip("\n").split(",")

        # update dataset_summary_dict
        for head, value in zip(heads, values_last_line):
            dataset_summary_dict["ft_{}".format(head)] = value
        # get val_accuracy and add into acc_list
        var_acc = values_last_line[3]
        acc_list.append(float(var_acc))

    # ################################################
    # monitor_detection.csv
    # checktype,current_epoch,issue_list,time_usage,Describe
    with open(autotrainer_dir, "r") as f:
        lines = f.readlines()
        # print(lines)
        total_time = float(lines[-1].strip("\n").split(",")[-2])
        ave_time = total_time / (len(lines) - 1)
        autoTrainer_identified = lines[-1].strip("\n").split(",")[-1] != "No Issue now"

        # print("last line: {}".format(lines[-1]))
        # print("Average time = {} / {} = {}, autoTrainer: {}".format(total_time, len(lines) - 1, ave_time,
        #                                                             autoTrainer_identified))

        dataset_summary_dict["time"] = ave_time
        dataset_summary_dict["autoTrainer"] = "1" if autoTrainer_identified else "0"

    # ################################################
    # get features
    df = read_csv(feature_dir)
    df = df.fillna(0.0)

    # if has_enough_feature(df, min_feature=10) and has_enough_sample(df, min_sample=5):
    # preprocess, convert bool dtype ot int if necessary
    df = convert_bool2int(df)
    # print(df['not_converge'])
    # print(df['vanish_gradient'])
    feature_dict = extract_feature(df)
    for feat_key, feat_val in feature_dict.items():
        dataset_summary_dict[feat_key] = feat_val
    neuron_df = summaryNeural(neuron_dir)
    mut_summary_df = pd.read_csv(mutants_summary_dir)
    test_activ_df = pd.read_csv(testActive_dir)
    merged_df = mut_summary_df.merge(neuron_df, on=['layer_id', 'neuron_idx'], how='outer')
    merged_df = merged_df.merge(test_activ_df, on=['layer_id', 'neuron_idx'], how='outer')
    tot_summary_df = pd.DataFrame([feature_dict] * len(merged_df), index=merged_df.index)
    merged_df_optimized = pd.concat([merged_df, tot_summary_df], axis=1)
    md_name=os.path.basename(model_dir)
    merged_df_optimized.insert(0, 'model_name', md_name)
    all_summary = merged_df_optimized
    all_summary_path = os.path.join(model_dir, "all_summary.csv")
    model_type_path = os.path.join(parent_dir, "task_type.csv")
    all_summary["model_name"] = all_summary["model_name"].astype(str)
    # model_type_df = pd.read_csv(model_type_path)
    # # 将 df2 的 model_name 转换为字符串
    # model_type_df["model_name"] = model_type_df["model_name"].astype(str)
    # df_merged = pd.merge(all_summary, model_type_df, on="model_name", how="inner")
    all_summary["is_class"]=isclass
    all_summary["DNN_type"]=model_type
    all_summary.to_csv(all_summary_path, index=False)

    # return dataset_summary_dict

def get_neural_feat(recal,dir):
    prefix = Config.ouputNeuralSummary
    program_dir = dir
    summary_csv_file = "{}.csv".format(prefix)
    # summary_json_file = "{}_dict.json".format(prefix)
    summary_file_path = os.path.join(program_dir, summary_csv_file)
    print("Calculate Neural_summary.csv")
    prev_summary_dict = {}
    if recal or (not os.path.exists(summary_file_path)):
        print("Calculate Neural_summary.csv")
        df=summaryNeural(program_dir)
        df.to_csv(summary_file_path,index=False)
        print("Successfully calculate Neural_summary.csv")



def get_feat(recal,dir):
    prefix=Config.outputSummary
    program_dir=dir
    summary_csv_file = "{}.csv".format(prefix)
    summary_json_file = "{}_dict.json".format(prefix)
    summary_file_path = os.path.join(program_dir,summary_json_file)

    print("Calculate summary.csv")
    # get summary of features and labels
    if recal or (not os.path.exists(summary_file_path)):
        print("Calculate summary.csv")
        # get summary of features and labels
        summary_dict = summary(program_dir)  # {"a":{"b":{"c":{"d":"e"}}}}
        # print("summary_dict", summary_dict)

        prev_summary_dict = {}

        if not recal and os.path.exists(summary_file_path):
            with open(summary_file_path, 'r') as fr:
                try:
                    prev_summary_dict = json.load(fr)
                except Exception as e:
                    print("Load json from {} failed because {}".format(summary_file_path, e))

        summary_dict.update(prev_summary_dict)
        with open(summary_file_path, 'w') as fw:
            json.dump(summary_dict, fw)
    else:
        print("Load Neural_summary from {}.".format(summary_file_path))
        with open(summary_file_path, 'r') as fr:
            summary_dict = json.load(fr)
    dict2csv(summary_dict, os.path.join(program_dir,summary_csv_file))

def dict2csv(dataset_summary_dict, output_dir):
    # transfer dict to csv
    # print("dataset_summary_dict", dataset_summary_dict)
    with open(output_dir, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=dataset_summary_dict.keys())

        # 写入表头
        writer.writeheader()
        # 写入数据
        writer.writerow(dataset_summary_dict)
    print("Output to {}".format(output_dir))

if __name__ == "__main__":
    # get_feat(True,'..\\..\\Dataset\\all-bugs\\31880720')
     # get_neural_feat(True, '..\\..\\Dataset\\all-bugs\\31880720')
    # summary('..\\..\\Dataset\\all-bugs')
    summary_Single( '..\\..\\Dataset\\all-bugs\\44758894')
