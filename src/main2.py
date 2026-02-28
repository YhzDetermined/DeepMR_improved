import json

import numpy as np


import mutation_generator
import test_case_splitter
import os
import metallaxis
import muse
import time
import sys
import mutation_executor_v2
from config import Config
from config.Config import mut_file_path, isRecord
import mutation_generator_v2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ML_rank.Mutant_Prior import MutantPrior
from comparator import comparator_factory
from Utils.utils import write_to_file, convert_keys_to_int, append_dict_to_csv, append_scores_to_csv, getIsClass, \
    append_execution_statistic_to_csv, exists_model_name
from feature_extraction.Util.cal_stats import summary, summary_Single

if __name__ == '__main__':
    if len(sys.argv) != 6 and len(sys.argv) != 7:
        raise ValueError('This program expects 6 or 7 command-line arguments')
    model_file = sys.argv[1]
    mutant_ratio = float(sys.argv[2])  #选择的变异体比例
    X_test = np.load(sys.argv[3])
    y_test = np.load(sys.argv[4])
    model_kind = sys.argv[5]
    delta = 1e-3
    print("选择变异体比例：" + str(mutant_ratio))
    last_index = max(model_file.rfind('/'), model_file.rfind('\\'))
    base_dir=model_file[:last_index]
    model_name=int(os.path.basename(os.path.normpath(base_dir)))
    if len(sys.argv) == 7:
        delta = float(sys.argv[6])
    fraction=1
    if isRecord == 1:
        #判断表格中有没有该模型的记录
        isExist = exists_model_name(Config.StatsFileName,model_name)
        if isExist == True:
            print("该模型已经完成测试")
            sys.exit(0)
    Isclass=getIsClass(model_kind)
    comparator = comparator_factory(model_kind, delta)
    print("State 1 finish!")
    tmp_dict = os.path.join(base_dir, "result_dir", "mut_dict.json")
    mut_gen_tot_time = 0
    # 根据配置决定日志文件的打开方式：1 为追加(a)，0 为覆盖写入(w)
    log_mode = 'a' if Config.openLogType == 1 else 'w'
    # 读取 JSON 文件
    with open('%d.txt' % int(fraction * 100), log_mode) as out_file:
        if not os.path.isfile(mut_file_path) or not os.path.isfile(tmp_dict):
            if os.path.exists(mut_file_path):
                os.remove(mut_file_path)
            start = time.time()
            mg = mutation_generator_v2.MutationGeneratorV2(model_file)
            # mg.apply_del_layer()
            mg.apply_dup_layer()
            mg.apply_math_weight()
            mg.apply_math_bias()
            mg.apply_math_weight_conv()
            mg.apply_math_bias_conv()
            mg.apply_math_filters()
            mg.apply_math_kernel_sz()
            mg.apply_math_strides()
            mg.apply_math_pool_sz()
            mg.apply_padding_replacement()
            mg.apply_activation_function_replacement()
            # mg.apply_del_layer()
            # mg.apply_dup_layer()
            mg.apply_math_lstm_input_weight()
            mg.apply_math_lstm_forget_weight()
            mg.apply_math_lstm_cell_weight()
            mg.apply_math_lstm_output_weight()
            mg.apply_math_lstm_input_bias()
            mg.apply_math_lstm_forget_bias()
            mg.apply_math_lstm_cell_bias()
            mg.apply_math_lstm_output_bias()
            mg.apply_recurrent_activation_function_replacement()
            # mut_dict = mg.get_dict()
            mg.store_dict()
            # mut_gen_time_dict,mut_gen_count_dict=mg.get_run_count_dict()
            mg.close()
            end = time.time()
            #变异体生成时间
            mut_gen_tot_time = end - start
            print('model %s: Mutation generation took %s seconds' % (model_name,end - start))
            out_file.write('model %s: Mutation generation took %s seconds\n' % (model_name,end - start))
        else:
            print("Hello World!")
        #加载变异体的特征
        with open(tmp_dict, 'r') as file:
            mut_dict = json.load(file)
        mut_dict = convert_keys_to_int(mut_dict)
        start = time.time()
        s = test_case_splitter.TestCaseSplitter(model_file, X_test, y_test, comparator)
        s.split()
        passDict={
            'pass':len(s.get_passing_test_outputs()),
            'fail':len(s.get_failing_test_actual_outputs())
        }
        end = time.time()
        test_split_time = end - start
        print('Test case splitting took %s seconds' % (end - start))
        out_file.write('model %s: Test case splitting took %s seconds\n' % (model_name, end - start))
        #将变异体特征存入表格中
        write_to_file(base_dir, mut_dict, passDict, s.loss_func)
        #得到所有特征
        print('Selected %2f%% of the mutants' % (mutant_ratio * 100))
        summary_Single(model_dir=base_dir,isclass=Isclass,model_type=s.getModelType())
        mp_xgb=MutantPrior(select_ratio=mutant_ratio,file_path=os.path.join(base_dir,"all_summary.csv"))
        mp_xgb.process()
        layer_mutant_dict = mp_xgb.getLayerMutantDict()
        start = time.time()
        # mt = mutation_executor.MutationExecutor(s, comparator)
        mt = mutation_executor_v2.NewMutationExecutor(s,comparator,layer_mutant_dict)
        mt.set_mutant_selection_fraction(fraction)
        mtr = mt.test_v2()
        exeStat = mt.getExecutionStatistics()
        end = time.time()
        exeStat.setExecuteTime(end - start)
        exeStat.setGenerateTime(mut_gen_tot_time)
        exeStat.setSelectRatio(mutant_ratio)
        print('Mutation execution took %s seconds' % (end - start))
        out_file.write('model %s: Mutation execution took %s seconds\n' % (model_name, end - start))
        print('DeepMPrior selected %d mutants of %d mutants, among them %d turned out to be non-viable'
              % (mt.get_select_mutants_count(),mt.get_mutants_total_count(), mt.get_non_viable_mutants_total_count()))
        out_file.write('model %s: DeepMPrior selected %d mutants of %d mutants, among them %d turned out to be non-viable\n'
              % (model_name, mt.get_select_mutants_count(),mt.get_mutants_total_count(), mt.get_non_viable_mutants_total_count()))
        print(mt.getLayerSus())
        if isRecord == 1:
            append_execution_statistic_to_csv(model_name,exeStat,Config.StatsFileName)
        #按照配置将各层的怀疑度写入csv中
        if Config.recordSus == True:
            append_scores_to_csv(mt.getLayerSus(), model_name)
        #按照配置删除变异体
        if Config.deleteMutFile == True:
            if os.path.isfile(mut_file_path):
                os.remove(mut_file_path)
                print("变异体已成功删除")

