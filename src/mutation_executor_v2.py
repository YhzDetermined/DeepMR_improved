from element_mutation_exec_info import ElementMutationExecInfo
import tensorflow as tf
from keras.models import load_model
import numpy as np
import tarfile
import os
from keras import backend as K
import io
import h5py

from Utils.utils import MutantNameParser
from runtime_calculator import RuntimeCalculator
import time
import math

from entity.ExecutionStrategyStatistic import ExecutionStatistic
from config.Config import mut_file_path, openLogType


class NewMutationExecutor:
    def __init__(self, test_case_splitter, comparator, layer_mutant_dict: dict):
        self.__test_case_splitter = test_case_splitter
        self.__comparator = comparator
        self.__passing_test_inputs = None
        self.__passing_test_outputs = None
        self.__failing_test_inputs = None
        self.__failing_test_old = None
        self.__failing_test_expected = None
        self.__tot_p = None
        self.__tot_f = None
        self.__mutants_total_count = None
        self.__non_viable_mutants_total_count = None
        self.__f2p_inputs = None
        self.__p2f_inputs = None
        self.__select_fraction = 1.0
        self.__debug = False
        self.model_name = None
        self.runCal = None
        self.layer_mutant_dict = layer_mutant_dict  # 每层被选择的变异体列表
        self.__selected_mutants_count = None
        self.layer_sus = dict()
        self.cur_layer_sus = None
        self.__exeRecord = ExecutionStatistic()
        self.CaseNotExecuteNum = 0  # 没有被运行的测试用例数
        self.MutantNotExecuteAllNum = 0  # 没有执行所有测试用例的变异体数量

    def test(self):
        os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
        self.__passing_test_inputs = np.asarray(self.__test_case_splitter.get_passing_test_inputs())
        self.__passing_test_outputs = self.__test_case_splitter.get_passing_test_outputs()
        self.__failing_test_inputs = np.asarray(self.__test_case_splitter.get_failing_test_inputs())
        self.__failing_test_old = self.__test_case_splitter.get_failing_test_actual_outputs()
        self.__failing_test_expected = self.__test_case_splitter.get_failing_test_expected_outputs()
        self.__tot_p = len(self.__passing_test_inputs)
        self.__tot_f = len(self.__failing_test_inputs)
        self.__mutants_total_count = 0
        self.__non_viable_mutants_total_count = 0

        # 选择的变异体，key:层号 value:选择的变异体列表
        selected_mutants = dict()
        # 变异体名和变异体文件的映射
        selected_mutants_mapping = dict()

        self.__selected_mutants_count = 0
        with tarfile.open(mut_file_path, 'r') as arc:
            for compressed_mutant in arc.getmembers():
                file_name = compressed_mutant.name
                if file_name.endswith('.h5'):
                    self.__mutants_total_count += 1
                    layer_id = file_name[:file_name.index('-')]
                    # 如果这层的变异体不被选择，这类变异体是需要执行的
                    mutant_name_true = file_name[:file_name.rfind("model.h5")]
                    layer_no, neuron_idx, mutant_oper = MutantNameParser(mutant_name_true)
                    flag = True
                    if layer_no == -1 or neuron_idx == -1 or mutant_oper == "":
                        if layer_id not in selected_mutants:
                            selected_mutants[layer_id] = []
                        selected_mutants[layer_id].append(compressed_mutant)
                    else:
                        selected_mutants_mapping[compressed_mutant.name] = compressed_mutant
            for layer_id, mutant_list in self.layer_mutant_dict.items():
                if layer_id not in selected_mutants:
                    selected_mutants[layer_id] = []
                for mutant_name in mutant_list:
                    selected_mutants[layer_id].append(selected_mutants_mapping[mutant_name])
            # 执行变异体
            for layer_id, compressed_mutants in selected_mutants.items():
                self.cur_layer_sus = 0.0
                if not compressed_mutants:
                    continue
                for compressed_mutant in compressed_mutants:
                    mutant_name = compressed_mutant.name
                    try:
                        # with tf.device('/CPU:0'):
                        arc.extract(compressed_mutant)
                        model = load_model(mutant_name)
                        self.__selected_mutants_count += 1
                        self.__exec_mutant(model)
                        '''删除模型相关的代码
                        '''
                        # import gc
                        # # 删除模型
                        # del model
                        # # 清理 TensorFlow 的会话内存（如果适用）
                        # K.clear_session()
                        # # 触发垃圾回收，释放 CPU 内存
                        # gc.collect()
                    except BaseException as error:
                        self.__non_viable_mutants_total_count = self.__non_viable_mutants_total_count + 1
                        print('Non-viable mutant ' + mutant_name)
                        print('Exception raised: {}'
                              ''.format(error))
                    finally:
                        os.remove(mutant_name)
                self.layer_sus[layer_id] = self.cur_layer_sus
        self.fill()

    def fill(self):
        self.__exeRecord.caseNotExecuteNum = self.CaseNotExecuteNum
        self.__exeRecord.mutantNotExecuteAllNum = self.MutantNotExecuteAllNum
        self.__exeRecord.selectedMutantNum = self.__selected_mutants_count
        self.__exeRecord.totalMutantNum = self.__mutants_total_count
        self.__exeRecord.caseCount = self.__tot_f + self.__tot_p

    def getExecutionStatistics(self):
        return self.__exeRecord

    def __exec_mutant(self, model):
        n_f2p = 0
        n_p2f = 0
        n_i_p = 0
        n_i_f = 0
        if len(self.__failing_test_inputs) > 0:
            p = model.predict(self.__failing_test_inputs)
            for i in range(0, len(p)):
                actual = p[i]
                expected = self.__failing_test_old[i]
                if not self.__comparator.compare(expected, actual):
                    n_i_f = n_i_f + 1
                expected = self.__failing_test_expected[i]
                if self.__comparator.compare(expected, actual):
                    # self.__f2p_inputs.add(hash(self.__failing_test_inputs[i].tobytes()))
                    n_f2p = n_f2p + 1
        #     根据ochiai公式计算出变异体怀疑度上界
        if len(self.__failing_test_inputs) == 0 or n_f2p == 0:
            sus0 = 0.0
        else:
            sus0 = float(n_f2p) / math.sqrt(float(len(self.__failing_test_inputs)) * float(n_f2p))
        if sus0 <= self.cur_layer_sus:
            self.CaseNotExecuteNum += len(self.__passing_test_inputs)
            if len(self.__passing_test_inputs) > 0:
                self.MutantNotExecuteAllNum += 1
            return
        if len(self.__passing_test_inputs) > 0:
            p = model.predict(self.__passing_test_inputs)
            for i in range(0, len(p)):
                actual = p[i]
                expected = self.__passing_test_outputs[i]
                if not self.__comparator.compare(expected, actual):
                    # self.__p2f_inputs.add(hash(self.__passing_test_inputs[i].tobytes()))
                    n_p2f = n_p2f + 1
                    n_i_p = n_i_p + 1
            if len(self.__failing_test_inputs) == 0 or n_i_f + n_i_p == 0:
                sus = 0.0
            else:
                sus = float(n_f2p) / math.sqrt(float(len(self.__failing_test_inputs)) * float(n_f2p + n_i_p))
            self.cur_layer_sus = max(self.cur_layer_sus, sus)
        if self.__debug:
            print('n_f2p=%d, n_p2f=%d, n_i_p=%d, n_i_f%d' % (n_f2p, n_p2f, n_i_p, n_i_f))

    #得到神经网络层的怀疑度
    def getLayerSus(self):
        return self.layer_sus


    def get_passing_tests_total_count(self):
        return self.__tot_p

    def get_failing_tests_total_count(self):
        return self.__tot_f

    def get_mutants_total_count(self):
        return self.__mutants_total_count

    def get_non_viable_mutants_total_count(self):
        return self.__non_viable_mutants_total_count

    def set_mutant_selection_fraction(self, fraction):
        self.__select_fraction = fraction

    def set_model_dict(self, model_name, dic):
        self.model_name = model_name
        self.runCal = RuntimeCalculator(model_name)
        self.runCal.set_dict(dic)

    def get_run_dict(self):
        self.runCal.integrate()
        run_dict, count_dict = self.runCal.to_dict()
        return run_dict

    def get_select_mutants_count(self):
        return self.__selected_mutants_count

    def test_v2(self):
        os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
        self.__passing_test_inputs = np.asarray(self.__test_case_splitter.get_passing_test_inputs())
        self.__passing_test_outputs = self.__test_case_splitter.get_passing_test_outputs()
        self.__failing_test_inputs = np.asarray(self.__test_case_splitter.get_failing_test_inputs())
        self.__failing_test_old = self.__test_case_splitter.get_failing_test_actual_outputs()
        self.__failing_test_expected = self.__test_case_splitter.get_failing_test_expected_outputs()
        self.__tot_p = len(self.__passing_test_inputs)
        self.__tot_f = len(self.__failing_test_inputs)
        self.__mutants_total_count = 0
        self.__non_viable_mutants_total_count = 0

        # 选择的变异体，key:层号 value:选择的变异体列表
        selected_mutants = dict()

        self.__selected_mutants_count = 0
        with tarfile.open(mut_file_path, 'r') as arc:
            for compressed_mutant in arc.getmembers():
                file_name = compressed_mutant.name
                if file_name.endswith('.h5'):
                    self.__mutants_total_count += 1
                    layer_id = file_name[:file_name.index('-')]
                    # 如果这层的变异体不被选择，这类变异体是需要执行的
                    mutant_name_true = file_name[:file_name.rfind("model.h5")]
                    layer_no, neuron_idx, mutant_oper = MutantNameParser(mutant_name_true)
                    flag = True
                    if layer_no == -1 or neuron_idx == -1 or mutant_oper == "":
                        if layer_id not in selected_mutants:
                            selected_mutants[layer_id] = []
                        selected_mutants[layer_id].append(file_name)
                        self.__selected_mutants_count += 1
            for layer_id, mutant_list in self.layer_mutant_dict.items():
                if layer_id not in selected_mutants:
                    selected_mutants[layer_id] = []
                for mutant_name in mutant_list:
                    selected_mutants[layer_id].append(mutant_name)
                    self.__selected_mutants_count += 1
            # 执行变异体
            total_mutants = self.__selected_mutants_count
            executed_mutants = 0
            next_percent = 5
            for layer_id, compressed_mutants in selected_mutants.items():
                self.cur_layer_sus = 0.0
                if not compressed_mutants:
                    continue
                for compressed_mutant in compressed_mutants:
                    mutant_name = compressed_mutant
                    try:
                        with tf.device('/CPU:0'):
                            with arc.extractfile(mutant_name) as f:
                                file_obj = io.BytesIO(f.read())
                                with h5py.File(file_obj, 'r') as h5f:
                                    model = load_model(h5f)
                            self.__exec_mutant(model)
                            executed_mutants += 1
                            if total_mutants > 0:
                                current_percent = executed_mutants * 100 // total_mutants
                                while current_percent >= next_percent and next_percent <= 100:
                                    self._log_progress(next_percent)
                                    next_percent += 5
                            '''删除模型相关的代码
                            '''
                            import gc
                            # 删除模型
                            del model
                            # 清理 TensorFlow 的会话内存（如果适用）
                            K.clear_session()
                            # 触发垃圾回收，释放 CPU 内存
                            gc.collect()
                    except BaseException as error:
                        self.__non_viable_mutants_total_count = self.__non_viable_mutants_total_count + 1
                        print('Non-viable mutant ' + mutant_name)
                        print('Exception raised: {}'
                              ''.format(error))
                    finally:
                        if os.path.exists(mutant_name):
                            os.remove(mutant_name)
                self.layer_sus[layer_id] = self.cur_layer_sus
        self.fill()

    def _log_progress(self, percentage: int):
        msg = '已执行%d%%变异体' % percentage
        print(msg)
        try:
            log_filename = '%d.txt' % int(self.__select_fraction * 100)
            with open(log_filename, 'a', encoding='utf-8') as f:
                f.write(msg + '\n')
        except Exception:
            # 日志写入失败时不影响主流程
            pass
        # try:
        #     with tarfile.open(mut_file_path, 'r') as arc:
        #         member = arc.getmember(mutant_name)
        #         f = arc.extractfile(member)
        #         model = load_model(mutant_name)
        #         self.__exec_mutant(model)
        # except BaseException as error:
        #     print('Exception raised: {}'
        #           ''.format(error))
        # finally:
        #     os.remove(mutant_name)


if __name__ == "__main__":
    pass
