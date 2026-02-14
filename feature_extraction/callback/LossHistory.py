import os
import sys

from src.Utils.utils import get_dense_weights
from ..Util.dataConvert import process_and_convert_activations, convert_activations

sys.path.append('.')
import csv
import numpy as np
import keras
from ..Util.Logger import Logger
import keras.backend as K
import feature_extraction.callback.Monitor as mn
import pickle
import time
from collections import defaultdict
import tensorflow as tf
from keras.layers import Dense, Activation, LeakyReLU

# import modules as md

logger = Logger()

default_param = {'beta_1': 1e-3,
                 'beta_2': 1e-4,
                 'beta_3': 70,
                 'gamma': 0.7,
                 'zeta': 0.03,
                 'eta': 0.2,
                 'delta': 0.01,
                 'alpha_1': 0,
                 'alpha_2': 0,
                 'alpha_3': 0,
                 'Theta': 0.7
                 }


class LossHistory(keras.callbacks.Callback):
    def __init__(self, training_data, model, batch_size, total_epoch, save_dir,determine_threshold=5,
                 satisfied_acc=0.7,
                 checktype='epoch_5', satisfied_count=3, retrain=False, pkl_dir=None, solution=None,
                 params={}):  # only support epoch method now
        """[summary]

        Args:
            training_data ([list]): [training data]
            model ([model]): [untrained model]
            batch_size ([int]): [batch size]
            save-dir([str]):[the dir to save the detect result]
            checktype (str, optional): [checktype,'a_b', a can be chosen from ['epoch', 'batch'], b is number, it means the monitor will check \
            the gradient and loss every 'b' 'a'.]. Defaults to 'epoch_5'.
            satisfied_acc (float, optional): [the satisfied accuracy, when val accuracy beyond this, the count ++, when the count is bigger or\
                equal to satisfied_count, training stop.]. Defaults to 0.7.
            satisfied_count (int, optional): []. Defaults to 3.
        """
        self.evaluated_gradients = 0
        self.trainX = training_data[0]
        self.trainy = training_data[1]
        self.batch_size = batch_size
        self.model = model
        self.satisfied_acc = satisfied_acc
        self.satisfied_count = satisfied_count
        self.count = 0
        self.checktype = checktype.split('_')[0]
        self.checkgap = int(checktype.split('_')[-1])
        self.issue_list = []
        self.feature_list = []
        self.neural_feature_list=[]  #每个神经元的特征信息
        self.save_dir = save_dir
        if not os.path.exists(save_dir):  # record monitor and repair message
            os.makedirs(save_dir)
        self.pkl_dir = pkl_dir
        self.retrain = retrain
        self.total_epoch = total_epoch
        self.determine_threshold = determine_threshold
        self.params = params
        if self.params == {}:
            self.params = default_param
        self.f = None
        self.history = defaultdict(list)
        self.history['loss'] = []
        self.history['acc'] = []
        self.history['val_loss'] = []
        self.history['val_acc'] = []
        self.Monitor = mn.IssueMonitor(total_epoch, self.satisfied_acc, self.params, self.determine_threshold)
        # 层数与激活函数的索引
        self.activate_list=[]
        self.activate_dict={}
        self.layer_idx={}
        self.start_time = time.time()

        # name of log file
        self.log_name = '{}_{}.log'.format('monitor', 'detection')

        if self.retrain:
            self.log_name = '{}_{}.log'.format('monitor', 'repair')
            self.solution = solution
        self.log_name = os.path.join(self.save_dir, self.log_name)
        print(self.log_name)
        # name of features file
        self.feature_name = os.path.join(self.save_dir, '{}_{}.csv'.format('monitor', 'features'))
        self.neural_feature_name=os.path.join(self.save_dir, '{}_{}_{}.csv'.format('monitor','neural','features'))
        print(self.feature_name)
        if os.path.exists(self.log_name):
            # avoid the repeat writing
            os.remove(self.log_name)

        if os.path.exists(self.feature_name):
            # avoid the repeat writing
            os.remove(self.feature_name)

        if os.path.exists(self.neural_feature_name):
            os.remove(self.neural_feature_name)
        # open two files
        self.log_file = open(self.log_name, 'a+')
        self.feature_log_file = open(self.feature_name, 'a+')
        self.neural_feature_log_file=open(self.neural_feature_name,'a+')

        # write head of two files
        self.log_file.write(
            '{},{},{},{},{}\n'.format('checktype', 'current_epoch', 'issue_list', 'time_usage', 'Describe'))

        self.feature_log_file.write(','.join(self.Monitor.get_features().keys()) + '\n')
        self.neural_feature_log_file.write(','.join(self.Monitor.neural_feature_name)+'\n')
        ### add code generated_by_gpt
        # self.activation_log_name = os.path.join(self.save_dir, '{}_{}_{}.csv'.format('monitor', 'activation', 'stats'))
        # if os.path.exists(self.activation_log_name):
        #     os.remove(self.activation_log_name)
        # self.activation_log_file = open(self.activation_log_name, 'a+')

    # def on_train_begin(self, logs=None):
    #     weights = self.model.trainable_weights  # get trainable weights
    #     grads = self.model.optimizer.get_gradients(self.model.total_loss, weights)
    #     '''
    #     update by Huaizhi on 2024-10-28
    #     '''
    #     # with tf.GradientTape() as tape:
    #     #     predictions = self.model(self.trainX, training=True)  # 前向传播
    #     #     loss = self.model.compiled_loss(self.trainy, predictions)  # 计算损失
    #     #
    #     # # 获取模型的可训练参数
    #     # weights = self.model.trainable_variables
    #     #
    #     # # 计算梯度
    #     # grads = tape.gradient(loss, weights)
    #     # input, corresponding label, weight of each sample(all of them are 1), learning rate(we set it to 0)
    #     symb_inputs = [self.model._feed_inputs, self.model._feed_targets, self.model._feed_sample_weights,
    #                    K.learning_phase()]
    #     self.f = K.function(symb_inputs, grads)
    #     if self.retrain:
    #         self.log_file.write(
    #             '-----Using {} solution to retrain Detail can be found in the directory!-----\n'.format(self.solution))
    def on_train_begin(self, logs=None):
        weights = self.model.trainable_variables  # 请注意，这里使用的是 trainable_variables
        # self.Monitor.last_weight=self.model.get_weights()
        # self.Monitor.initial_weight=self.model.get_weights()
        idx = 0
        # self.model.summary()
        num_layers = len(self.model.layers)
        self.Monitor.last_weight = get_dense_weights(self.model)
        self.Monitor.initial_weight= get_dense_weights(self.model)
        dict = {}
        neuron_counts={}
        dense_count = 0
        # 建立神经网络层和DENSE 层的索引
        for layer_index, layer in enumerate(self.model.layers):
            if isinstance(layer, Dense):
                dense_count += 1  # 当前Dense层计数
                self.layer_idx[dense_count] = layer_index
        self.Monitor.setDenseMapping(self.layer_idx)
        for idx, layer in enumerate(self.model.layers):
            if isinstance(layer, Dense):
                # 获取 Dense 层的输入和输出神经元数量
                input_neurons = layer.input_shape[-1]
                output_neurons = layer.units
                # 存入字典，键为层号，值为 (输入神经元, 输出神经元)
                neuron_counts[idx] = (input_neurons, output_neurons)
        self.check_activation_layers()
        print(self.activate_dict)
        # print(self.activate_list)
        self.Monitor.setNeuralCount(neuron_counts)
        ###add code generated by gpt
        # 写入激活值日志文件的表头
        # self.activation_log_file.write('epoch,layer_index,neuron_index,mean,var\n')
        layer_outputs = [self.model.layers[i].output for i in self.activate_dict]
        # 创建用于获取激活值的模型
        # self.activation_model = keras.models.Model(inputs=self.model.input,
        #                                            outputs=[layer.output for layer in self.model.layers])
        self.activation_model = keras.models.Model(inputs=self.model.input,
                                                   outputs=layer_outputs)
        # self.Monitor.setActivateDict(self.activate_dict)
        self.Monitor.setActivateList(self.activate_list)
        # 定义一个计算梯度的函数
        @tf.function
        def compute_grads(inputs, targets, sample_weights, training=True):
            # with tf.GradientTape() as tape:
            #     # 前向传播，获取预测值
            #     predictions = self.model(inputs, training=training)
            #     # 计算损失
            #     loss = self.model.compiled_loss(
            #         targets, predictions, sample_weights
            #     )
            #     # 添加其他损失（如正则化损失）
            #     loss += sum(self.model.losses)
            # # 计算梯度
            # grads = tape.gradient(loss, weights)
            # return grads
            with tf.GradientTape() as tape:
                # Forward pass to get predictions
                predictions = self.model(inputs, training=training)
                # Compute loss
                loss = self.model.compiled_loss(targets, predictions, sample_weights)
                # Add any other losses (like regularization losses)
                loss += sum(self.model.losses)
            # Get the trainable variables of the Dense layers
            dense_trainable_vars = []
            for layer in self.model.layers:
                if isinstance(layer, tf.keras.layers.Dense):
                    dense_trainable_vars.extend(layer.trainable_variables)
            # Compute gradients with respect to Dense layers' variables
            grads = tape.gradient(loss, dense_trainable_vars)
            return grads

        # 保存计算梯度的函数
        self.f = compute_grads

        if self.retrain:
            self.log_file.write(
                '-----Using {} solution to retrain. Details can be found in the directory!-----\n'.format(self.solution)
            )


    # def on_batch_end(self, batch, logs=None):
    #     if logs is None:
    #         logs = {}
    #     self.history['loss'].append(logs.get('loss'))
    #     self.history['acc'].append(logs.get('accuracy'))
    #     self.history['val_loss'].append(logs.get('val_loss'))
    #     self.history['val_acc'].append(logs.get('val_accuracy'))
    #
    #     if self.checktype == 'batch' and batch % self.checkgap == 0:
    #         trainingExample = self.trainX[0: self.batch_size]
    #         trainingY = self.trainy[0: self.batch_size]
    #         x, y, sample_weight = self.model._standardize_user_data(trainingExample, trainingY)
    #         # output_grad = self.f(x + y + sample_weight)
    #         self.evaluated_gradients = self.f([x, y, sample_weight, 0])
    #         gradient_list = []
    #         for i in range(len(self.evaluated_gradients)):
    #             if isinstance(self.evaluated_gradients[i], np.ndarray):
    #                 gradient_list.append(self.evaluated_gradients[i])
    #
    #         self.issue_list = self.Monitor.determine(self.model, self.history, gradient_list, self.checkgap)
    #         self.feature_list = self.Monitor.get_features()
    #         # self.issue_list = md.filtered_issue(self.issue_list)
    #
    #         self.evaluated_gradients = None
    #
    #         dictwriter_object = csv.DictWriter(self.feature_log_file, fieldnames=self.feature_list.keys())
    #         dictwriter_object.writerow(self.feature_list)
    #         self.feature_log_file.flush()
    #
    #         if not self.retrain:
    #             if not self.issue_list:
    #                 self.log_file.write('{},{},{},{},{}\n'.format(self.checktype, batch, self.issue_list,
    #                                                               str(time.time() - self.start_time), 'No Issue now'))
    #                 self.log_file.flush()
    #             elif self.issue_list == ['need_train']:
    #                 self.issue_list = list(set(self.issue_list))
    #                 self.log_file.write('{},{},{},{},{}\n'.format(self.checktype, batch, self.issue_list,
    #                                                               str(time.time() - self.start_time),
    #                                                               'NO training problems now. You need to train this '
    #                                                               'new model more times.'))
    #                 self.log_file.flush()
    #                 print('NO training problems now. You need to train this new model more iterations.')
    #             else:
    #                 self.issue_list = list(set(self.issue_list))
    #                 self.log_file.write('{},{},{},{},{}\n'.format(self.checktype, batch, self.issue_list,
    #                                                               str(time.time() - self.start_time),
    #                                                               'Found Issue Stop Training! Starting the repair '
    #                                                               'procedure.'))
    #                 self.log_file.flush()
    #                 # self.log_file.close()
    #                 # self.model.stop_training = True
    #         else:
    #             if not self.issue_list:
    #                 self.log_file.write('{},{},{},{,{}\n'.format(self.checktype, batch, self.issue_list,
    #                                                               str(time.time() - self.start_time), 'No Issue now'))
    #                 self.log_file.flush()
    #             elif self.issue_list == ['need_train']:
    #                 self.issue_list = list(set(self.issue_list))
    #                 self.log_file.write('{},{},{},{},{}\n'.format(self.checktype, batch, self.issue_list,
    #                                                               str(time.time() - self.start_time),
    #                                                               'NO training problems now. You need to train this '
    #                                                               'new model more times.'))
    #                 self.log_file.flush()
    #                 self.log_file.write('-------------------------------\n')
    #                 self.log_file.flush()
    #                 # print('NO training problems now. You need to train this new model more iterations.')
    #             else:
    #                 self.issue_list = list(set(self.issue_list))
    #                 self.log_file.write('{},{},{},{},{}\n'.format(self.checktype, batch, self.issue_list,
    #                                                               str(time.time() - self.start_time),
    #                                                               'Found Issue Stop Training! Starting the repair '
    #                                                               'procedure.'))
    #                 self.log_file.flush()
    #                 self.log_file.write('-------------------------------\n')
    #                 self.log_file.flush()
    #                 # self.model.stop_training = True

    def on_batch_end(self, batch, logs=None):
        if logs is None:
            logs = {}
        self.history['loss'].append(logs.get('loss'))
        self.history['acc'].append(logs.get('accuracy'))
        self.history['val_loss'].append(logs.get('val_loss'))
        self.history['val_acc'].append(logs.get('val_accuracy'))

        if self.checktype == 'batch' and batch % self.checkgap == 0:
            # 获取一个批次的训练数据
            training_example = self.trainX[0: self.batch_size]
            training_y = self.trainy[0: self.batch_size]

            # 将数据转换为张量，并确保形状正确
            x = tf.convert_to_tensor(training_example)
            y = tf.convert_to_tensor(training_y)
            sample_weight = None  # 如果没有样本权重，可以设置为 None

            # 调用计算梯度的函数 self.f
            evaluated_gradients = self.f(x, y, sample_weight, training=False)
            gradient_list = []
            for grad in evaluated_gradients:
                if isinstance(grad, tf.Tensor):
                    gradient_list.append(grad.numpy())

            self.evaluated_gradients = gradient_list

            # 后续处理逻辑
            self.issue_list = self.Monitor.determine(self.model, self.history, gradient_list, self.checkgap)
            self.feature_list = self.Monitor.get_features()

            self.evaluated_gradients = None
            dictwriter_object = csv.DictWriter(self.feature_log_file, fieldnames=self.feature_list.keys())
            dictwriter_object.writerow(self.feature_list)
            self.feature_log_file.flush()

            # 日志记录和其他逻辑（保持不变）
            if not self.retrain:
                if not self.issue_list:
                    self.log_file.write('{},{},{},{},{}\n'.format(
                        self.checktype, batch, self.issue_list,
                        str(time.time() - self.start_time), 'No Issue now'))
                    self.log_file.flush()
                elif self.issue_list == ['need_train']:
                    self.issue_list = list(set(self.issue_list))
                    self.log_file.write('{},{},{},{},{}\n'.format(
                        self.checktype, batch, self.issue_list,
                        str(time.time() - self.start_time),
                        'No training problems now. You need to train this new model more times.'))
                    self.log_file.flush()
                    print('No training problems now. You need to train this new model more iterations.')
                else:
                    self.issue_list = list(set(self.issue_list))
                    self.log_file.write('{},{},{},{},{}\n'.format(
                        self.checktype, batch, self.issue_list,
                        str(time.time() - self.start_time),
                        'Found Issue Stop Training! Starting the repair procedure.'))
                    self.log_file.flush()
                    # self.model.stop_training = True
            else:
                if not self.issue_list:
                    self.log_file.write('{},{},{},{},{}\n'.format(
                        self.checktype, batch, self.issue_list,
                        str(time.time() - self.start_time), 'No Issue now'))
                    self.log_file.flush()
                elif self.issue_list == ['need_train']:
                    self.issue_list = list(set(self.issue_list))
                    self.log_file.write('{},{},{},{},{}\n'.format(
                        self.checktype, batch, self.issue_list,
                        str(time.time() - self.start_time),
                        'No training problems now. You need to train this new model more times.'))
                    self.log_file.flush()
                    self.log_file.write('-------------------------------\n')
                    self.log_file.flush()
                else:
                    self.issue_list = list(set(self.issue_list))
                    self.log_file.write('{},{},{},{},{}\n'.format(
                        self.checktype, batch, self.issue_list,
                        str(time.time() - self.start_time),
                        'Found Issue Stop Training! Starting the repair procedure.'))
                    self.log_file.flush()
                    self.log_file.write('-------------------------------\n')
                    self.log_file.flush()

    def on_epoch_end(self, epoch, logs=None):
        if epoch==0:
            isFirstEpoch=True
        else:
            isFirstEpoch = False
        if logs is None:
            logs = {}
        self.history['loss'].append(logs.get('loss'))
        self.history['acc'].append(logs.get('accuracy'))
        self.history['val_loss'].append(logs.get('val_loss'))
        self.history['val_acc'].append(logs.get('val_accuracy'))

        if epoch % self.checkgap == 0 and self.checktype == 'epoch':
            # trainingExample = self.trainX[0: self.batch_size, ]
            # trainingY = self.trainy[0:self.batch_size]
            # x, y, sample_weight = self.model._standardize_user_data(trainingExample, trainingY)
            # # output_grad = self.f(x + y + sample_weight)
            # self.evaluated_gradients = self.f([x, y, sample_weight, 0])
            # gradient_list = []
            # for i in range(len(self.evaluated_gradients)):
            #     if isinstance(self.evaluated_gradients[i], np.ndarray):
            #         gradient_list.append(self.evaluated_gradients[i])
            # 获取一个批次的训练数据
            training_example = self.trainX[0: self.batch_size]
            training_y = self.trainy[0: self.batch_size]

            # 将数据转换为张量，并确保形状正确
            x = tf.convert_to_tensor(training_example)
            y = tf.convert_to_tensor(training_y)
            sample_weight = None  # 如果没有样本权重，可以设置为 None

            # 调用计算梯度的函数 self.f
            evaluated_gradients = self.f(x, y, sample_weight, training=False)
            gradient_list = []
            for grad in evaluated_gradients:
                if isinstance(grad, tf.Tensor):
                    gradient_list.append(grad.numpy())
            self.evaluated_gradients = gradient_list
            #设置激活值
            activations = self.activation_model.predict(x,batch_size=self.batch_size)
            # print(activations)
            activations = convert_activations(activations)
            # print(activations)
            print("Start monitoring")
            self.issue_list = self.Monitor.determine(self.model, self.history, gradient_list, self.checkgap,activations,isFirstEpoch)
            self.feature_list = self.Monitor.get_features()
            self.neural_feature_list=self.Monitor.get_neural_features()
            # self.issue_list = md.filtered_issue(self.issue_list)

            self.evaluated_gradients = None

            dictwriter_object = csv.DictWriter(self.feature_log_file, fieldnames=self.feature_list.keys())
            dictwriter_object.writerow(self.feature_list)
            self.feature_log_file.flush()
            dictwriter_neural_object=csv.DictWriter(self.neural_feature_log_file, fieldnames=self.Monitor.neural_feature_name)
            # print(self.neural_feature_name)
            dictwriter_neural_object.writerows(self.neural_feature_list)
            self.neural_feature_log_file.flush()

            # with open(self.neural_feature_log_file, 'a', newline='', encoding='utf-8') as output_file:
            #     # 创建一个 DictWriter 对象
            #     dict_writer = csv.DictWriter(output_file, fieldnames=keys)
            #     # 写入列名
            #     dict_writer.writeheader()
            #     # 写入字典列表的数据
            #     dict_writer.writerows(dict_list)


            if not self.retrain:
                if not self.issue_list:
                    self.log_file.write('{},{},{},{},{}\n'.format(self.checktype, epoch, self.issue_list,
                                                                  str(time.time() - self.start_time), 'No Issue now'))
                    self.log_file.flush()
                elif self.issue_list == ['need_train']:
                    self.issue_list = list(set(self.issue_list))
                    self.log_file.write('{},{},{},{},{}\n'.format(self.checktype, epoch, self.issue_list,
                                                                  str(time.time() - self.start_time),
                                                                  'NO training problems now. You need to train this '
                                                                  'new model more times.'))
                    self.log_file.flush()
                    print('NO training problems now. You need to train this new model more iterations.')
                else:
                    self.issue_list = list(set(self.issue_list))
                    self.log_file.write('{},{},{},{},{}\n'.format(self.checktype, epoch, self.issue_list,
                                                                  str(time.time() - self.start_time),
                                                                  'Found Issue Stop Training! Starting the repair '
                                                                  'procedure.'))
                    self.log_file.flush()
                    # self.log_file.close()
                    # self.model.stop_training = True
            else:
                if not self.issue_list:
                    self.log_file.write('{},{},{},{},{}\n'.format(self.checktype, epoch, self.issue_list,
                                                                  str(time.time() - self.start_time), 'No Issue now'))
                    self.log_file.flush()
                elif self.issue_list == ['need_train']:
                    self.issue_list = list(set(self.issue_list))
                    self.log_file.write('{},{},{},{},{}\n'.format(self.checktype, epoch, self.issue_list,
                                                                  str(time.time() - self.start_time),
                                                                  'NO training problems now. You need to train this '
                                                                  'new model more times.'))
                    self.log_file.flush()
                    self.log_file.write('-------------------------------\n')
                    self.log_file.flush()
                    # print('NO training problems now. You need to train this new model more iterations.')
                else:
                    self.issue_list = list(set(self.issue_list))
                    self.log_file.write('{},{},{},{},{}\n'.format(self.checktype, epoch, self.issue_list,
                                                                  str(time.time() - self.start_time),
                                                                  'Found Issue Stop Training! Starting the repair '
                                                                  'procedure.'))
                    self.log_file.flush()
                    self.log_file.write('-------------------------------\n')
                    self.log_file.flush()
                    self.model.stop_training = False

            #add code generated by gpt
                # 获取一个批次的数据
                # x = self.trainX[0:self.batch_size]

                # 计算激活值
            # 遍历每一层，计算每个神经元的激活值均值和方差
            # for layer_index, activation in enumerate(activations):
            #     activation_flat = activation.reshape(activation.shape[0], -1)
            #     neuron_means = np.mean(activation_flat, axis=0)
            #     neuron_vars = np.var(activation_flat, axis=0)
            #     num_neurons = neuron_means.shape[0]
            #     for neuron_index in range(num_neurons):
            #         mean = neuron_means[neuron_index]
            #         var = neuron_vars[neuron_index]
            #         self.activation_log_file.write(
            #             '{},{},{},{},{}\n'.format(epoch, layer_index, neuron_index, mean, var))
            # self.activation_log_file.flush()

    def on_train_end(self, logs=None):
        if self.retrain and self.issue_list == []:
            self.log_file.write('------------Solved!-----------\n')
            self.log_file.flush()

        solution_dir = os.path.join(self.save_dir, 'solution')
        if self.retrain:
            solution_dir = self.pkl_dir
        if not os.path.exists(solution_dir):
            os.makedirs(solution_dir)
        issue_path = os.path.join(solution_dir, 'issue_history.pkl')
        tmpset = {'issue_list': self.issue_list, 'history': self.history, 'feature': self.feature_list}
        with open(issue_path, 'wb') as f:
            pickle.dump(tmpset, f)
        self.log_file.close()
        self.feature_log_file.close()
        # self.activation_log_file.close()
        print('Finished Training')

    # def check_activation_layers(self):
    #     for idx, layer in enumerate(self.model.layers):
    #         # 检查层是否为 Activation 层
    #         if isinstance(layer, Activation):
    #             if idx > 0 and isinstance(self.model.layers[idx - 1], Dense):
    #                 activation_name = layer.activation.__name__
    #                 self.activate_list.append(activation_name)
    #                 self.activate_dict[idx]=activation_name
    #         # 检查层是否具有 'activation' 属性（如 Dense 层）
    #         elif isinstance(layer, Dense) and hasattr(layer, 'activation') and layer.activation is not None:
    #             activation = layer.activation
    #             # 获取激活函数的名称
    #             if isinstance(activation, str):
    #                 activation_name = activation
    #             else:
    #                 activation_name = activation.__name__
    #             # 过滤掉 'linear' 激活函数
    #             if activation_name != 'linear':
    #                 self.activate_list.append(activation_name)
    #                 self.activate_dict[idx] = activation_name
    def check_activation_layers(self):
        num_layers = len(self.model.layers)
        idx = 0
        while idx < num_layers:
            layer = self.model.layers[idx]
            # print(f"Layer Name: {layer.name}, Layer Type: {type(layer).__name__}")
            # 如果层是 Dense 层
            if isinstance(layer, Dense) or type(layer).__name__ == 'Dense':
                target_idx=idx
                activation_name = None
                # 检查 Dense 层的 activation 属性
                if hasattr(layer, 'activation') and layer.activation is not None:
                    if isinstance(layer.activation, str):
                        activation_name = layer.activation
                    else:
                        activation_name = layer.activation.__name__
                    # if activation_name == 'linear':
                    #     activation_name = None  # 视为没有激活函数
                # 如果 Dense 层的 activation 为 None 或 'linear'，向后搜索 Activation 层
                if activation_name is None or activation_name=='linear':
                    forward_idx = idx + 1
                    while forward_idx < num_layers:
                        next_layer = self.model.layers[forward_idx]
                        if isinstance(next_layer, Activation):
                            target_idx=forward_idx
                            activation_name = next_layer.activation.__name__
                            break
                        elif isinstance(next_layer,LeakyReLU):
                            target_idx = forward_idx
                            activation_name = "LeakyReLu"
                            break
                        elif isinstance(next_layer, Dense):
                            # 在下一个 Dense 层之前未找到 Activation 层，停止搜索
                            break
                        else:
                            forward_idx += 1
                    # 如果未找到 Activation 层，activation_name 保持为 None
                # 如果找到激活函数，执行操作

                if activation_name is not None:
                    self.activate_list.append(activation_name)
                    self.activate_dict[target_idx] = activation_name
            idx += 1





