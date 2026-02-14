import json
import os.path
from collections import defaultdict
import numpy as np
from keras.layers import Dense, SimpleRNN

def float32_to_float64(obj):
    if isinstance(obj, np.float32):
        return float(obj)
    raise TypeError("Type not serializable")
class Mutation_info:
    def __init__(self,weights,dir):
        self.initial_weight=weights
        self.base_dir=dir
        self.mut_feat_dict=defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        self.layer_index_dict={}
    def store_mutation_info(self,layer_idx,neural_idx,mut_model_weight,mut_oper_type):
        layer_pos=self.layer_index_dict[layer_idx]
        mut_weight=mut_model_weight[layer_pos*2][:,neural_idx]
        mut_bias=mut_model_weight[layer_pos*2+1][neural_idx]
        origin_weight=self.initial_weight[layer_pos*2][:,neural_idx]
        origin_bias=self.initial_weight[layer_pos*2+1][neural_idx]
        # origin_layer=self.initial_model.layers[layer_idx]
        # mut_layer=mut_model.layers[layer_idx]
        # mutWeights,mutBias=mut_layer.get_weights()
        # originWeights,originBias=origin_layer.get_weights()
        # origin_weight=originWeights[:,neural_idx]
        # mut_weight=mutWeights[:,neural_idx]
        # origin_bias=originBias[neural_idx]
        # mut_bias=mutBias[neural_idx]
        weight_change=mut_weight-origin_weight
        self.mut_feat_dict[layer_idx][neural_idx][mut_oper_type]['mut_avg_weight']=np.mean(mut_weight)
        self.mut_feat_dict[layer_idx][neural_idx][mut_oper_type]['mut_avg_abs_weight'] = np.mean(np.abs(mut_weight))
        self.mut_feat_dict[layer_idx][neural_idx][mut_oper_type]['mut_l2_norm_weight'] = np.linalg.norm(mut_weight)/mut_weight.size
        self.mut_feat_dict[layer_idx][neural_idx][mut_oper_type]['mut_abs_max_weight'] = np.max(np.abs(mut_weight))
        self.mut_feat_dict[layer_idx][neural_idx][mut_oper_type]['mut_max_weight'] = np.max(mut_weight)
        self.mut_feat_dict[layer_idx][neural_idx][mut_oper_type]['mut_min_weight'] = np.min(mut_weight)
        self.mut_feat_dict[layer_idx][neural_idx][mut_oper_type]['mut_max_weight_change'] = np.max(weight_change)
        self.mut_feat_dict[layer_idx][neural_idx][mut_oper_type]['mut_avg_weight_change'] = np.mean(weight_change)
        self.mut_feat_dict[layer_idx][neural_idx][mut_oper_type]['mut_l2_norm_weight_change'] = np.linalg.norm(weight_change)/weight_change.size
        self.mut_feat_dict[layer_idx][neural_idx][mut_oper_type]['mut_bias']=mut_bias
        self.mut_feat_dict[layer_idx][neural_idx][mut_oper_type]['mut_relat_weight_change']=np.linalg.norm(weight_change)/(np.linalg.norm(origin_weight)+0.00001)
        self.mut_feat_dict[layer_idx][neural_idx][mut_oper_type]['mut_bias_change']=mut_bias-origin_bias
        self.mut_feat_dict[layer_idx][neural_idx][mut_oper_type]['mmt_relat_bias_change'] = abs(mut_bias - origin_bias)/(abs(origin_bias)+0.00001)
    def store_file(self):
        mut_dict_filename = os.path.join(self.base_dir, "result_dir", "mut_dict.json")
        with open(mut_dict_filename, 'w') as fw:
            json.dump(self.mut_feat_dict, fw,default=float32_to_float64)

    def set_index_dict(self,model):
        index=0
        for idx, layer in enumerate(model.layers):
            if isinstance(layer, Dense) or isinstance(layer, SimpleRNN):  # 判断该层是否是 Dense 层
                # 使用层的索引作为字典的键，层名称作为值
                self.layer_index_dict[idx] =index
                index+=1
    def get_mut_feat_dict(self):
        return self.mut_feat_dict
