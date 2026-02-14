from collections import defaultdict

import numpy as np
class ActivInfo:
    def __init__(self,activation,layer_idx):
        self.activation=activation
        self.layer_idx=layer_idx
        self.test_activ_feat_dict={}

    def cal_dict(self):
        print(self.layer_idx)
        self.test_activ_feat_dict= defaultdict(lambda: defaultdict(dict))

        # print(self.activation)
        # print(len(self.activation))
        for layer_id in range(len(self.activation)):
            layer_num=self.layer_idx[layer_id+1]
            sz=len(self.activation[layer_id])
            print(layer_id,sz)
            for neuron_idx in range(sz):
                activ = self.activation[layer_id][neuron_idx]
                if activ.size==0:
                    print("!!!!!!!!!!!!!!!!!!!!!!!!!!")
                self.test_activ_feat_dict[layer_num][neuron_idx]['test_avg_activ'] = np.mean(activ)
                self.test_activ_feat_dict[layer_num][neuron_idx]['test_max_activ'] = np.max(activ)
                self.test_activ_feat_dict[layer_num][neuron_idx]['test_min_activ'] = np.min(activ)
                self.test_activ_feat_dict[layer_num][neuron_idx]['test_std_activ'] = np.std(activ)
                self.test_activ_feat_dict[layer_num][neuron_idx]['test_var_activ'] = np.var(activ)
                self.test_activ_feat_dict[layer_num][neuron_idx]['test_median_activ'] = np.median(activ)
                self.test_activ_feat_dict[layer_num][neuron_idx]['test_activ_zero_ratio'] = np.sum(activ == 0) / activ.size
                self.test_activ_feat_dict[layer_num][neuron_idx]['test_activ_l2_norm']=np.linalg.norm(activ) / activ.size

    def get_dict(self):
        return self.test_activ_feat_dict




