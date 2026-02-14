class RuntimeCalculator:
    def __init__(self,model_name:int):
        self.run_10=0
        self.run_15=0
        self.run_20=0
        self.run_25=0
        self.run_30=0
        self.run_35=0
        self.cnt_10=0
        self.cnt_15=0
        self.cnt_20=0
        self.cnt_25=0
        self.cnt_30=0
        self.cnt_35=0
        self.mut_dic={}
        self.model_name=model_name
        self.tot=0
        self.count=0
    def set_dict(self,dic):
        self.mut_dic=dic
    def integrate(self):
        self.run_15+=self.run_10
        self.run_20+=self.run_15
        self.run_25+=self.run_20
        self.run_30+=self.run_25
        self.run_35+=self.run_30
        self.cnt_15 += self.cnt_10
        self.cnt_20 += self.cnt_15
        self.cnt_25 += self.cnt_20
        self.cnt_30 += self.cnt_25
        self.cnt_35 += self.cnt_30

    def cal_time(self,layer_id,neural_idx,mutant_oper,dur):
        self.tot+=dur
        self.count+=1
        if not self.check_keys_exist(layer_id,neural_idx,mutant_oper):
            return
        val=self.mut_dic[self.model_name][layer_id][neural_idx][mutant_oper]
        if val==10:
            self.run_10+=dur
            self.cnt_10+=1
        elif val==15:
            self.run_15+=dur
            self.cnt_15+=1
        elif val==20:
            self.run_20+=dur
            self.cnt_20+=1
        elif val==25:
            self.run_25+=dur
            self.cnt_25+=1
        elif val==30:
            self.run_30+=dur
            self.cnt_30+=1
        elif val==35:
            self.run_35+=dur
            self.cnt_35+=1

    def to_dict(self):
        run_dict={"tot":self.tot,"run_10":self.run_10,"run_15":self.run_15,"run_20":self.run_20,"run_25":self.run_25,"run_30":self.run_30,
                  "run_35":self.run_35}
        cnt_dict={"tot":self.count,"cnt_10":self.cnt_10,"cnt_15":self.cnt_15,"cnt_20":self.cnt_20,"cnt_25":self.cnt_25,"cnt_30":self.cnt_30,
                  "cnt_35":self.cnt_35}
        return run_dict,cnt_dict

    def check_keys_exist(self, layer_id, neuron_idx, mutant_oper):
        return self.mut_dic.get(self.model_name, {}).get(layer_id, {}).get(neuron_idx, {}).get(mutant_oper)

