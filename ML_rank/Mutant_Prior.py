import pandas as pd
import numpy as np
from ML_rank import Config
import xgboost as xgb
import os
import pickle
class MutantPrior:
    def __init__(self, select_ratio,file_path):
        self.select_ratio=select_ratio
        self.file_path=file_path
        self.selected_mutant_name = set()
        self.constantColumns=Config.constantColumns
        self.layer_mutant_dict = dict()


    def process(self):
        data = pd.read_csv(self.file_path)
        data = self.fix_feat_loss_func(data)
        df=self.dataPreprocess(data)
        X_test, mut_info_df = self.split_target(df)
        dtest = xgb.DMatrix(X_test)
        #加载XGBoost模型
        current_dir = os.path.dirname(os.path.abspath(__file__))
        best_model_dir = os.path.join(current_dir,'best_model', 'best_model_rank', 'xgb_30.pkl')
        with open(best_model_dir, 'rb') as f:
            model = pickle.load(f)
        y_pred = model.predict(dtest)
        res_df = self.concat_df(y_pred, mut_info_df)
        res_df = (res_df.groupby("layer_id", group_keys=False)
                  .apply(self.select_rows)
                  .reset_index(drop=True)
                  )
        self.df_to_mutant_set(res_df)

    def dataPreprocess(self,data):
        #缺失值处理
        df_filled = data.fillna(0)
        #异常值处理
        df_filled = self.deal_abnormal_value(df_filled)
        #去除常数列
        df_filled = df_filled.drop(columns=self.constantColumns, errors='ignore')
        return df_filled

    '''
    自定义排序规则
    '''
    def select_rows(self,group):
        n = len(group)
        g_sorted = group.sort_values('y_pred', ascending=False)
        # 情况 1：行数 <= 80，选全部
        if n <= 80:
            return g_sorted

        # 情况 2：行数 > 80，先按 y_pred 排序
        # 计算前 k% 数量
        top_count = int(np.ceil(n * self.select_ratio))

        # 至少要取 80 行：max(前 25% 行数, 80)
        need = max(top_count, 80)
        return g_sorted.head(need)

    def fix_feat_loss_func(self,data):
        value_to_find = 'ListWrapper([<function mean_squared_error at 0x0000023F51871310>])'
        value_to_find2 = 'CategoricalCrossentropy'
        value_to_find3 = 'MeanSquaredError'
        value_to_find4 = 'BinaryCrossentropy'
        data.loc[data["loss_func"] == value_to_find, "loss_func"] = 'mean_squared_error'
        data.loc[data["loss_func"] == value_to_find2, "loss_func"] = 'categorical_crossentropy'
        data.loc[data["loss_func"] == value_to_find3, "loss_func"] = 'mean_squared_error'
        data.loc[data["loss_func"] == value_to_find4, "loss_func"] = 'mean_squared_error'
        return data

    def deal_abnormal_value(self,df):
        float32_max = np.finfo(np.float32).max
        float32_min = np.finfo(np.float32).min
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        # 使用 clip 将数值限定在 float32 范围
        df = df.replace([np.inf, -np.inf], [float32_max, float32_min])  # 替换正负 inf
        df[numeric_cols] = df[numeric_cols].clip(lower=float32_min, upper=float32_max)
        return df

    def split_target(self,df):
        mut_info_df = df[['model_name', 'layer_id', 'neuron_idx', 'mutant_oper']]
        df = pd.get_dummies(df)
        X = df.drop(columns=['model_name'])
        return X, mut_info_df

    def concat_df(self,y_pred, mut_info_df):
        arr_df = pd.DataFrame({'y_pred': y_pred})
        result = pd.concat([mut_info_df, arr_df], axis=1)
        return result

    def df_to_mutant_set(self, df):
        for idx, row in df.iterrows():
            layer_str = str(row['layer_id'])
            neuron_str = str(row['neuron_idx'])
            mutant_oper_str = row['mutant_oper']
            mutant_name_str = 'L' + layer_str + '-N' + neuron_str + '-' + mutant_oper_str + 'model.h5'
            layer_id = 'L' + layer_str
            self.selected_mutant_name.add(mutant_name_str)
            if layer_id not in self.layer_mutant_dict:
                self.layer_mutant_dict[layer_id] = []
            self.layer_mutant_dict[layer_id].append(mutant_name_str)

    def getMutantSet(self):
        return self.selected_mutant_name

    def getLayerMutantDict(self):
        return self.layer_mutant_dict



if __name__ == "__main__":
    mutant_ratio = 1.0
    base_dir = 'D:\\DeepMPrior\\Dataset\\all-bugs\\44758894'
    mp_xgb = MutantPrior(select_ratio = mutant_ratio, file_path = os.path.join(base_dir, "all_summary.csv"))
    mp_xgb.process()
    layer_mutant_dict = mp_xgb.getLayerMutantDict()
    # print(layer_mutant_dict)
    mt_set = mp_xgb.getMutantSet()
    print(len(mt_set))






