import math

import numpy as np
import pandas as pd

from cc.triplet_cc_identify.FailingTestsHandler import FailingTestsHandler
# from cc.triplet_cc_identify.PassingTestsHandler import PassingTestsHandler
from fl_evaluation.metrics.calc_corr import calc_corr
from CONFIG import *
from fl_evaluation.metrics.metrics2 import *
from fl_evaluation.metrics.new_calc_corr import new_calc_corr
from read_data.Defects4JDataLoader import Defects4JDataLoader


class Features:
    def __init__(self, data_df):
        self.data_df = data_df
        self.suspicious_list = [ new_calc_corr(self.data_df, DStar()), new_calc_corr(self.data_df,Ochiai()),
                                 new_calc_corr(self.data_df,Barinel()),new_calc_corr(self.data_df,ER1()),
                                 new_calc_corr(self.data_df,ER5()), new_calc_corr(self.data_df,GP02()),
                                 new_calc_corr(self.data_df,GP03()), new_calc_corr(self.data_df,GP19()),
                                 new_calc_corr(self.data_df,Jaccard()),new_calc_corr(self.data_df,Op2())]
        self.statements = self.data_df.iloc[:, :-1]
        self.passing_features = self.get_passing_tests(self.data_df).iloc[:, :-1]
        self.failing_features = FailingTestsHandler.get_failing_tests(self.data_df).iloc[:, :-1]

        # 假设data_df是你的原始DataFrame，dict_list是你的包含10个字典的列表
        columns = self.statements.columns.tolist()
        # 创建一个新的DataFrame，行数为10（dict_list的长度），列数为7058（每个字典的长度）
        self.suspicious_df = pd.DataFrame(index=range(len(self.suspicious_list)), columns=columns)
        # 将字典的值添加到新DataFrame中
        for i, d in enumerate(self.suspicious_list):
            self.suspicious_df.loc[i] = d

    def get_suspicious_df(self):
        return self.suspicious_df

    def get_augmentation_data(self):
        print(self.statements)
        print(self.suspicious_df)



    def get_passing_tests(self, data_df):
        return data_df[data_df["error"] == 0]



if __name__ == "__main__":
    data = Defects4JDataLoader(os.path.join(project_dir, '..', 'data'), "Chart", "1")
    data.load()
    features = Features(data.data_df)
    # features.covRatio()
    # features.similarityFactor()
    a = 1
