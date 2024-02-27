import sys
from collections import Counter
from CONFIG import *
import numpy as np
import pandas as pd

from cc.cc_baselines.BaseCCPipeline import BaseCCPipeline
from sklearn.cluster import KMeans

from cc.core import run
from utils.task_util import task_complete



class NewTechnique1CCPipeline(BaseCCPipeline):
    def __init__(self, project_dir, configs, cita, way):
        super().__init__(project_dir, configs, way)
        self.CCT = None
        self.CCE = None
        self.cita = cita

    def _find_cc_index(self):
        self._find_CCE()
        if len(self.CCE) == 0:
            return

        # data_df = self.load_data()
        features = self.data_df[self.CCE]
        self.CCE.append("error")
        new_data_df = self.data_df[self.CCE]

        passing_array = np.array(new_data_df[new_data_df["error"] == 0])[:, :-1]
        # print(passing_array)
        cc_flags=[]
        for item in passing_array:
            if np.any(item):
                cc_flags.append(1)
            else:
                cc_flags.append(0)

        # nailed it !
        self.cc_index = pd.Series(cc_flags, index=self.ground_truth_cc_index.index)
        # print(self.cc_index)


    def getfT(self, data):
        uncover = sum(data == 0)
        cover = sum(data == 1)
        fT = cover / (uncover + cover)
        return fT

    def getpT(self, data):
        uncover = sum(data == 0)
        cover = sum(data == 1)
        pT = cover / (uncover + cover)
        return pT

    def _is_CCE(self, fail_data, pass_data):
        fT = self.getfT(fail_data)
        pT = self.getpT(pass_data)
        if ((fT == 1.0) and (pT < self.cita)):
            return True
        else:
            return False

    def _find_CCE(self):
        data = self.dataloader
        data_df = data.data_df
        failing_df = data_df[data_df["error"] == 1]
        passing_df = data_df[data_df["error"] == 0]
        CCE = []
        for i in failing_df.columns:
            if i != "error":
                if self._is_CCE(failing_df[i], passing_df[i]):
                    CCE.append(i)
        # print(CCE)
        # cct=[]
        self.CCE = CCE
        # new_CCE=self.get_multi_columns_by_index(data_df, CCE)
        # self.CCE = new_CCE
        # #找CCT，一个一个测试
        # for t in data_df.index:
        #     if(sum(data_df.iloc[t,new_CCE]==1)>=1):
        #         cct.append(t)
        # self.CCT=cct

    def get_multi_columns_by_index(self, dataset, columns_list):
        datas_with_columns = []
        for i in range(len(columns_list)):
            data = dataset.columns.get_loc(columns_list[i])
            datas_with_columns.append(data)
        return datas_with_columns


def main():
    program_list = [
        "Chart",
        # "Closure-2023-12-6-1",
        # "Lang",
        # "Math",
        # "Mockito",
        # "Time"
    ]
    run(program_list, "Chart", 1, NewTechnique1CCPipeline, "2022-9-27-Tech-I", 1)



if __name__ == "__main__":
    # main()
    # task_complete("Tech-I end")
    program_list = [
        # "Chart",
        # "Closure-2023-12-6-1",
        # "Lang",
        "Math",
        "Mockito",
        # "Time"
    ]
    for program in program_list:
        for i in cc_info[program]:
            configs = {'-d': 'd4j', '-p': program, '-i': i, '-m': method_para, '-e': 'origin'}
            sys.argv = os.path.basename(__file__)
            ccpl = NewTechnique1CCPipeline(project_dir, configs, 0.7, "2022-09-29-New-Tech-I")
            ccpl.find_cc_index()
            ccpl.evaluation()
            ccpl.calRes("trim")
    # a = 1
