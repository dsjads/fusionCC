import sys
from collections import Counter
from CONFIG import *
import numpy as np
import pandas as pd
from cc.triplet_cc_identify.FailingTestsHandler import FailingTestsHandler
from fl_evaluation.metrics.calc_corr import calc_corr

from cc.cc_baselines.BaseCCPipeline import BaseCCPipeline
from sklearn.cluster import KMeans

from cc.core import run
from utils.task_util import task_complete



class NewTechnique2CCPipeline(BaseCCPipeline):
    def __init__(self, project_dir, configs, cita, way):
        super().__init__(project_dir, configs, way)
        self.suspicious_list = [calc_corr(self.data_df, "ochiai")]
        self.CCT = None
        self.CCE = None
        self.ssp=dict()
        self.cr=dict()
        self.cita = cita
        self.passing_features = self.get_passing_tests(self.data_df).iloc[:, :-1]
        self.failing_features = FailingTestsHandler.get_failing_tests(self.data_df).iloc[:, :-1]

    def get_passing_tests(self, data_df):
        return data_df[data_df["error"] == 0]

    def suspScore(self):
        for row_index, row in self.passing_features.iterrows():
            # total_ssp = 0
            h_cnt = 0
            h_ssum = 0
            l_cnt = 0
            l_ssum = 0
            self.ssp[row_index]=[]
            for item in self.suspicious_list:
                for line_num, s in item.items():
                    if row.loc[line_num] == 1 and 0.5 <= s <= 1.0:
                        h_cnt = h_cnt + 1
                        h_ssum = h_ssum + s
                    elif row.loc[line_num] == 1 and 0.5 > s >= 0:
                        l_cnt = l_cnt + 1
                        l_ssum = l_ssum + s
                if h_cnt == 0 and l_cnt != 0:
                    self.ssp[row_index].append(l_ssum / l_cnt)
                elif h_cnt != 0:
                    self.ssp[row_index].append(h_ssum / h_cnt)
                else:
                    self.ssp[row_index].append(0)
        ssp_list = np.array(list(self.ssp.values()))
        return ssp_list

    def covRatio(self):
        S_cnt = len(self.suspicious_list)
        for row_index, row in self.passing_features.iterrows():
            h_cnt = 0
            self.cr[row_index] = []
            for item in self.suspicious_list:
                for line_num, s in item.items():
                    if row.loc[line_num] == 1 and 0.5 <= s <= 1.0:
                        h_cnt = h_cnt + 1
                self.cr[row_index].append(h_cnt / S_cnt)
        cr_list = np.array(list(self.cr.values()))
        return cr_list

    def _find_cc_index(self):
        self._find_CCE()
        if len(self.CCE) == 0:
            return

        # data_df = self.load_data()
        features = self.passing_features[self.CCE]
        self.CCE.append("error")
        new_data_df = self.data_df[self.CCE]
        self.data_df=features
        self.ssp=self.suspScore()
        self.cr=self.covRatio()
        self.ssp.reshape(-1)
        self.cr.reshape(-1)
        score=[]
        for i in range(len(self.ssp)):
            score.append(self.ssp[i][0]+self.cr[i][0])
        score_list=pd.Series(score,index=self.ground_truth_cc_index.index)
        rank_list=score_list.sort_values(ascending=False).index

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
        self.cc_index.loc[rank_list[round(0.7*len(rank_list)):]]=0
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
    run(program_list, "Chart", 1, NewTechnique2CCPipeline, "2022-9-27-Tech-II", 1)



if __name__ == "__main__":
    # main()
    # task_complete("Tech-I end")
    program_list = [
        "Chart",
        # "Closure-2023-12-6-1",
        "Lang",
        "Math",
        "Mockito",
        "Time"
    ]
    for program in program_list:
        for i in cc_info[program]:
            configs = {'-d': 'd4j', '-p': program, '-i': i, '-m': method_para, '-e': 'origin'}
            sys.argv = os.path.basename(__file__)
            ccpl = NewTechnique2CCPipeline(project_dir, configs, 0.7, "2022-09-29-New-Tech-II")
            ccpl.find_cc_index()
            ccpl.evaluation()
            ccpl.calRes("trim")
    a = 1
    # configs = {'-d': 'd4j', '-p': "Lang", '-i': 1, '-m': method_para, '-e': 'origin'}
    # sys.argv = os.path.basename(__file__)
    # ccpl = NewTechnique2CCPipeline(project_dir, configs, 0.7, "2022-09-29-New-Tech-II")
    # ccpl.find_cc_index()