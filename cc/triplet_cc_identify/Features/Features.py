import math

import numpy as np
import pandas as pd

from cc.triplet_cc_identify.FailingTestsHandler import FailingTestsHandler
from fl_evaluation.metrics.calc_corr import calc_corr
from CONFIG import *
from read_data.Defects4JDataLoader import Defects4JDataLoader


class Features:

    def __init__(self, data_df):
        self.data_df = data_df
        self.suspicious_list = calc_corr(self.data_df, "ochiai")
        self.passing_features = self.get_passing_tests(self.data_df).iloc[:, :-1]
        self.failing_features = FailingTestsHandler.get_failing_tests(self.data_df).iloc[:, :-1]
        self.ssp = dict()
        self.cr = dict()
        self.sf = dict()

    def getAllFeatures(self):
        self.suspScore()
        self.covRatio()
        self.similarityFactor()

        ssp_list = np.array(list(self.ssp.values())).reshape(-1, 1)
        cr_list = np.array(list(self.cr.values())).reshape(-1, 1)
        sf_list = np.array(list(self.sf.values())).reshape(-1, 1)
        comb_list = np.concatenate((ssp_list, cr_list, sf_list), axis=1)
        featuresDf = pd.DataFrame(comb_list, index=list(self.ssp.keys()))
        return featuresDf

    def get_passing_tests(self, data_df):
        return data_df[data_df["error"] == 0]

    def suspScore(self):
        for row_index, row in self.passing_features.iterrows():
            h_cnt = 0
            h_ssum = 0
            l_cnt = 0
            l_ssum = 0
            for line_num, s in self.suspicious_list.items():
                if row.loc[line_num] == 1 and 0.5 <= s <= 1.0:
                    h_cnt = h_cnt + 1
                    h_ssum = h_ssum + s
                elif row.loc[line_num] == 1 and 0.5 > s >= 0:
                    l_cnt = l_cnt + 1
                    l_ssum = l_ssum + s
            if h_cnt == 0 and l_cnt != 0:
                self.ssp[row_index] = l_ssum / l_cnt
            elif h_cnt != 0:
                self.ssp[row_index] = h_ssum / h_cnt
            else:
                self.ssp[row_index] = 0

    def covRatio(self):
        S_cnt = len(self.suspicious_list)
        for row_index, row in self.passing_features.iterrows():
            h_cnt = 0
            for line_num, s in self.suspicious_list.items():
                if row.loc[line_num] == 1 and 0.5 <= s <= 1.0:
                    h_cnt = h_cnt + 1
            self.cr[row_index] = h_cnt / S_cnt

    def similarityFactor(self):
        suspicious_list = pd.Series(self.suspicious_list)
        for row_index, passing in self.passing_features.iterrows():
            min_dis = 1e+5
            passing_vector = passing * suspicious_list
            for _, failing in self.failing_features.iterrows():
                failing_vector = failing * suspicious_list
                dis = np.sqrt(np.sum((passing_vector - failing_vector) ** 2))
                if dis < min_dis:
                    if math.fabs(dis) <= 1e-8:  # make sure the value is not inf
                        min_dis = 1e-8
                        break
                    min_dis = dis

            self.sf[row_index] = 1 / min_dis

    def faultMaskingFactor(self):
        pass


if __name__ == "__main__":
    data = Defects4JDataLoader(os.path.join(project_dir, 'data'), "Chart", "1")
    data.load()
    features = Features(data.data_df)
    features.getAllFeatures()
