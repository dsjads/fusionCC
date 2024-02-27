import copy
import sys
import scipy
import numpy as np
import pandas as pd
from pyclustering.utils import type_metric, distance_metric
import time

from cc.ReadData import ReadData
from CONFIG import *
from cc.cc_baselines.BaseCCPipeline import BaseCCPipeline
from cc.core import run
from utils.task_util import task_complete
from utils.write_util import write_rank_to_txt
from cc.cc_baselines import Technique1CCPipeline

from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.cluster.kmeans import kmeans


class Technique2CCPipeline(BaseCCPipeline):
    def __init__(self, project_dir, configs, _, way):
        super().__init__(project_dir, configs, way)
        self.fCCE = []

    def get_fCCE(self, data_df):
        columns = data_df.columns.values.tolist()
        # column为各个element的索引
        for column in columns[:-1]:
            # 如果
            if (data_df[data_df["error"] == 1][column] == 1).all():
                p_e = sum(data_df[data_df["error"] == 0][column] == 1) / len(data_df[data_df["error"] == 0])
                self.fCCE.append(u(p_e))
            else:
                self.fCCE.append(float(0))

    # def find_cc_index(self):
    #     data_df = self.data_obj.data_df
    #     if len(data_df[data_df["error"] == 0]) == 0:
    #         record = dict()
    #         record["msg"] = "No passing tests"
    #         save_path = os.path.join(self.project_dir, "new_results", self.way, "record.txt")
    #         write_rank_to_txt(record, save_path, self.program, self.bug_id)
    #         return
    #
    #     self._find_cc_index()

    def _find_cc_index(self):
        self.get_fCCE(self.data_df)
        Tcc = pd.DataFrame()
        Tp = self.load_data()
        Tp_copy0 = copy.deepcopy(Tp)
        Tp = Tp.iloc[:, :-1]
        Tp_copy = copy.deepcopy(Tp)
        Tp_len = len(Tp)
        dis = scipy.spatial.distance.cdist(Tp.iloc[:, :], Tp.iloc[:, :], metric='euclidean')
        row_select = list(pd.DataFrame(dis).stack().idxmax())
        initial_centers = kmeans_plusplus_initializer(Tp.iloc[row_select, :], 2).initialize()
        metric = distance_metric(type_metric.USER_DEFINED, func=self.CCE_distance)
        kmeans_instance = kmeans(Tp, initial_centers, metric=metric)
        kmeans_instance.process()
        clusters = kmeans_instance.get_clusters()
        cluster1 = clusters[0]
        try:
            cluster2 = clusters[1]
        except:
            return
        R0 = max(self.get_Relevance(cluster1, Tp_copy), self.get_Relevance(cluster2, Tp_copy))
        R = R0
        while R / R0 >= 0.75:
            selected = cluster1 if self.get_Relevance(cluster1, Tp_copy) > self.get_Relevance(cluster2,
                                                                                              Tp_copy) else cluster2
            Tcc = pd.concat([Tcc, Tp.iloc[selected, :]])
            Tp_index = Tp.index.values.tolist()
            new_Tp_index = list(set(Tp_index).difference(set(selected)))
            Tp = Tp.loc[new_Tp_index]
            dis = scipy.spatial.distance.cdist(Tp.iloc[:, :], Tp.iloc[:, :], metric='euclidean')
            row_select = list(pd.DataFrame(dis).stack().idxmax())
            initial_centers = kmeans_plusplus_initializer(Tp.iloc[row_select, :], 2).initialize()
            metric = distance_metric(type_metric.USER_DEFINED, func=self.CCE_distance)
            kmeans_instance = kmeans(Tp, initial_centers, metric=metric)
            kmeans_instance.process()
            clusters = kmeans_instance.get_clusters()
            if (len(clusters) <= 1):
                break
            cluster1 = clusters[0]
            cluster2 = clusters[1]
            R = max(self.get_Relevance(cluster1, Tp_copy), self.get_Relevance(cluster2, Tp_copy))
        Tcc_index = Tcc.index.values.tolist()
        Tcc_Bool = pd.DataFrame([False for i in range(Tp_len)])
        Tcc_Bool.loc[Tcc_index, :] = True
        passing_df = Tp_copy0[Tp_copy0["error"] == 0]
        passing_index = passing_df.index.values.tolist()
        final_Tcc = Tcc_Bool.loc[passing_index].values.T[0]
        self.cc_index = pd.Series(final_Tcc, index=self.ground_truth_cc_index.index)

    def get_Relevance(self, cluster, data):
        relevance = 0
        for clu in cluster:
            clu_list = list(data.loc[clu])
            for i in range(len(clu_list)):
                relevance += clu_list[i] * self.fCCE[i]
        return relevance

    # @staticmethod
    # def f_i_e(self,i, e):
    #     data=self.load_data()
    #     return data[i][e]

    def CCE_distance(self, a, b):
        distance = np.sqrt(sum(pd.Series(self.fCCE) * (a - b)) ** 2)
        return distance


def u(p_e):
    if p_e < 0.56:
        return p_e / 0.56
    else:
        return (1 / 0.44) - (p_e / 0.44)


# def main():
#     for program, value in all_info.items():
#         for i in value:
#             print(program, i)
#             configs = {'-d': 'd4j', '-p': program, '-i': i, '-m': method_para, '-e': 'origin'}
#             sys.argv = os.path.basename(__file__)
#             svmccpl = Technique2CCPipeline(project_dir, configs, "2022-7-27-Tech-II")
#             svmccpl.find_cc_index()
#             svmccpl.evaluation()
#             svmccpl.calRes()

def main():
    program_list = [
        "Chart"
    ]
    run(program_list, "Chart", 1, Technique2CCPipeline, "2024-Tech-II", 1)

if __name__ == "__main__":
    main()
    task_complete("Tech-II end")
    # configs = {'-d': 'd4j', '-p': "Chart", '-i': 14, '-m': method_para, '-e': 'origin'}
    # sys.argv = os.path.basename(__file__)
    # svmccpl = Technique2CCPipeline(project_dir, configs, "2022-7-27-Tech-II")
    # svmccpl.find_cc_index()
    # svmccpl.evaluation()
    # svmccpl.calRes()
