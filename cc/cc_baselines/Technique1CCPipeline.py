from collections import Counter

import numpy as np
import pandas as pd

from cc.cc_baselines.BaseCCPipeline import BaseCCPipeline
from sklearn.cluster import KMeans

from cc.core import run
from utils.task_util import task_complete


class Technique1CCPipeline(BaseCCPipeline):
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
        # init
        # search for the initial centers
        # select a failing point
        failing_df = new_data_df[new_data_df["error"] == 1]
        failing_size = len(failing_df)
        selected_failing_index = np.random.randint(0, failing_size)
        selected_failing_array = np.array(failing_df.iloc[selected_failing_index])[:-1]

        # select the most distant passing point
        passing_array = np.array(new_data_df[new_data_df["error"] == 0])[:, :-1]
        dis = np.sum((passing_array - selected_failing_array) ** 2, axis=1)
        min_dis_index = np.argmax(dis)
        selected_passing_array = passing_array[min_dis_index]

        # 初始化两个中心点，一开始的中心点设置为failing point和离它最远的passing point
        init_centers = np.vstack([selected_failing_array, selected_passing_array])

        kmeans = KMeans(init=init_centers, n_clusters=2, n_init=1)
        kmeans.fit(features)

        # find the cluster of most failing tests
        cluster_labels = kmeans.labels_
        cluster_failings = cluster_labels[new_data_df["error"] == 1]
        votes = Counter(cluster_failings)
        most_vote = votes.most_common(1)[0][0]

        # nailed it !
        cluster_passing_labels = cluster_labels[new_data_df["error"] == 0]
        cc_flags = (cluster_passing_labels == most_vote)
        self.cc_index = pd.Series(cc_flags, index=self.ground_truth_cc_index.index)


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
        "Chart"
    ]
    run(program_list, "Chart", 1, Technique1CCPipeline, "2024-Tech-I", 1)



if __name__ == "__main__":
    main()
    task_complete("Tech-I end")

    # configs = {'-d': 'd4j', '-p': 'Chart', '-i': '0', '-m': 'dstar', '-e': 'origin'}
    # sys.argv = ["CCGroundTruthPipeline.py"]
    # ccpl = Technique1CCPipeline(project_dir, configs, 1)
    # ccpl.find_cc_index()
    # ccpl.evaluation("Tech-I-with-only-cc")
    # ccpl.calRes("Tech-I-with-only-cc")
    # a = 1
