import copy
import sys
import numpy as np
import pandas as pd
from sklearn import svm
from cc.CCGroundTruthPipeline import CCGroundTruthPipeline
from cc.cc_baselines.BaseCCPipeline import BaseCCPipeline
from cc.cc_evaluation.Evaluation import Evaluation
from utils.task_util import task_complete
from utils.write_util import write_rank_to_txt

from CONFIG import *
from cc.core import run


class SVMBasedCCPipeline(BaseCCPipeline):
    def __init__(self, project_dir, configs, N, way):
        super().__init__(project_dir, configs, way)
        data_df = self.load_data()
        self.N = min(N, len(data_df[data_df["error"] == 0]))
        self.cc_index = None

    def _find_cc_index(self):
        data_df = self.load_data()
        N = self.N
        TP_data = data_df[data_df["error"] == 0]
        TF_data = data_df[data_df["error"] == 1]
        TP_split = self.TPsetSplit(TP_data, N)
        Z = []
        TR = []
        TS = []
        classifier = []
        for i in range(TP_data.shape[0]):
            Z.append([])
        for i in range(N):
            TR.append(pd.concat([TP_split[i], TF_data]))
            TS.append(self.get_diff(TP_data, TP_split[i]))
            TR_set = TR[i].iloc[:, [i for i in range(TP_data.shape[1] - 1)]]
            TR_label = TR[i].iloc[:, [TP_data.shape[1] - 1]]
            TS_set = TS[i].iloc[:, [i for i in range(TP_data.shape[1] - 1)]]
            TS_label = TS[i].iloc[:, [TP_data.shape[1] - 1]]
            classifier.append(svm.SVC(C=2, kernel='rbf', gamma=10, decision_function_shape='ovo'))  # ovr:一对多策略
            classifier[i].fit(TR_set, np.array(TR_label.T)[0])  # ravel函数在降维时默认是行序优先
            for item in TS_set.index:
                loc_item = TS_set.index.get_loc(item)
                z_item = TP_data.index.get_loc(item)
                Z[z_item].append(classifier[i].score(TS_set.iloc[[loc_item], :], TS_label.iloc[[loc_item], :]))
                # print(Z[z_item])
        cc_index = []
        for i in range(TP_data.shape[0]):
            coi = 0
            non = 0
            for j in range(len(Z[i])):
                if (Z[i][j] == 0):
                    coi += 1
                else:
                    non += 1
            if (coi >= non):
                cc_index.append(i)
        container = pd.Series([False for i in range(len(TP_data.index.tolist()))], index=TP_data.index)
        container.iloc[cc_index] = True
        self.cc_index = container

    # tp为需要拆分的集合，n为拆分数，result为获得的拆分结果
    def TPsetSplit(self, tp, n):
        shuffled = tp.sample(frac=1)
        result = np.array_split(shuffled, n)
        return result

    def get_diff(self, df1, df2):
        index = df2.index
        set_diff_df = df1.drop(index)
        return set_diff_df

        # 根据索引获得iloc的行号

    def get_multi_row_by_index(self, dataset, index_list):
        datas_with_index = []
        for i in range(len(index_list)):
            data = dataset.index.get_loc(index_list[i])
            datas_with_index.append(data)
        return datas_with_index

def main():
    program_list = [
        "Chart",
        "Closure-2023-12-6-1",
        "Lang",
        "Math",
        "Mockito",
        "Time"
    ]
    run(program_list, "Chart", 1, SVMBasedCCPipeline, "2024-SVM", 4)

if __name__ == "__main__":
    main()
    task_complete("SVM end")
#
#     # # project_dir = os.path.join(os.path.dirname(__file__), "..", "..")
#     configs = {'-d': 'd4j', '-p': 'Chart', '-i': '0', '-m': 'dstar', '-e': 'origin'}
#     sys.argv = ["SVMBasedCCPipeline.py"]
#     svmccpl = SVMBasedCCPipeline(project_dir, configs, 4, "SVM")
#     svmccpl.find_cc_index()
#     print(svmccpl.cc_index)
#     svmccpl.evaluation()



