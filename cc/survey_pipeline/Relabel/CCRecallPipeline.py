import math
import sys
import numpy as np
import pandas as pd
import copy

from cc.ReadData import ReadData
from cc.CCGroundTruthPipeline import CCGroundTruthPipeline
from fl_evaluation.calculate_suspiciousness.CalculateSuspiciousness import CalculateSuspiciousness

from CONFIG import *

class CCRecallPipeline(ReadData):

    def __init__(self, project_dir, configs):
        # 加载基础数据
        super().__init__(project_dir, configs)
        # 加载ground truth
        self.cc_ground_truth_pipeline = CCGroundTruthPipeline(project_dir, configs)
        self.cc_ground_truth_pipeline.load_ground_truth()
        self.cc_index = copy.deepcopy(self.cc_ground_truth_pipeline.all_cc_index)
        # 所需变量
        self.recall_percent = 100

    # def cc_survey(self):
    #     # self._calRes("original")
    #     self._change_label()
    #     self._calRes("survey_cc")

    def cc_num_survey(self):
        num = np.sum(np.array(self.cc_ground_truth_pipeline.all_cc_index, dtype=int))
        with open("cc_num_survey.txt", "a") as f:
            print(num, file=f)

    def cc_survey_percent(self):
        self._change_label_percent()
        self._calRes("survey_cc_recall"+str(self.recall_percent))

    def set_recall_percent(self, recall_percent):
        self.recall_percent = recall_percent

    # def _change_label(self):
    #     # for index in self.cc_ground_truth_pipeline.all_cc_index:
    #     data_df = self.load_data()
    #     passing_df = data_df[data_df["error"] == 0]
    #     failing_df = data_df[data_df["error"] == 1]
    #     # passing_df["error"][self.cc_ground_truth_pipeline.all_cc_index] = 1
    #     passing_df.loc[self.cc_ground_truth_pipeline.all_cc_index, "error"] = 1
    #     self.cc_data_df = pd.concat([passing_df, failing_df])
    #     self.data_obj.reload(self.cc_data_df)

    def _change_label_percent(self):
        # self.cc_ground_truth_pipeline.all_cc_index
        data_df = self.load_data()
        self.cc_index = copy.deepcopy(self.cc_ground_truth_pipeline.all_cc_index)
        passing_df = data_df[data_df["error"] == 0]
        failing_df = data_df[data_df["error"] == 1]

        if self.cc_ground_truth_pipeline.all_cc_index is None:
            return
        # 成功测试用例的id
        test_index = np.array(self.cc_index.index, dtype=int)
        recall_size = math.trunc(np.sum(self.cc_index)*self.recall_percent/100)
        select_index = np.where(np.array(self.cc_index) == 1)[0]
        np.random.shuffle(select_index)
        select_index = select_index[recall_size:]
        # 开始降低召回率了
        for index in select_index:
            self.cc_index.loc[test_index[index]] = False

        passing_df["error"][self.cc_index] = 1
        self.cc_data_df = pd.concat([passing_df, failing_df])
        self.data_obj.reload(self.cc_data_df)
        # m = np.where(n == 1)[0]
        # np.random.shuffle(m)
        # select_index = m[random_size:]
        # for index in select_index:
        #     self.cc_ground_truth_pipeline.all_cc_index.iloc[index] = 1
        # a = 1

    def _calRes(self, way):
        save_rank_path = os.path.join(self.project_dir, "results", "survey")
        cc = CalculateSuspiciousness(self.data_obj, self.method, save_rank_path, way)
        cc.run()
        # print(cc.rank_MFR_dict)

def main():

    for program, program_version_num in zip(program_list, program_version_num_list):
        for i in range(1, program_version_num + 1):
            if (program == "Mockito" and i == 19) or (program == "Chart" and i == 10) or (
                    program == "Closure-2023-12-6-1" and (i == 110 or i == 117 or i == 118 or i == 120 or i == 125 or i == 129)):
                continue
            print(program, i)
            configs = {'-d': 'd4j', '-p': program, '-i': i, '-m': method_para, '-e': 'origin'}
            sys.argv = os.path.basename(__file__)
            # ccspl = CCSurveyPipeline(project_dir, configs)
            # ccspl.cc_survey()
            # ccsp = CCRecallPipeline(project_dir, configs)
            # ccsp.cc_num_survey()
            for p in range(10, 101, 10):
               ccsp.set_recall_percent(p)
               ccsp.cc_survey_percent()


if __name__ == "__main__":
    # main()
    configs = {'-d': 'd4j', '-p': 'Chart', '-i': '0', '-m': 'dstar', '-e': 'origin'}
    sys.argv = os.path.basename(__file__)
    # CCSurveyPipeline 继承了ReadData，所以，最开始的时候，就加载了原始数据，和原始数据的备份
    ccsp = CCRecallPipeline(project_dir, configs)

    ccsp.set_recall_percent(50)
    ccsp.cc_survey_percent()
    # ccsp.cc_survey()
    # ccsp.cc_num_survey()