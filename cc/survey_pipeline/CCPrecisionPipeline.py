import math
import sys
import numpy as np
import pandas as pd
import copy

from cc.cc_baselines.BaseCCPipeline import BaseCCPipeline

from CONFIG import *
from utils.task_util import task_complete


class CCPrecisionPipeline(BaseCCPipeline):

    def __init__(self, project_dir, configs, way):
        # 加载基础数据
        super().__init__(project_dir, configs, way)
        # 所需变量
        self.precision_percent = 100

    # def cc_num_survey(self):
    #     num = np.sum(np.array(self.ground_truth_cc_index, dtype=int))
    #     with open("cc_num_survey.txt", "a") as f:
    #         print(num, file=f)

    def cc_survey_percent(self):
        self._find_percent()

    def set_precision_percent(self, precision_percent):
        self.precision_percent = precision_percent

    def _find_percent(self):
        # self.cc_ground_truth_pipeline.all_cc_index
        # data_df = self.load_data()
        if self.ground_truth_cc_index is None:
            return

        self.cc_index = copy.deepcopy(self.ground_truth_cc_index)
        passing_df = self.data_df[self.data_df["error"] == 0]
        failing_df = self.data_df[self.data_df["error"] == 1]

        # 成功测试用例的id
        test_index = np.array(self.cc_index.index, dtype=int)
        select_index = np.where(np.array(self.cc_index, dtype=int) == 1)[0]
        success_index = np.where(np.array(self.cc_index, dtype=int) == 0)[0]
        np.random.shuffle(success_index)

        if self.precision_percent != 0:
            x = math.ceil(len(select_index) / (self.precision_percent / 100) - len(select_index))
            precision_size = min(x, len(success_index))

            # precision_size = math.trunc(len(select_index)*self.precision_percent/100)
            # select_index = select_index[:precision_size]

            false_positive_index = success_index[:precision_size]

        else:
            precision_size = math.ceil(len(success_index) * np.random.rand())
            false_positive_index = success_index[:precision_size]

        # 开始降低precision了
        for index in false_positive_index:
            self.cc_index.loc[test_index[index]] = True

        self.calRes("relabel")
        # self.calRes("trim")
        # passing_df["error"][self.cc_index] = 1
        # passing_df = passing_df[self.cc_index == False]
        # self.cc_data_df = pd.concat([passing_df, failing_df])
        # self.data_obj.reload(self.cc_data_df)
    #
    # def clean(self, passing_df, failing_df):
    #     passing_df = passing_df[self.cc_index == False]
    #     cc_data_df = pd.concat([passing_df, failing_df])
    #     self.data_obj.reload(cc_data_df)
    #     self.way = str(self.recall_percent)+"-precision-trim"
    #     self.calRes()
    #
    # def relabel(self, passing_df, failing_df):
    #     passing_df["error"][self.cc_index] = 1
    #     cc_data_df = pd.concat([passing_df, failing_df])
    #     self.way = str(self.recall_percent)+"-precision-relabel"
    #     self.data_obj.reload(cc_data_df)
    #     self.calRes()

def main():
    for program in program_list:
        for i in test_info[program]:
            print(program, i)
            configs = {'-d': 'd4j', '-p': program, '-i': i, '-m': method_para, '-e': 'origin'}
            sys.argv = os.path.basename(__file__)
            # ccspl = CCSurveyPipeline(project_dir, configs)
            # ccspl.cc_survey()
            ccsp = CCPrecisionPipeline(project_dir, configs, "precision-relabel")
            # ccsp.cc_num_survey()
            for p in range(0, 101, 10):
                ccsp.set_precision_percent(p)
                ccsp.cc_survey_percent()


if __name__ == "__main__":
    main()
    task_complete("precision end")
    #
    # configs = {'-d': 'd4j', '-p': 'Chart', '-i': '0', '-m': 'dstar', '-e': 'origin'}
    # sys.argv = os.path.basename(__file__)
    # # CCSurveyPipeline 继承了ReadData，所以，最开始的时候，就加载了原始数据，和原始数据的备份
    # ccsp = CCRecallPipeline(project_dir, configs, "recall")
    #
    # ccsp.set_recall_percent(50)
    # ccsp.cc_survey_percent()
