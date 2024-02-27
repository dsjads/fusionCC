import math
import sys
import numpy as np
import pandas as pd
import copy

from cc.cc_baselines.BaseCCPipeline import BaseCCPipeline

from CONFIG import *
from utils.task_util import task_complete


class CCRecallPipeline(BaseCCPipeline):

    def __init__(self, project_dir, configs, way):
        # 加载基础数据
        super().__init__(project_dir, configs, way)
        # 所需变量
        self.recall_percent = 100

    # def cc_num_survey(self):
    #     num = np.sum(np.array(self.ground_truth_cc_index, dtype=int))
    #     with open("cc_num_survey.txt", "a") as f:
    #         print(num, file=f)

    def cc_survey_percent(self):
        self._find_percent()

    def set_recall_percent(self, recall_percent):
        self.recall_percent = recall_percent

    def _find_percent(self):
        # self.cc_ground_truth_pipeline.all_cc_index
        # data_df = self.load_data()
        if self.ground_truth_cc_index is None:
            return

        self.cc_index = copy.deepcopy(self.ground_truth_cc_index)
        select_index = np.where(np.array(self.cc_index) == 1)[0]
        np.random.shuffle(select_index)

        # 成功测试用例的id
        test_index = np.array(self.cc_index.index, dtype=int)
        if self.recall_percent == 0:
            recall_size = math.ceil(len(select_index) * np.random.rand())
        else:
            recall_size = math.trunc(np.sum(self.cc_index) * self.recall_percent / 100)

        select_index = select_index[recall_size:]

        # 开始降低召回率了
        for index in select_index:
            self.cc_index.loc[test_index[index]] = False

        # self.calRes("relabel")
        self.calRes("trim")
        # self.clean(passing_df, failing_df)
        # self.relabel(passing_df, failing_df)
        # passing_df["error"][self.cc_index] = 1
        # passing_df = passing_df[self.cc_index == False]
        # self.cc_data_df = pd.concat([passing_df, failing_df])
        # self.data_obj.reload(self.cc_data_df)
    #
    # def clean(self, passing_df, failing_df):
    #     passing_df = passing_df[self.cc_index == False]
    #     cc_data_df = pd.concat([passing_df, failing_df])
    #     self.data_obj.reload(cc_data_df)
    #     self.way = str(self.recall_percent)+"recall-trim"
    #     self.calRes()
    #
    # def relabel(self, passing_df, failing_df):
    #     passing_df["error"][self.cc_index] = 1
    #     cc_data_df = pd.concat([passing_df, failing_df])
    #     self.way = str(self.recall_percent)+"recall-relabel"
    #     self.data_obj.reload(cc_data_df)
    #     self.calRes()

def main():
    for program in program_list:
        for i in com_info[program]:
            print(program, i)
            configs = {'-d': 'd4j', '-p': program, '-i': i, '-m': method_para, '-e': 'origin'}
            sys.argv = os.path.basename(__file__)
            # ccspl = CCSurveyPipeline(project_dir, configs)
            # ccspl.cc_survey()
            ccsp = CCRecallPipeline(project_dir, configs, "recall-trim")
            # ccsp.cc_num_survey()
            for p in range(0, 101, 10):
                ccsp.set_recall_percent(p)
                ccsp.cc_survey_percent()


if __name__ == "__main__":
    main()
    task_complete("recall end")
    #
    # configs = {'-d': 'd4j', '-p': 'Chart', '-i': '0', '-m': 'dstar', '-e': 'origin'}
    # sys.argv = os.path.basename(__file__)
    # # CCSurveyPipeline 继承了ReadData，所以，最开始的时候，就加载了原始数据，和原始数据的备份
    # ccsp = CCRecallPipeline(project_dir, configs, "recall")
    #
    # ccsp.set_recall_percent(50)
    # ccsp.cc_survey_percent()
