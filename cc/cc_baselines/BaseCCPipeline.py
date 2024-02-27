import sys

import pandas as pd

from cc.CCGroundTruthPipeline import CCGroundTruthPipeline
from CONFIG import *
from cc.cc_evaluation.Evaluation import Evaluation
from fl_evaluation.calculate_suspiciousness.CalculateSuspiciousness import CalculateSuspiciousness
from utils.write_util import write_rank_to_txt


class BaseCCPipeline(CCGroundTruthPipeline):
    def __init__(self, project_dir, configs, way):
        super().__init__(project_dir, configs)
        self.cc_index = None
        self.way = way
        self.data_df = self.load_data()
        self.init_cc_index()

    def init_cc_index(self):
        self.cc_index = pd.Series([False] * len(self.ground_truth_cc_index.index),
                                  index=self.ground_truth_cc_index.index)

    def find_cc_index(self):

        if len(self.data_df[self.data_df["error"] == 0]) == 0:
            record = dict()
            record["msg"] = "No passing tests"
            save_path = os.path.join(self.project_dir, "new_results", self.way, "record.txt")
            write_rank_to_txt(record, save_path, self.program, self.bug_id)
            return

        self._find_cc_index()

    def evaluation(self):
        if self.cc_index is None:
            print("Calculate CC index first")
            return
        else:
            original_record, record = Evaluation.evaluation(self.ground_truth_cc_index, self.cc_index)
            original_record_path = os.path.join(self.project_dir, "new_results", self.way, "origin_record.txt")
            write_rank_to_txt(original_record, original_record_path, self.program, self.bug_id)
            record_path = os.path.join(self.project_dir, "new_results", self.way, "record.txt")
            write_rank_to_txt(record, record_path, self.program, self.bug_id)

    def calRes(self, operation):
        if self.cc_index is None:
            print("Calculate CC index first")
            return
        data_df = self.load_data()
        passing_df = data_df[data_df["error"] == 0]

        if operation == "relabel":
            passing_df["error"][self.cc_index] = 1
        if operation == "trim":
            passing_df = passing_df[self.cc_index == False]

        failing_df = data_df[data_df["error"] == 1]
        cc_data_df = pd.concat([passing_df, failing_df])
        self.data_obj.reload(cc_data_df)

        op_way = self.way+"-"+operation
        # op_way = self.way
        save_rank_path = os.path.join(self.project_dir, "new_results", op_way)
        cc = CalculateSuspiciousness(self.data_obj, self.method, save_rank_path, op_way)
        cc.run()


if __name__ == "__main__":

    configs = {'-d': 'd4j', '-p': 'Chart', '-i': '0', '-m': 'dstar', '-e': 'origin'}
    sys.argv = os.path.basename(__file__)
    bpl = BaseCCPipeline(project_dir, configs, "test")
    a = 1
