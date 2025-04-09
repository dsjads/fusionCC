import copy
import linecache
from multiprocessing import Pool

from tqdm import trange
import math
import sys
import random
import time
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from CONFIG import *
from cc.cc_baselines.BaseCCPipeline import BaseCCPipeline
from cc.mlcci_identify.FeatureTestsHandler import FeatureTestsHandler
from cc.mlcci_identify.ReadTrainData import ReadTrainData, ReadTestData
from utils.write_util import write_rank_to_txt
from sklearn.ensemble import RandomForestClassifier
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class MLCCIIdentify(BaseCCPipeline):
    def __init__(self, project_dir, configs, cita, way,K):
        super().__init__(project_dir, configs, way)
        self.train_test_config_list = []
        self.train_config_list=[]
        self.test_config_list = []
        self.test_part_list=[]
        self.cc_target = None

    def _find_cc_index(self):
        program = self.configs['-p']
        info_list = cc_info[program]
        program_len = len(info_list)
        program_method = self.configs['-m']

        for i in range(program_len):
            config = {'-d': 'd4j', '-p': program, '-i': str(info_list[i]), '-m': program_method,
                      '-e': 'origin'}
            self.train_test_config_list.append(config)
        train_config_list = self.train_test_config_list.copy()
        train_rtd = ReadTrainData(self.project_dir, train_config_list, self.way)
        for i in range(program_len):
            train_ccpls = train_rtd.ccpls.copy()
            test_ccpl = train_ccpls[i]
            train_ccpls.pop(i)
            self.machine_learning(train_ccpls, test_ccpl)

    def machine_learning(self, train_ccpls, test_ccpl):
        train_features_list =[]
        train_cc_target_list = []
        for ccpl in train_ccpls:
            features = FeatureTestsHandler.get_feature_from_file(project_dir, ccpl.program, ccpl.bug_id)
            train_features_list.append(features)
            target = ccpl.ground_truth_cc_index.astype("int").values
            train_cc_target_list.append(target)

        train_sample = np.vstack([df for df in train_features_list])
        train_cc_target = np.concatenate(train_cc_target_list)

        treeNum = 85
        depth = 85

        model = RandomForestClassifier(random_state=0,
                                         n_estimators=treeNum,
                                         max_depth=depth,
                                         n_jobs=18)
        model.fit(train_sample, train_cc_target)

        test_sample = FeatureTestsHandler.get_feature_from_file(project_dir,
                                                                test_ccpl.program, test_ccpl.bug_id)
        y_predict = model.predict_proba(test_sample)
        for i in range(y_predict.shape[0]):
            if y_predict[i][0]<y_predict[i][1]:
                test_ccpl.cc_index.iloc[i] = True

        test_ccpl.evaluation()
        test_ccpl.calRes("relabel")
        test_ccpl.calRes("trim")

if __name__ == "__main__":
    program_list=["Chart", "Lang", "Math", "Mockito", "Time"]
    for program in program_list:
        configs = {'-d': 'd4j', '-p': program, '-i': '1', '-m': method_para, '-e': 'origin'}
        sys.argv = os.path.basename(__file__)
        cbccpl = MLCCIIdentify(project_dir, configs, 1, "MLCCI", 5)
        cbccpl.find_cc_index()
    a = 1



