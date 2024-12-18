import math
import random
import sys
import time

import numpy as np
import pandas as pd
import torch

import torch.optim as optim

from CONFIG import *
from cc.MLP_DFL_cc_model.MLPDFLNet import Net2
from cc.MLP_DFL_cc_model.ReadTrainData import ReadTrainData
from cc.cc_baselines.BaseCCPipeline import BaseCCPipeline
from cc.fusion_cc_identify.BaseIdentify import BaseIdentify
from cc.fusion_cc_identify.FailingTestsHandler import FailingTestsHandler
from cc.fusion_cc_identify.FeatureTestsHandler import FeatureTestsHandler
from cc.fusion_cc_identify.PassingTestsHandler import PassingTestsHandler
from cc.fusion_cc_model.CnnNet import CnnSematicNet
from cc.fusion_cc_model.EFCDataLoader import CombinedInfoLoaderWithoutCovInfo, CombinedInfoLoader
from cc.fusion_cc_model.ExpertFeatureCombinedNetwork import Net1, Net3, Net4, EFCNetwork
import argparse

from cc.fusion_cc_model.MlpNet import MlpSematicNet

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 16)')
parser.add_argument('--epochs', type=int, default=30, metavar='N',
                    help='number of epochs to train (default: 30)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--cuda', type=bool, default=True, help='CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--margin', type=float, default=0, metavar='M',
                    help='margin for triplet loss (default: 0.2)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='MLPDFLCCNet', type=str,
                    help='name of experiment')

args = parser.parse_args()


class FusionIdentifyAddSus(BaseIdentify):
    def __init__(self, project_dir, configs, args_dict, way):
        super().__init__(project_dir, configs, args_dict, way)

    def _find_cc_index(self):
        self._find_CCE()
        if len(self.CCE) == 0:
            self.train_flag = False
            return
        self.CCE.append("error")
        new_data_df = self.data_df[self.CCE]

        self.failing_tests = FailingTestsHandler.get_failing_tests(new_data_df)
        self.passing_tests = PassingTestsHandler.get_passing_tests(new_data_df)
        self.failing_tests = self.failing_tests.iloc[:, :-1]
        self.passing_tests = self.passing_tests.iloc[:, :-1]
        target = self.ground_truth_cc_index.astype("int").values
        self.cc_target = torch.FloatTensor([[0, 1]] * self.passing_tests.shape[0])
        for i in range(len(target)):
            if target[i] == 1:
                self.cc_target[i] = torch.FloatTensor([0, 1])
            else:
                self.cc_target[i] = torch.FloatTensor([1, 0])

        origin_data = self.load_suspicious_data()

    def load_suspicious_data(self):
        suspicious_df = FeatureTestsHandler.get_sus_data_from_file(project_dir, self.program,
                                                                   self.bug_id)

        print(suspicious_df)
        print(self.passing_tests)

        origin_data = self.passing_tests.values
        sus_data = suspicious_df.values

        result_np = np.zeros((origin_data.shape[0], origin_data.shape[1], sus_data.shape[0]))

        # 遍历sus_data的每一行，与origin_data的每一行对应元素相乘
        for i in range(sus_data.shape[0]):
            result_np[:, :, i] = np.multiply(origin_data, sus_data[i, :])

        # 将原始矩阵与结果矩阵拼接
        result_np = np.concatenate((origin_data[:, :, np.newaxis], result_np), axis=2)

        return result_np
