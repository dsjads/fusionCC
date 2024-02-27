import math

import numpy as np
import pandas as pd
import torch

import torch.optim as optim

from CONFIG import *
from cc.cc_baselines.BaseCCPipeline import BaseCCPipeline
from cc.cc_evaluation.Evaluation import Evaluation
from cc.triplenet_model.TripletNetwork import Net, Tripletnet
from cc.triplenet_model.TripletTestCaseLoader import TripletTestCaseLoader
from cc.triplet_cc_identify.PassingTestsHandler import PassingTestsHandler
from utils.task_util import task_complete
from utils.write_util import write_rank_to_txt
from cc.triplet_cc_identify.FailingTestsHandler import FailingTestsHandler
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Training settings
parser = argparse.ArgumentParser(description='Triplet for CC')
parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                    help='input batch size for training (default: 16)')
parser.add_argument('--epochs', type=int, default=30, metavar='N',
                    help='number of epochs to train (default: 30)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
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
parser.add_argument('--name', default='TripletNet', type=str,
                    help='name of experiment')

args = parser.parse_args()

weight = 1


class TripletCCIdentify(BaseCCPipeline):
    def __init__(self, project_dir, configs, paras, way):
        super().__init__(project_dir, configs, way)
        self.CCT = None
        self.CCE = None
        self.threshold = paras[0]
        self.cita = paras[1]
        self.true_passing_tests = None
        self.failing_tests = None
        self.cc_candidates = None
        self.sus_dict = {}
        self.train_loader = None
        self.criterion = None
        self.optimizer = None
        self.tnet = None
        self.train_flag = True

    def _find_cc_index(self):
        self._find_CCE()
        if len(self.CCE) == 0:
            self.train_flag = False
            return
        self._data_preparation()
        if self.cc_candidates is None or len(self.true_passing_tests) == 0:
            self.train_flag = False
            return
        self._model_construction()
        self._train()
        self._test()

    def _data_preparation(self):
        self.CCE.append("error")
        new_data_df = self.data_df[self.CCE]
        self.failing_tests = FailingTestsHandler.get_failing_tests(new_data_df)

        # prepare data
        self.true_passing_tests, self.cc_candidates = PassingTestsHandler.get_true_passing_tests(new_data_df,
                                                                                                 self.threshold)
        if self.cc_candidates is None or len(self.true_passing_tests) == 0:
            return

        self.train_loader = torch.utils.data.DataLoader(
            TripletTestCaseLoader(passing_df=self.true_passing_tests.iloc[:, :-1] * weight,
                                  failing_df=self.failing_tests.iloc[:, :-1] * weight,
                                  ),
            batch_size=min(args.batch_size, self.true_passing_tests.shape[0]),
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )

    def _model_construction(self):

        # build model
        model = Net(len(self.CCE) - 1)
        self.tnet = Tripletnet(model)

        if args.cuda:
            self.tnet.cuda()

        # loss function and optimizer
        self.criterion = torch.nn.MSELoss()
        self.optimizer = optim.SGD(self.tnet.parameters(), lr=args.lr, momentum=args.momentum)

    def _train(self):
        for epoch in range(1, args.epochs):
            self.__train(epoch)

    def __train(self, epoch):
        self.tnet.train()
        # reference是从指定item，选取的测试用例；passing是随机选择的passing用例，failing是随机选择的failing用例
        for batch_idx, (reference, passing, failing) in enumerate(self.train_loader):
            if args.cuda:
                reference, passing, failing = reference.cuda(), passing.cuda(), failing.cuda()
            reference = reference.to(torch.float)
            passing = passing.to(torch.float)
            failing = failing.to(torch.float)

            dista, distb, prob, embedded_x, embedded_y, embedded_z = self.tnet(reference, passing, failing)
            target = torch.t(torch.FloatTensor([[0, weight]] * dista.shape[0]))

            if args.cuda:
                target = target.cuda()

            loss_triplet = self.criterion(prob, target)
            loss = loss_triplet

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # if epoch % 10 == 0:
            print('Train Epoch: {} [{}/{}]\t'
                  'loss: {}'.format(
                epoch, batch_idx * len(reference), len(self.train_loader.dataset),
                loss,
            ))

    def _test(self):
        cc_candidates_features = self.cc_candidates.iloc[:, :-1] * weight
        true_passsing_features = self.true_passing_tests.iloc[:, :-1] * weight
        failing_features = self.failing_tests.iloc[:, :-1] * weight

        self.tnet.eval()
        with torch.no_grad():
            for index, cc_candidate in cc_candidates_features.iterrows():
                # 遍历cc_candidates_features, cc_candidate为遍历的每一行张量
                cc_candidate = torch.tensor(cc_candidate.values)
                votes = []
                for _, failing_test in failing_features.iterrows():
                    random_passing_index = np.random.randint(true_passsing_features.shape[0])
                    true_passing_test = true_passsing_features.iloc[random_passing_index]

                    true_passing_test = torch.tensor(true_passing_test.values)
                    failing_test = torch.tensor(failing_test.values)

                    if args.cuda:
                        cc_candidate, true_passing_test, failing_test = cc_candidate.cuda(), true_passing_test.cuda(), failing_test.cuda()

                    reference = cc_candidate.to(torch.float)
                    passing = true_passing_test.to(torch.float)
                    failing = failing_test.to(torch.float)

                    reference = reference.unsqueeze(0)
                    passing = passing.unsqueeze(0)
                    failing = failing.unsqueeze(0)

                    dista, distb, prob, embedded_x, embedded_y, embedded_z = self.tnet(reference, passing, failing)

                    if args.cuda:
                        votes.append(prob[0].item())
                    else:
                        votes.append(prob[0])

                res = np.sum(np.array(np.array(votes) >= self.cita, dtype=int))
                if res >= math.ceil(self.failing_tests.shape[0] / 2):
                    self.cc_index[index] = True
        self.sus_dict = sorted(self.sus_dict.items(), key=lambda x: x[1], reverse=True)

    def _getfT(self, data):
        uncover = sum(data == 0)
        cover = sum(data == 1)
        fT = cover / (uncover + cover)
        return fT

    def _getpT(self, data):
        uncover = sum(data == 0)
        cover = sum(data == 1)
        pT = cover / (uncover + cover)
        return pT

    def _is_CCE(self, fail_data, pass_data):
        fT = self._getfT(fail_data)
        pT = self._getpT(pass_data)
        if fT == 1.0 and pT <= self.cita:
            return True
        else:
            return False

    def _find_CCE(self):
        failing_df = self.data_df[self.data_df["error"] == 1]
        passing_df = self.data_df[self.data_df["error"] == 0]
        CCE = []
        for i in failing_df.columns:
            if i != "error":
                if self._is_CCE(failing_df[i], passing_df[i]):
                    CCE.append(i)
        self.CCE = CCE
