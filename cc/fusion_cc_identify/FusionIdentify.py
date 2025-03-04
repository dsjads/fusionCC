import math
import time
import numpy as np
import pandas as pd
import torch
from torch import optim

from CONFIG import *
from cc.fusion_cc_identify.BaseIdentify import BaseIdentify
from cc.fusion_cc_identify.FailingTestsHandler import FailingTestsHandler
from cc.fusion_cc_identify.FeatureTestsHandler import FeatureTestsHandler
from cc.fusion_cc_identify.PassingTestsHandler import PassingTestsHandler
from cc.fusion_cc_model.EFCDataLoader import CombinedInfoLoader
import argparse

from cc.fusion_cc_model.FocalLoss import FocalLoss
from cc.fusion_cc_model.FusionNet import FusionNet
from cc.fusion_cc_model.FusionNetConcat import FusionNetConcat

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Training settings
parser = argparse.ArgumentParser(description='Triplet for CC')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
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


class FusionIdentify(BaseIdentify):
    def __init__(self, project_dir, configs, args_dict, way):
        super().__init__(project_dir, configs, args_dict, way)
        self.cost = 0

    def _find_cc_index(self):
        self._find_CCE()
        if len(self.CCE) == 0:
            self.train_flag = False
            return
        self.CCE.append("error")
        new_data_df = self.data_df[self.CCE]

        self.failing_tests = FailingTestsHandler.get_failing_tests(new_data_df)
        self.passing_tests = PassingTestsHandler.get_passing_tests(new_data_df)
        self.train_tests = self.passing_tests[self.passing_tests.sum(axis=1) != 0]
        target = self.ground_truth_cc_index.astype("int").values
        self.cc_target = torch.FloatTensor([[0, 1]] * len(target))
        for i in range(len(target)):
            if target[i] == 1:
                self.cc_target[i] = torch.FloatTensor([0, 1])
            else:
                self.cc_target[i] = torch.FloatTensor([1, 0])

        size = self.train_tests.shape[0]
        # indices = np.arange(size)
        indices = np.array(self.train_tests.index)
        np.random.shuffle(indices)

        k = 5
        part_size = math.ceil(size / k)

        # 依次取出
        for i in range(k):
            start = i * part_size
            end = (i + 1) * part_size if i < k - 1 else size
            test_index = indices[start:end]
            train_index = np.concatenate([indices[:start], indices[end:]])
            # train_tests = self.passing_tests.loc[train_index,:-1]
            train_index = self.passing_tests.index.get_indexer(train_index)
            train_tests = self.passing_tests.iloc[train_index, :-1]
            train_target = self.cc_target[train_index]
            test_index = self.passing_tests.index.get_indexer(test_index)
            if len(train_index) == 0:
                for item in test_index:
                    self.cc_index.iloc[item] = True
                return
            self.ssp, self.cr, self.sf = FeatureTestsHandler.get_feature_from_file(project_dir, self.program,
                                                                                   self.bug_id)
            ssp_feature = self.ssp.iloc[train_index, :]
            cr_feature = self.cr.iloc[train_index, :]
            sf_feature = self.sf.iloc[train_index, :]

            train_loader = torch.utils.data.DataLoader(
                CombinedInfoLoader(tests=train_tests * weight,
                                   target=train_target,
                                   ssp=ssp_feature,
                                   cr=cr_feature,
                                   sf=sf_feature
                                   ),
                batch_size=min(args.batch_size, self.passing_tests.shape[0]),
                shuffle=True,
                num_workers=0,
                pin_memory=True,
            )

            elements_length = len(self.CCE) - 1
            model = FusionNetConcat(elements_length)

            if args.cuda:
                model.cuda()
            # loss function and optimizer
            # criterion = torch.nn.MSELoss()
            loss_weights = torch.tensor([0.25, 0.75])
            if args.cuda:
                loss_weights = loss_weights.cuda()
            # criterion = torch.nn.CrossEntropyLoss(weight=loss_weights)
            criterion = FocalLoss(gamma=5, weight=loss_weights)
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

            for epoch in range(1, args.epochs):
                self._train(train_loader, model, criterion, optimizer, epoch)
            self._test(model, test_index)

    def _train(self, train_loader, model, criterion, optimizer, epoch):
        model.train()
        for batch_idx, (tests, target, ssp, cr, sf) in enumerate(train_loader):
            if args.cuda:
                tests, target, ssp, cr, sf = tests.cuda(), target.cuda(), ssp.cuda(), cr.cuda(), sf.cuda()
            tests = tests.to(torch.float)
            ssp = ssp.to(torch.float)
            cr = cr.to(torch.float)
            sf = sf.to(torch.float)
            expert_feature = torch.hstack((ssp, cr, sf))

            if tests.size(0) == 1:
                tests = tests.repeat(2, 1)
                expert_feature = expert_feature.repeat(2, 1)
                target = target.repeat(2, 1)

            prob = model(tests, expert_feature)

            if args.cuda:
                target = target.cuda()

            loss = criterion(prob, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # if epoch % 10 == 0:
            print('Train Epoch: {} [{}/{}]\t'
                  'loss: {}'.format(
                epoch, batch_idx * len(target), len(train_loader.dataset),
                loss,
            ))

    def _test(self, model, test_index):
        model.eval()
        start = time.time()
        with torch.no_grad():
            for item in test_index:
                test = self.passing_tests.iloc[item, :-1]
                ssp = self.ssp.iloc[item]
                cr = self.cr.iloc[item]
                sf = self.sf.iloc[item]

                test = torch.tensor(test.values)
                ssp = torch.tensor(ssp.values)
                cr = torch.tensor(cr.values)
                sf = torch.tensor(sf.values)

                if args.cuda:
                    test, ssp, cr, sf = test.cuda(), ssp.cuda(), cr.cuda(), sf.cuda()
                test = torch.unsqueeze(test.to(torch.float), dim=0)
                ssp = torch.unsqueeze(ssp.to(torch.float), dim=0)
                cr = torch.unsqueeze(cr.to(torch.float), dim=0)
                sf = torch.unsqueeze(sf.to(torch.float), dim=0)

                expert_feature = torch.hstack((ssp, cr, sf))

                prob = model(test, expert_feature)

                if prob[0][0] < prob[0][1]:
                    self.cc_index.iloc[item] = True
        end = time.time()
        self.cost += end - start

    def cost(self):
        return self.cost

    def get_TP_when_already_find_cce(self, data_df, feature_matrix):
        passing_df = data_df[data_df["error"] == 0]
        new_data_df = passing_df.drop(passing_df.columns[-1], axis=1)
        sum_df = new_data_df.sum(axis=1)

        cc_candidate_list = list(sum_df[sum_df > 0].index)
        true_passing_list = list(sum_df[sum_df == 0].index)
        true_passing_test = data_df.iloc[true_passing_list, :].astype('float32')
        cc_candidate = data_df.iloc[cc_candidate_list, :].astype('float32')
        true_passing_test_feature = feature_matrix.loc[true_passing_list]
        cc_candidate_feature = feature_matrix.loc[cc_candidate_list]

        return true_passing_test, cc_candidate, true_passing_test_feature, cc_candidate_feature

    def get_TP_when_not_find_cce(self, data_df):
        failing_df = data_df[data_df["error"] == 1]
        passing_df = data_df[data_df["error"] == 0]
        CCE = []
        for i in failing_df.columns:
            if i != "error":
                if self._is_CCE(failing_df[i], passing_df[i], self.cita):
                    CCE.append(i)
        new_data_df = passing_df[CCE]
        sum_df = new_data_df.sum(axis=1)
        cc_candidate_list = list(sum_df[sum_df > 0].index)
        true_passing_list = list(sum_df[sum_df == 0].index)
        return data_df.iloc[true_passing_list, :].astype('float32'), data_df.iloc[cc_candidate_list, :].astype(
            'float32')
