import argparse
import math

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import optim

from CONFIG import *
from cc.cc_baselines.BaseCCPipeline import BaseCCPipeline
from cc.expert_feature_cc_model.EFCDataLoader import CombinedInfoTestCaseLoader
from cc.discard_method.TripletWithExpertFeatureNetwork import Net, TripletWithExpertFeatureNet, Net4, Net1, \
    Net2, Net3

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


class ExpertFeatureCombinedIdentify(BaseCCPipeline):
    def __init__(self, project_dir, configs, args_dict, way):
        super().__init__(project_dir, configs, way)
        self.CCT = None
        self.CCE = None
        self.args_dict = args_dict
        self.cita = None
        self.true_passing_tests = None
        self.failing_tests = None
        self.cc_candidates = None
        self.sus_dict = {}
        self.train_loader = None
        self.criterion = None
        self.optimizer = None
        self.tnet = None
        self.train_flag = True
        # print(self.args_dict["cce_threshold"])
        file_path = f"features-{self.program}-{self.bug_id}.csv"
        self.feature_dir = os.path.join(self.project_dir, "feature", f"{self.program}-2023-12-6-1", file_path)
        # print(self.feature_dir)
        # self.feature_matrix = np.load(self.feature_dir)
        expert_matrix = pd.read_csv(self.feature_dir, index_col=0)
        scaler = StandardScaler()
        scaler.fit(expert_matrix.values)
        self.expert_matrix = pd.DataFrame(scaler.transform(expert_matrix.values), columns=expert_matrix.columns)

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

        # self.failing_tests = FailingTestsHandler.get_failing_tests(new_data_df)
        #
        # failing_tests_index = list(self.failing_tests.index)

        # 获取被认为是真实通过的测试用例，潜在cc用例，真实通过测试用例的feature，潜在cc用例的feature
        self.failing_tests, self.true_passing_tests, self.cc_candidates, \
        self.failing_tests_expert, self.true_passing_tests_expert, self.cc_candidates_expert = self.get_TP_when_already_find_cce(
            new_data_df)

        if self.cc_candidates is None or len(self.true_passing_tests) == 0:
            return

        self.train_loader = torch.utils.data.DataLoader(
            CombinedInfoTestCaseLoader(passing_df=self.true_passing_tests.iloc[:, :-1] * weight,
                                       true_passing_test_expert=self.true_passing_tests_expert,
                                       failing_df=self.failing_tests.iloc[:, :-1] * weight,
                                       failing_test_expert=self.failing_tests_expert
                                       ),
            batch_size=min(args.batch_size, self.true_passing_tests.shape[0]),
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )

    def _model_construction(self):
        # build the model
        model = Net(len(self.CCE) - 1)
        model1 = Net1(self.expert_matrix.shape[1] // 3)
        model2 = Net2(self.expert_matrix.shape[1] // 3)
        model3 = Net3(self.expert_matrix.shape[1] // 3)
        model4 = Net4(self.expert_matrix.shape[1], model1, model2, model3)
        self.tnet = TripletWithExpertFeatureNet(model, model4)

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
        for batch_idx, (reference, reference_expert, passing, passing_expert, failing, failing_expert) in enumerate(
                self.train_loader):
            if args.cuda:
                reference, reference_expert, passing, passing_expert, failing, failing_expert = \
                    reference.cuda(), reference_expert.cuda(), passing.cuda(), passing_expert.cuda(), failing.cuda(), failing_expert.cuda()

            reference = reference.to(torch.float)
            passing = passing.to(torch.float)
            failing = failing.to(torch.float)
            reference_expert = reference_expert.to(torch.float)
            passing_expert = passing_expert.to(torch.float)
            failing_expert = failing_expert.to(torch.float)

            dista, distb, prob, embedded_x, embedded_y, embedded_z = self.tnet(reference, reference_expert, passing,
                                                                               passing_expert, failing,
                                                                               failing_expert)
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
        cc_candidates_expert_features = self.cc_candidates_expert
        true_passing_features = self.true_passing_tests.iloc[:, :-1] * weight
        true_passing_expert_features = self.true_passing_tests_expert
        failing_features = self.failing_tests.iloc[:, :-1] * weight
        failing_expert_features = self.failing_tests_expert

        self.tnet.eval()
        with torch.no_grad():
            for index, cc_candidate in cc_candidates_features.iterrows():
                # cc candidate test and expert feature
                cc_candidate_expert = cc_candidates_expert_features.loc[index]
                cc_candidate, cc_candidate_expert = torch.tensor(cc_candidate.values), torch.tensor(
                    cc_candidate_expert.values)

                votes = []
                for failing_index, failing_test in failing_features.iterrows():
                    # failing test and expert features
                    failing_test_expert = failing_expert_features.loc[failing_index]
                    failing_test, failing_test_expert = torch.tensor(failing_test.values), torch.tensor(
                        failing_test_expert.values)

                    # true passing test and expert features
                    random_passing_index = np.random.randint(true_passing_features.shape[0])
                    true_passing_test = true_passing_features.iloc[random_passing_index]
                    true_passing_test_expert = true_passing_expert_features.iloc[random_passing_index]
                    true_passing_test, true_passing_test_expert = torch.tensor(true_passing_test.values), torch.tensor(
                        true_passing_test_expert.values)

                    if args.cuda:
                        cc_candidate, cc_candidate_expert, true_passing_test, true_passing_test_expert, failing_test, failing_test_expert \
                            = cc_candidate.cuda(), cc_candidate_expert.cuda(), true_passing_test.cuda(), true_passing_test_expert.cuda(), \
                              failing_test.cuda(), failing_test_expert.cuda()

                    reference = cc_candidate.to(torch.float)
                    passing = true_passing_test.to(torch.float)
                    failing = failing_test.to(torch.float)
                    cc_candidate_expert = cc_candidate_expert.to(torch.float)
                    true_passing_test_expert = true_passing_test_expert.to(torch.float)
                    failing_test_expert = failing_test_expert.to(torch.float)

                    reference = reference.unsqueeze(0)
                    passing = passing.unsqueeze(0)
                    failing = failing.unsqueeze(0)
                    reference_expert = cc_candidate_expert.unsqueeze(0)
                    true_passing_test_expert = true_passing_test_expert.unsqueeze(0)
                    failing_test_expert = failing_test_expert.unsqueeze(0)

                    dista, distb, prob, embedded_x, embedded_y, embedded_z = self.tnet(reference, reference_expert,
                                                                                       passing,
                                                                                       true_passing_test_expert,
                                                                                       failing,
                                                                                       failing_test_expert)
                    if args.cuda:
                        votes.append(prob[0].item())
                    else:
                        votes.append(prob[0])

                res = np.sum(np.array(np.array(votes) >= self.cita, dtype=int))
                if res >= math.ceil(self.failing_tests.shape[0] / 2):
                    self.cc_index[index] = True
        self.sus_dict = sorted(self.sus_dict.items(), key=lambda x: x[1], reverse=True)

    def _out(self):
        pass

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

    def _is_CCE(self, fail_data, pass_data, cita):
        fT = self._getfT(fail_data)
        pT = self._getpT(pass_data)
        if fT == 1.0 and pT < cita:
            return True
        else:
            return False

    def _find_CCE(self):
        if "cce_threshold" not in self.args_dict:
            column = self.data_df.columns[:-1]
            self.CCE = list(column)
            return
        self.cita = self.args_dict["cce_threshold"]
        failing_df = self.data_df[self.data_df["error"] == 1]
        passing_df = self.data_df[self.data_df["error"] == 0]
        CCE = []
        for i in failing_df.columns:
            if i != "error":
                if self._is_CCE(failing_df[i], passing_df[i], self.cita):
                    CCE.append(i)
        self.CCE = CCE

    def get_TP_when_already_find_cce(self, data_df):
        # failing tests
        failing_test = data_df[data_df["error"] == 1]
        failing_tests_index = list(failing_test.index)
        failing_test_feature = self.expert_matrix.loc[failing_tests_index]

        # separate passing tests
        passing_df = data_df[data_df["error"] == 0]
        new_data_df = passing_df.drop(passing_df.columns[-1], axis=1)
        sum_df = new_data_df.sum(axis=1)

        cc_candidate_list = list(sum_df[sum_df > 0].index)
        true_passing_list = list(sum_df[sum_df == 0].index)
        true_passing_test = data_df.iloc[true_passing_list, :].astype('float32')
        cc_candidate = data_df.iloc[cc_candidate_list, :].astype('float32')
        true_passing_test_feature = self.expert_matrix.loc[true_passing_list]
        cc_candidate_feature = self.expert_matrix.loc[cc_candidate_list]

        return failing_test, true_passing_test, cc_candidate, \
               failing_test_feature, true_passing_test_feature, cc_candidate_feature

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
