import math
import numpy as np
import pandas as pd
import torch
from torch import optim
from torch.nn import MSELoss
import torch.nn.functional as F
from CONFIG import *
from cc.fusion_cc_identify.BaseIdentify import BaseIdentify
from cc.fusion_cc_identify.FailingTestsHandler import FailingTestsHandler
from cc.fusion_cc_identify.PassingTestsHandler import PassingTestsHandler
from cc.fusion_cc_model.ContraDataLoader import ContraDataLoader
import argparse
from cc.fusion_cc_model.FusionNet import FusionNet, CnnNet, SupCENet
from cc.fusion_cc_model.SupConLoss import SupConLoss

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

class SupContraCCIdentify(BaseIdentify):
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

        columns_with_ones = self.failing_tests.columns[(self.failing_tests == 1).any()]
        augmented_data = pd.DataFrame(0, index=self.passing_tests.index, columns=self.passing_tests.columns)

        # 将指定列的数据从 self.passing_tests 复制到 augmented_data 中
        augmented_data[columns_with_ones] = self.passing_tests[columns_with_ones]
        self.augmentation_tests = augmented_data
        target = self.ground_truth_cc_index.astype("int").values
        # self.cc_target = torch.from_numpy(target.astype(np.float32)).unsqueeze(1)
        self.cc_target = torch.FloatTensor([[0, 1]] * len(target))
        for i in range(len(target)):
            if target[i] == 1:
                self.cc_target[i] = torch.FloatTensor([0, 1])
            else:
                self.cc_target[i] = torch.FloatTensor([1, 0])


        size = self.passing_tests.shape[0]
        # indices = np.arange(size)
        indices = np.array(self.passing_tests.index)
        # np.random.shuffle(indices)

        k = 5
        part_size = math.ceil(size / k)

        for i in range(k):
            start = i * part_size
            end = (i + 1) * part_size if i < k - 1 else size
            test_index = indices[start:end]
            train_index = np.concatenate([indices[:start], indices[end:]])
            train_index = self.passing_tests.index.get_indexer(train_index)

            train_tests = self.passing_tests.iloc[train_index, :-1]
            train_augmented_tests = self.augmentation_tests.iloc[train_index, :-1]
            train_target = self.cc_target[train_index]
            test_index = self.passing_tests.index.get_indexer(test_index)

            train_loader = torch.utils.data.DataLoader(
                ContraDataLoader(
                    tests=train_tests,
                    augmentation= train_augmented_tests,
                    target=train_target,
                ),
                batch_size=min(args.batch_size, self.passing_tests.shape[0]),
                shuffle=True,
                num_workers=0,
                pin_memory=True
            )

            encoder = CnnNet()

            # if args.cuda:
            #     encoder.cuda()

            # criterion = SupConLoss()
            optimizer = optim.SGD(encoder.parameters(), lr=args.lr, momentum=args.momentum)
            # optimizer = optim.Adam(encoder.parameters(), lr=args.lr)
            # for epoch in range(1, args.epochs):
            #     self._train_feat(train_loader, encoder, criterion, optimizer, epoch)
            model = SupCENet(64, encoder)
            if args.cuda:
                model.cuda()
            criterion = torch.nn.MSELoss()
            for epoch in range(1, args.epochs):
                self._train_ce(train_loader, model, criterion, optimizer, epoch)
            self._test(model, test_index)

    def _train_ce(self, train_loader, model, criterion, optimizer, epoch):
        model.train()
        for batch_idx, (test, aug_test, target) in enumerate(train_loader):
            if args.cuda:
                test, aug_test, target = test.cuda(), aug_test.cuda(), target.cuda()
            test = test.to(torch.float)
            aug_test = aug_test.to(torch.float)
            if test.size(0) == 1:
                test = test.repeat(2, 1)
                target = target.repeat(2, 1)

            prob = model(test)
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


    def _train_feat(self, train_loader, model, criterion, optimizer, epoch):
        model.train()
        for batch_idx, (test, aug_test, target) in enumerate(train_loader):
            if args.cuda:
                test, aug_test, target = test.cuda(), aug_test.cuda(), target.cuda()
            test = test.to(torch.float)
            aug_test = aug_test.to(torch.float)

            if test.size(0) == 1:
                test = test.repeat(2, 1)
                aug_test = aug_test.repeat(2, 1)
                target = target.repeat(2, 1)

            f1 = model(test)
            f2 = model(aug_test)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            target = torch.argmax(target, dim=1, keepdim=True)
            loss = criterion(features, target)

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
        with torch.no_grad():
            for item in test_index:
                test = self.passing_tests.iloc[item, :-1]
                test = torch.tensor(test.values)
                if args.cuda:
                    test = test.cuda()
                test = torch.unsqueeze(test.to(torch.float), dim=0)
                prob = model(test)
                if prob[0][0] < prob[0][1]:
                    self.cc_index.iloc[item] = True
