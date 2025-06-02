import math
import numpy as np
import torch
from torch import optim
from CONFIG import *
from cc.fusion_cc_identify.BaseIdentify import BaseIdentify
from cc.fusion_cc_identify.FailingTestsHandler import FailingTestsHandler
from cc.fusion_cc_identify.FeatureTestsHandler import FeatureTestsHandler
from cc.fusion_cc_identify.PassingTestsHandler import PassingTestsHandler
import argparse
from cc.fusion_cc_model.EFCDataLoader import CombinedInfoLoader
from cc.fusion_cc_model.FocalLoss import FocalLoss
from cc.fusion_cc_model.other_models import BiLSTMNet, CnnNet, MlpNet

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

class FusionIdentifyWithoutExpertFeature(BaseIdentify):
    def __init__(self, project_dir, configs, args_dict, way):
        super().__init__(project_dir, configs, args_dict, way)
        self.cost = 0

    def _find_cc_index(self):
        self._find_CCE()
        if len(self.CCE) == 0:
            self.train_flag = False
            return
        CCE = self.CCE[-1]
        CCE.append("error")
        new_data_df = self.data_df[CCE]

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
        indices = np.array(self.train_tests.index)

        k = 5
        part_size = math.ceil(size / k)

        for i in range(k):
            start = i * part_size
            end = (i + 1) * part_size if i < k - 1 else size
            test_index = indices[start:end]
            train_index = np.concatenate([indices[:start], indices[end:]])
            train_index = self.passing_tests.index.get_indexer(train_index)
            train_tests = self.passing_tests.iloc[train_index, :-1]
            # train_augmented_tests = self.augmentation_tests.iloc[train_index, :-1]
            train_target = self.cc_target[train_index]
            self.ssp, self.cr, self.sf = FeatureTestsHandler.get_feature_from_file(project_dir, self.program,
                                                                                   self.bug_id)
            ssp = self.ssp.iloc[train_index, :]
            cr = self.cr.iloc[train_index, :]
            sf = self.sf.iloc[train_index, :]

            train_tests,train_target,ssp,cr,sf = self.data_augmentation_with_ef(train_tests,train_target,ssp,cr,sf)
            test_index = self.passing_tests.index.get_indexer(test_index)
            if len(train_index) == 0:
                for item in test_index:
                    self.cc_index.iloc[item] = True
                return


            train_loader = torch.utils.data.DataLoader(
                CombinedInfoLoader(tests=train_tests * weight,
                                   target=train_target,
                                   ssp=ssp,
                                   cr=cr,
                                   sf=sf
                                   ),
                batch_size=min(args.batch_size, self.passing_tests.shape[0]),
                shuffle=True,
                num_workers=0,
                pin_memory=True,
            )

            # model = MSResNet()
            elements_length = len(self.CCE[-1]) - 1
            model = CnnNet(elements_length)
            # model = MlpNet(elements_length)
            # model = BiLSTMNet(elements_length)
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

            if args.cuda:
                model.cuda()
            loss_weights = torch.tensor([0.25, 0.75])
            if args.cuda:
                loss_weights = loss_weights.cuda()
            criterion = FocalLoss(gamma=5,weight=loss_weights)

            for epoch in range(1, args.epochs):
                self._train_ce(train_loader, model, criterion, optimizer, epoch)
            self._test(model, test_index)

    def _train_ce(self, train_loader, model, criterion, optimizer, epoch):
        model.train()
        for batch_idx, (test, target, ssp, cr, sf) in enumerate(train_loader):
            if args.cuda:
                test, target, ssp, cr, sf = test.cuda(), target.cuda(),ssp.cuda(), cr.cuda(), sf.cuda()
            test = test.to(torch.float)
            ssp = ssp.to(torch.float)
            cr = cr.to(torch.float)
            sf = sf.to(torch.float)
            ef = torch.hstack((ssp, cr, sf))
            # aug_test = aug_test.to(torch.float)
            if test.size(0) == 1:
                test = test.repeat(2, 1)
                target = target.repeat(2, 1)
                ef = ef.repeat(2,1)

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

                ef = torch.hstack((ssp, cr, sf))
                prob = model(test)
                prob = torch.softmax(prob, dim=1)
                if prob[0][0] < prob[0][1]:
                    self.cc_index.iloc[item] = True
