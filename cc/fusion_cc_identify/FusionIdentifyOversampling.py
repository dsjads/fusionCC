import time
import math
import numpy as np
# import shap
import torch
from torch import optim
from CONFIG import *
from cc.fusion_cc_identify.BaseIdentify import BaseIdentify
from cc.fusion_cc_identify.FailingTestsHandler import FailingTestsHandler
from cc.fusion_cc_identify.FeatureTestsHandler import FeatureTestsHandler
from cc.fusion_cc_identify.PassingTestsHandler import PassingTestsHandler
from cc.fusion_cc_model.ContraDataLoader import ContraDataLoader, TestsDataLoader
import argparse

from cc.fusion_cc_model.EFCDataLoader import CombinedInfoLoader
from cc.fusion_cc_model.FocalLoss import FocalLoss
from cc.fusion_cc_model.model import MSFusionNet

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

class FusionIdentifyOversampling(BaseIdentify):
    def __init__(self, project_dir, configs, args_dict, way):
        super().__init__(project_dir, configs, args_dict, way)
        self.infer_cost = 0
        self.train_cost = 0

    def _find_cc_index(self):
        self._find_CCE()
        if len(self.CCE[-1]) == 0:
            self.train_flag = False
            return
        CCE = self.CCE[-1]
        CCE.append("error")
        new_data_df = self.data_df[CCE]

        self.failing_tests = FailingTestsHandler.get_failing_tests(new_data_df)
        self.passing_tests = PassingTestsHandler.get_passing_tests(new_data_df)
        # self.train_tests = self.passing_tests[self.passing_tests.sum(axis=1) != 0]
        self.train_tests = self.passing_tests

        # 将指定列的数据从 self.passing_tests 复制到 augmented_data中
        target = self.ground_truth_cc_index.astype("int").values
        self.cc_target = torch.FloatTensor([[0, 1]] * len(target))
        for i in range(len(target)):
            if target[i] == 1:
                self.cc_target[i] = torch.FloatTensor([0, 1])
            else:
                self.cc_target[i] = torch.FloatTensor([1, 0])
        size = self.train_tests.shape[0]
        indices = np.array(self.train_tests.index)
        # np.random.shuffle(indices)

        k = 2
        part_size = math.ceil(size / k)
        coverage_shap, handcrafted_shap = 0, 0
        for i in range(k):
            start = i * part_size
            end = (i + 1) * part_size if i < k - 1 else size
            test_index = indices[start:end]
            train_index = np.concatenate([indices[:start], indices[end:]])
            train_index = self.passing_tests.index.get_indexer(train_index)
            train_tests = self.passing_tests.iloc[train_index, :-1]
            train_target = self.cc_target[train_index]
            self.ssp, self.cr, self.sf = FeatureTestsHandler.get_feature_from_file(project_dir, self.program,
                                                                                   self.bug_id)
            ssp = self.ssp.iloc[train_index, :]
            cr = self.cr.iloc[train_index, :]
            sf = self.sf.iloc[train_index, :]


            # train_tests,train_target,ssp,cr,sf = self.data_augmentation_under_imblearn(train_tests,train_target,ssp,cr,sf,"undersample")

            test_index = self.passing_tests.index.get_indexer(test_index)

            if len(train_index) == 0:
                for item in test_index:
                    self.cc_index.iloc[item] = True
                return

            train_target = torch.argmax(train_target, dim=1)

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


            model = MSFusionNet()
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

            if args.cuda:
                model.cuda()
            # loss_weights = torch.tensor([0.25, 0.75])
            # if args.cuda:
            #     loss_weights = loss_weights.cuda()
            # criterion = FocalLoss(weight=loss_weights)
            # class0_count = self.cc_target[:,0].sum().item()
            # class1_count = self.cc_target[:, 1].sum().item()
            # total_samples = self.cc_target.shape[0]  # 样本数 = 行数

            # 3. 计算权重（平衡权重公式）
            # 避免除以0（如果某类样本数为0）
            # epsilon = 1e-8
            # weight0 = total_samples / (2 * (class0_count + epsilon))
            # weight1 = total_samples / (2 * (class1_count + epsilon))
            # class_weights = torch.tensor([weight0, weight1], dtype=torch.float32)
            # class_weights = class_weights.cuda()
            # print(f"计算得到的类别权重: {class_weights}")
            criterion = torch.nn.CrossEntropyLoss()
            # criterion = torch.nn.MSELoss()

            train_start_time = time.time()
            for epoch in range(1, args.epochs):
                self._train(train_loader, model, criterion, optimizer, epoch)
            train_end_time = time.time()
            self.train_cost += train_end_time - train_start_time
            infer_start_time = time.time()
            self._test(model, test_index)
            temp_coverage_shap, temp_hand_shap = self._shap(model, test_index)
            coverage_shap += temp_coverage_shap
            handcrafted_shap += temp_hand_shap
            infer_end_time = time.time()
            self.infer_cost += infer_end_time - infer_start_time
        coverage_shap /= k
        handcrafted_shap /= k
        program_bug_id = f"{self.program}-{self.bug_id}"
        # 2. 格式化SHAP值（保留4位小数，确保数值可读性）
        coverage_str = f"{coverage_shap:.4f}"
        handcrafted_str = f"{handcrafted_shap:.4f}"

        # 3. 组合一行数据（用空格分隔各字段）
        line = f"{program_bug_id} {coverage_str} {handcrafted_str}\n"

        # 4. 写入文件（使用追加模式，避免覆盖已有内容）
        with open("../../results/shap_results.txt", "a", encoding="utf-8") as f:
            f.write(line)


    def _train(self, train_loader, model, criterion, optimizer, epoch):
        model.train()
        for batch_idx, (test, target, ssp, cr, sf) in enumerate(train_loader):
            if args.cuda:
                test, target, ssp, cr, sf = test.cuda(), target.cuda(),ssp.cuda(), cr.cuda(), sf.cuda()
            test = test.to(torch.float)
            ssp = ssp.to(torch.float)
            cr = cr.to(torch.float)
            sf = sf.to(torch.float)
            ef = torch.hstack((ssp, cr, sf))
            # target = target.float()
            # aug_test = aug_test.to(torch.float)
            if test.size(0) == 1:
                test = test.repeat(2, 1)
                target = target[0].repeat(2)
                # target = target.repeat(2, 1)
                ef = ef.repeat(2,1)

            # prob = model(test,ef)

            x = torch.hstack((test, ef))
            # prob = model(test, ef)
            prob = model(x)

            loss = criterion(prob, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 30 == 0:
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

                x = torch.hstack((test,ef))
                # prob = model(test, ef)
                prob = model(x)
                if prob[0][0] < prob[0][1]:
                    self.cc_index.iloc[item] = True

    def _shap(self, model, test_index):
        test = self.passing_tests.iloc[test_index, :-1]
        ssp = self.ssp.iloc[test_index]
        cr = self.cr.iloc[test_index]
        sf = self.sf.iloc[test_index]

        test = torch.tensor(test.values)
        ssp = torch.tensor(ssp.values)
        cr = torch.tensor(cr.values)
        sf = torch.tensor(sf.values)
        if args.cuda:
            test, ssp, cr, sf = test.cuda(), ssp.cuda(), cr.cuda(), sf.cuda()
        test = test.to(torch.float)
        ssp = ssp.to(torch.float)
        cr = cr.to(torch.float)
        sf = sf.to(torch.float)
        hf = torch.hstack([ssp, cr, sf])
        X_combined = torch.hstack([test, hf])
        n_background = min(X_combined.shape[0], 50)
        background_indices = np.random.choice(X_combined.shape[0], n_background, replace=False)
        background = X_combined[background_indices]  # 背景数据是张量
        explainer = shap.GradientExplainer(model, background)

        sample_size = X_combined.shape[0]
        sample_indices = np.random.choice(X_combined.shape[0], sample_size, replace=False)
        X_sample = X_combined[sample_indices]

        shap_values = explainer.shap_values(X_sample)  # 输出：(n_samples, n_features,2)，对应两个类别\
        n_test = test.shape[1]

        depth_shap = np.mean(np.abs(np.sum(shap_values[:,:n_test,1], axis= 1)))

        handcraft_shap = np.mean(np.abs(np.sum(shap_values[:,n_test:,1], axis= 1)))

        # # 7. 输出重要性结果
        # print(f"深度提取特征组平均SHAP值：{depth_shap:.4f}")
        # print(f"手工特征组平均SHAP值：{handcraft_shap:.4f}")
        # if depth_shap > handcraft_shap:
        #     print("结论：深度提取特征对模型决策的影响更大")
        # else:
        #     print("结论：手工特征对模型决策的影响更大")
        return depth_shap, handcraft_shap

