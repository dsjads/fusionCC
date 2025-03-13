import numpy as np
import pandas as pd
import torch

from cc.cc_baselines.BaseCCPipeline import BaseCCPipeline
from cc.fusion_cc_identify.FailingTestsHandler import FailingTestsHandler
from cc.fusion_cc_identify.PassingTestsHandler import PassingTestsHandler


def append_to_excel(version, c1, c2, c3, c4):
    file_path = "augmentation_results.xlsx"
    data = {
        'version': [version],
        'c1': [c1],
        'c2': [c2],
        'c3': [c3],
        'c4': [c4]
    }
    new_df = pd.DataFrame(data)

    try:
        # 尝试读取现有的 Excel 文件
        existing_df = pd.read_excel(file_path)
        # 将新的 DataFrame 追加到现有的 DataFrame 中
        result_df = pd.concat([existing_df, new_df], ignore_index=True)
    except FileNotFoundError:
        # 如果文件不存在，就使用新的 DataFrame
        result_df = new_df

        # 将结果保存到 Excel 文件中
    result_df.to_excel(file_path, index=False)


class BaseIdentify(BaseCCPipeline):
    def __init__(self, project_dir, configs, args_dict, way):
        super().__init__(project_dir, configs, way)
        self.CCT = None
        self.CCE = []
        self.feature = None
        self.args_dict = args_dict
        self.cita = None
        self.true_passing_tests = None
        self.failing_tests = None
        self.sus_dict = {}
        self.train_flag = True

    def _getfT(self, data):
        uncover = sum(data == 0)
        cover = sum(data == 1)
        fT = cover / (uncover + cover)
        return fT

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

        count1_01 = torch.sum((torch.eq(self.cc_target, torch.tensor([0, 1]))).all(dim=1)).item()
        count1_10 = torch.sum((torch.eq(self.cc_target, torch.tensor([1, 0]))).all(dim=1)).item()

        indices = np.array(self.train_tests.index)
        train_index = self.passing_tests.index.get_indexer(indices)
        train_target = self.cc_target[train_index]
        # count1_10 = torch.sum((torch.eq(train_target, torch.tensor([1, 0]))).all(dim=1)).item()
        if count1_10 > 0 and count1_01 / count1_10 < 0.2:
            self.train_tests, train_target = self.data_augmentation(self.train_tests, train_target)

        count2_01 = torch.sum((torch.eq(train_target, torch.tensor([0, 1]))).all(dim=1)).item()
        count2_10 = torch.sum((torch.eq(train_target, torch.tensor([1, 0]))).all(dim=1)).item()

        append_to_excel(self.program + str(self.bug_id), count1_01, count1_10, count2_01, count2_10)

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
        for cita in self.cita:
            CCE = []
            for i in failing_df.columns:
                if i != "error":
                    if self._is_CCE(failing_df[i], passing_df[i], cita):
                        CCE.append(i)
            self.CCE.append(CCE)

    def data_augmentation(self, train_tests, train_target):
        # resampling
        indices = (train_target == torch.tensor([0, 1], dtype=torch.float32)).all(dim=1)
        sub_train_target = train_target[indices]
        sub_train_tests = train_tests[indices.numpy()]
        for CCE in self.CCE[:-1]:
            new_sub_train_tests = sub_train_tests.copy()
            columns_to_zero = [col for col in sub_train_tests.columns if col not in CCE]
            new_sub_train_tests[columns_to_zero] = 0
            train_tests = pd.concat([train_tests, new_sub_train_tests], ignore_index=True)
            train_target = torch.cat((train_target, sub_train_target), dim=0)
        return train_tests, train_target

    def data_augmentation_with_ef(self, train_tests, train_target, ssp, cr, sf):
        # resampling
        indices = (train_target == torch.tensor([0, 1], dtype=torch.float32)).all(dim=1)
        sub_train_target = train_target[indices]
        sub_ssp, sub_cr, sub_sf = ssp[indices.numpy()], cr[indices.numpy()], sf[indices.numpy()]
        sub_train_tests = train_tests[indices.numpy()]
        for CCE in self.CCE[:-1]:
            new_sub_train_tests = sub_train_tests.copy()
            columns_to_zero = [col for col in sub_train_tests.columns if col not in CCE]
            new_sub_train_tests[columns_to_zero] = 0
            train_tests = pd.concat([train_tests, new_sub_train_tests], ignore_index=True)
            train_target = torch.cat((train_target, sub_train_target), dim=0)
            ssp = pd.concat([ssp, sub_ssp],ignore_index=True)
            cr = pd.concat([cr, sub_cr],ignore_index=True)
            sf = pd.concat([sf,sub_sf],ignore_index=True)
        return train_tests, train_target, ssp, cr, sf
