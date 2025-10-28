import multiprocessing
from collections import Counter

import numpy as np
import pandas as pd
import torch
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler

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

    def get_failing_tests(self):
        self.failing_tests = FailingTestsHandler.get_failing_tests(self.data_df)
        return len(self.failing_tests)

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
        # 提前计算索引
        indices = (train_target == torch.tensor([0, 1], dtype=torch.float32)).all(dim=1)
        index_np = indices.numpy()
        sub_train_target = train_target[indices]
        sub_ssp, sub_cr, sub_sf = ssp[index_np], cr[index_np], sf[index_np]
        sub_train_tests = train_tests[index_np]
        # 存储待拼接的数据
        train_tests_list = [train_tests]
        train_target_list = [train_target]
        ssp_list = [ssp]
        cr_list = [cr]
        sf_list = [sf]

        for CCE in self.CCE[:-1]:
            new_sub_train_tests = sub_train_tests.copy()
            columns_to_zero = [col for col in sub_train_tests.columns if col not in CCE]
            new_sub_train_tests[columns_to_zero] = 0

            train_tests_list.append(new_sub_train_tests)
            train_target_list.append(sub_train_target)
            ssp_list.append(sub_ssp)
            cr_list.append(sub_cr)
            sf_list.append(sub_sf)
        # 一次性拼接所有数据
        train_tests = pd.concat(train_tests_list, ignore_index=True)
        train_target = torch.cat(train_target_list, dim=0)
        ssp = pd.concat(ssp_list, ignore_index=True)
        cr = pd.concat(cr_list, ignore_index=True)
        sf = pd.concat(sf_list, ignore_index=True)
        return train_tests, train_target, ssp, cr, sf

    def data_augmentation_under_imblearn(self, train_tests, train_target, ssp, cr, sf, imbalance_method='smote'):
        train_labels = np.argmax(train_target, axis=1)
        unique_classes = np.unique(train_labels)
        if len(unique_classes) < 2:
            print(f"警告: 训练数据中只有一个类别 ({unique_classes[0]})，跳过重采样")
            return train_tests, train_target, ssp, cr, sf

        if imbalance_method == 'undersample':
            # 随机欠采样
            sampler = RandomUnderSampler(random_state=42)
        elif imbalance_method == 'oversample':
            # 随机过采样
            sampler = RandomOverSampler(random_state=42)
        else:  # 默认使用SMOTE
            # SMOTE过采样
            # 统计类别分布，找到少数类样本数
            class_counts = Counter(train_labels.numpy())
            minority_count = min(class_counts.values())  # 少数类样本数量
            # 计算安全的k_neighbors（需小于少数类样本数，且不超过5）
            if minority_count <= 1:
                # 少数类样本太少，SMOTE效果有限，强制设置最小邻居数
                sampler = RandomOverSampler(random_state=42)
            else:
                k_neighbors = min(5, minority_count - 1)  # 确保邻居数小于少数类样本数
                sampler = SMOTE(random_state=42, k_neighbors=k_neighbors)
        n_cols_train = train_tests.shape[1]
        all_features = np.hstack([train_tests, ssp, cr, sf])
        train_target = np.argmax(train_target, axis=1)
        all_features_resampled, train_target_resampled = sampler.fit_resample(all_features, train_target)
        train_tests_resampled = all_features_resampled[:, :n_cols_train]
        ssp_resampled = all_features_resampled[:, n_cols_train:n_cols_train + 10]
        cr_resampled = all_features_resampled[:, n_cols_train + 10:n_cols_train + 2 * 10]
        sf_resampled = all_features_resampled[:, n_cols_train + 2 * 10:n_cols_train + 3 * 10]
        train_target_resampled = np.eye(2)[train_target_resampled]
        return train_tests_resampled, train_target_resampled, ssp_resampled, cr_resampled, sf_resampled