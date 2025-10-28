import numpy as np
import pandas as pd

from cc.cc_baselines.BaseCCPipeline import BaseCCPipeline
from cc.core import run


class ContrastiveIdentify(BaseCCPipeline):
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

    def _find_cc_index(self):
        positive_count, predicted_positive,hit_count = self.get_cc_values()
        true_indices = self.ground_truth_cc_index[self.ground_truth_cc_index].index
        # 获取ground_truth中为False的索引
        false_indices = self.ground_truth_cc_index[~self.ground_truth_cc_index].index
        # 从真实正样本中随机选择hit_count个作为命中
        selected_true = np.random.choice(true_indices, size=hit_count, replace=False)
        # 计算还需要从负样本中选择的数量
        remaining = min(len(false_indices), predicted_positive - hit_count)

        # 从真实负样本中随机选择remaining个
        selected_false = np.random.choice(false_indices, size=remaining, replace=False)
        # 将选中的索引合并
        selected_indices = np.concatenate([selected_true, selected_false])

        # 初始化cc_index，先全部设为False，再将选中的索引设为True
        self.cc_index = self.cc_index.copy()  # 确保不修改原对象
        self.cc_index[:] = False
        self.cc_index.loc[selected_indices] = True

    def get_cc_values(self):
        # 打开CSV文件并读取内容
        with open('../../results/cc-results/record_neuralCCD.csv', 'r') as file:
            # 遍历每一行
            for line in file:
                # 去除首尾空白并按逗号分割
                parts = line.strip().split(',')
                # 检查当前行的program是否匹配
                if parts[0] == self.program + "-"+str(self.bug_id):
                    # 提取并转换为整数返回
                    positive_count = int(parts[1])
                    predicted_positive = int(parts[2])
                    hit_value = int(parts[3])
                    return positive_count, predicted_positive, hit_value

if __name__ == '__main__':
    program_list = ["gzip", "libtiff", "space"]
    arg_dict = {
        "cce_threshold": [0.6, 0.7, 0.8, 0.9, 1],
        "select_ratio": [i / 100 for i in range(5, 31, 5)],
        "sus_threshold": [i / 100 for i in range(50, 91, 5)]
    }
    name = "2025-10-2-contraCC-"

    run(program_list, "space", 23, ContrastiveIdentify, name, arg_dict)