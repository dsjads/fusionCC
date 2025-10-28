import math

import numpy as np
import pandas as pd
# from cc.triplet_cc_identify.PassingTestsHandler import PassingTestsHandler
from fl_evaluation.metrics.calc_corr import calc_corr
from CONFIG import *
from fl_evaluation.metrics.metrics2 import *
from fl_evaluation.metrics.new_calc_corr import new_calc_corr
from read_data.Defects4JDataLoader import Defects4JDataLoader


class Features:

    def __init__(self, data_df):
        self.data_df = data_df
        self.suspicious_list = [ new_calc_corr(self.data_df, DStar()), new_calc_corr(self.data_df,Ochiai()),
                                 new_calc_corr(self.data_df,Barinel()),new_calc_corr(self.data_df,ER1()),
                                 new_calc_corr(self.data_df,ER5()), new_calc_corr(self.data_df,GP02()),
                                 new_calc_corr(self.data_df,GP03()), new_calc_corr(self.data_df,GP19()),
                                 new_calc_corr(self.data_df,Jaccard()),new_calc_corr(self.data_df,Op2())]
        # self.suspicious_list = [
        #     new_calc_corr(self.data_df, DStar()), new_calc_corr(self.data_df, DStarSub1()),
        #     new_calc_corr(self.data_df,Ochiai()), new_calc_corr(self.data_df, OchiaiSubOne()),
        #     new_calc_corr(self.data_df, OchiaiSubTwo()), new_calc_corr(self.data_df, GP13()),
        #     new_calc_corr(self.data_df, GP13_sub_one()), new_calc_corr(self.data_df, GP13_sub_two()),
        #     new_calc_corr(self.data_df, Op2()), new_calc_corr(self.data_df, Op2_sub_one()),
        #     new_calc_corr(self.data_df, Op2_sub_two()), new_calc_corr(self.data_df, Jaccard()),
        #     new_calc_corr(self.data_df, Jaccard_sub_one()),new_calc_corr(self.data_df, Russell()),
        #     new_calc_corr(self.data_df, Russell_sub_one()),new_calc_corr(self.data_df, Tarantula()),
        #     new_calc_corr(self.data_df, Tarantula_sub_one()), new_calc_corr(self.data_df, Naish1()),
        #     new_calc_corr(self.data_df, Binary()), new_calc_corr(self.data_df, CrossTab()),
        # ]
        self.features = self.data_df.iloc[:, :-1]
        self.passing_features = self.get_passing_tests(self.data_df).iloc[:, :-1]
        self.failing_features = self.get_failing_tests(self.data_df).iloc[:, :-1]
        self.ssp = dict()
        self.cr = dict()
        self.sf = dict()

    def get_passing_tests(self, data_df):
        return data_df[data_df["error"] == 0]

    def get_failing_tests(self, data_df):
        return data_df[data_df["error"] == 1]

    def getAllFeatures(self):
        ssp = self.suspScore()
        cr = self.covRatio()
        sf = self.similarityFactor()

        ssp_list = np.array(list(ssp.values())).reshape(-1, 1)
        cr_list = np.array(list(cr.values())).reshape(-1, 1)
        sf_list = np.array(list(sf.values())).reshape(-1, 1)
        comb_list = np.concatenate((ssp_list, cr_list, sf_list), axis=1)
        featuresDf = pd.DataFrame(comb_list, index=list(self.ssp.keys()))
        return featuresDf

    def suspScore(self):
        columns = self.passing_features.columns
        # 预构建高/低可疑值矩阵
        high_suspicious_masks = []
        high_suspicious_values = []
        low_suspicious_masks = []
        low_suspicious_values = []
        for item in self.suspicious_list:
            high_mask = np.zeros(len(columns), dtype=bool)
            high_values = np.zeros(len(columns))
            low_mask = np.zeros(len(columns), dtype=bool)
            low_values = np.zeros(len(columns))
            for j, col in enumerate(columns):
                if col in item:
                    s = item[col]
                    if 0.5 <= s <= 1.0:
                        high_mask[j] = True
                        high_values[j] = s
                    elif 0 <= s < 0.5:
                        low_mask[j] = True
                        low_values[j] = s
            high_suspicious_masks.append(high_mask)
            high_suspicious_values.append(high_values)
            low_suspicious_masks.append(low_mask)
            low_suspicious_values.append(low_values)
        passing_array = self.passing_features.values
        ssp_list = []
        for i in range(passing_array.shape[0]):
            row = passing_array[i]
            row_ssp = []
            h_cnt_total = 0
            h_ssum_total = 0
            l_cnt_total = 0
            l_ssum_total = 0
            for j in range(len(self.suspicious_list)):
                # 高可疑值计算
                high_covered = row & high_suspicious_masks[j]
                h_cnt_current = np.sum(high_covered)
                h_ssum_current = np.sum(high_covered * high_suspicious_values[j])
                low_covered = row & low_suspicious_masks[j]
                l_cnt_current = np.sum(low_covered)
                l_ssum_current = np.sum(low_covered * low_suspicious_values[j])
                h_cnt_total += h_cnt_current
                h_ssum_total += h_ssum_current
                l_cnt_total += l_cnt_current
                l_ssum_total += l_ssum_current
                if h_cnt_total == 0 and l_cnt_total != 0:
                    row_ssp.append(l_ssum_total / l_cnt_total)
                elif h_cnt_total != 0:
                    row_ssp.append(h_ssum_total / h_cnt_total)
                else:
                    row_ssp.append(0)
            ssp_list.append(row_ssp)
        return np.array(ssp_list)

    def covRatio(self):
        S_cnt = len(self.suspicious_list)
        # 预计算每个可疑值公式对应的有效语句集合
        valid_statements_per_formula = []
        for item in self.suspicious_list:
            valid_lines = set()
            for line_num, s in item.items():
                if 0.5 <= s <= 1.0:
                    valid_lines.add(line_num)
            valid_statements_per_formula.append(valid_lines)
        cr_list = []
        for row_index, row in self.passing_features.iterrows():
            row_cr = []
            # 对每个可疑值公式计算覆盖比例
            for valid_lines in valid_statements_per_formula:
                # 计算当前行覆盖的有效语句数量
                h_cnt = sum(1 for line_num in valid_lines if row.loc[line_num] == 1)
                row_cr.append(h_cnt / S_cnt)
            cr_list.append(row_cr)
        return np.array(cr_list)

    def similarityFactor(self):
        columns = self.passing_features.columns
        passing_array = np.nan_to_num(self.passing_features.values, nan=0.0, posinf=0.0, neginf=0.0)
        failing_array = np.nan_to_num(self.failing_features.values, nan=0.0, posinf=0.0, neginf=0.0)

        sf_list = []

        for item in self.suspicious_list:
            susp_values = np.array([item.get(col, 0) for col in columns])
            susp_values = np.nan_to_num(susp_values, nan=0.0, posinf=0.0, neginf=0.0)
            susp_values = np.maximum(susp_values, 0)

            # 应用权重
            weighted_passing = passing_array * susp_values
            weighted_failing = failing_array * susp_values

            # 使用矩阵运算公式：||a-b||² = ||a||² + ||b||² - 2a·b
            passing_norm = np.sum(weighted_passing ** 2, axis=1)  # (m_p,)
            failing_norm = np.sum(weighted_failing ** 2, axis=1)  # (m_f,)
            dot_product = np.dot(weighted_passing, weighted_failing.T)  # (m_p, m_f)

            # 计算平方距离：passing_norm[:, None] + failing_norm[None, :] - 2*dot_product
            # 但为了避免大矩阵，我们逐行处理
            min_distances = np.zeros(len(weighted_passing))

            for i in range(len(weighted_passing)):
                squared_distances = passing_norm[i] + failing_norm - 2 * dot_product[i, :]
                # 处理数值误差导致的负值
                squared_distances = np.maximum(squared_distances, 0)
                min_squared_distance = np.min(squared_distances)
                min_distance = np.sqrt(min_squared_distance) if min_squared_distance > 0 else 0

                if min_distance <= 1e-8:
                    min_distance = 1e-8

                min_distances[i] = min_distance

            sf_features = np.where(min_distances > 1e-8, 1.0 / min_distances, 1e8)
            sf_list.append(sf_features)

        return np.array(sf_list).T

    def faultMaskingFactor(self):
        pass


if __name__ == "__main__":
    data = Defects4JDataLoader(os.path.join(project_dir, '..', 'data'), "Chart", "1")
    data.load()
    features = Features(data.data_df)
    ssp = features.suspScore()
    cr = features.covRatio()
    sf = features.similarityFactor()
    print(ssp, cr, sf)

    # features.covRatio()
    # features.similarityFactor()
    a = 1
