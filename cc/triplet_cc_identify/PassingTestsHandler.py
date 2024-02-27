import numpy as np
import pandas as pd

from cc.triplet_cc_identify.Features.Features import Features
from sklearn.preprocessing import StandardScaler


def getfT(data):
    uncover = sum(data == 0)
    cover = sum(data == 1)
    fT = cover / (uncover + cover)
    return fT


def getpT(data):
    uncover = sum(data == 0)
    cover = sum(data == 1)
    pT = cover / (uncover + cover)
    return pT


def _is_CCE( fail_data, pass_data, cita):
    fT = getfT(fail_data)
    pT = getpT(pass_data)
    if ((fT == 1.0) and (pT < cita)):
        return True
    else:
        return False
class PassingTestsHandler:
    def __init__(self):
        pass

    @staticmethod
    def get_passing_tests(data_df):
        return data_df[data_df["error"] == 0]

    @staticmethod
    def get_true_passing_tests(data_df, threshold):
        passing_df = PassingTestsHandler.get_passing_tests(data_df)
        passing_num = len(passing_df)
        if passing_num <= 1:
            return passing_df, None

        features = Features(data_df)

        features_df = features.getAllFeatures()

        standardScaler = StandardScaler()
        standardScaler.fit(features_df)
        feature_standard = standardScaler.transform(features_df)

        # threshold = [0.5, 0.5, 0.5]
        vote_matrix = pd.DataFrame(feature_standard <= threshold, dtype=int, index=features_df.index)
        vote_result_matrix = vote_matrix.sum(axis=1)
        true_passing_result = vote_result_matrix[vote_result_matrix >= len(threshold) / 2]
        cc_candidate_result = vote_result_matrix[vote_result_matrix < len(threshold) / 2]

        true_passing_list = list(true_passing_result.index)
        cc_candidate_list = list(cc_candidate_result.index)

        return data_df.iloc[true_passing_list, :].astype('float32'), data_df.iloc[cc_candidate_list, :].astype(
            'float32')

    @staticmethod
    def get_TP_by_Tech_1(data_df, cita):
        failing_df = data_df[data_df["error"] == 1]
        passing_df = data_df[data_df["error"] == 0]
        CCE = []
        for i in failing_df.columns:
            if i != "error":
                if _is_CCE(failing_df[i], passing_df[i], cita):
                    CCE.append(i)

        new_data_df = passing_df[CCE]
        sum_df = new_data_df.sum(axis=1)
        cc_candidate_list = list(sum_df[sum_df > 0].index)
        true_passing_list = list(sum_df[sum_df == 0].index)
        return data_df.iloc[true_passing_list, :].astype('float32'), data_df.iloc[cc_candidate_list, :].astype(
            'float32')

    @staticmethod
    def getStandardization(dict):
        dict_mean = np.mean(list(dict.values()))
        dict_std = np.std(list(dict.values()))
        for item in dict:
            dict[item] = (dict[item] - dict_mean) / dict_std
        return dict
