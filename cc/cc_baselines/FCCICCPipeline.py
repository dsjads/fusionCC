import os
import sys

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from CONFIG import cc_info, project_dir, method_para
from Features import Features
from cc.cc_baselines.BaseCCPipeline import BaseCCPipeline
from cc.fusion_cc_identify.FeatureTestsHandler import FeatureTestsHandler
from cc.mlcci_identify.ReadTrainData import ReadTrainData
import numpy as np
from skfuzzy import control as ctrl, gaussmf

def build_expert_model():
    SS = ctrl.Antecedent(np.arange(0, 1, 0.01), 'SS')
    SF = ctrl.Antecedent(np.arange(0, 1, 0.01), 'SF')
    CR = ctrl.Antecedent(np.arange(0, 1, 0.01), 'CR')
    CC = ctrl.Consequent(np.arange(0, 1, 0.01), 'CC')

    centers = [0,0.25,0.5,0.75,1]
    sigma = 0.01
    for i, center in enumerate(centers):
        label = ['poor', 'mediocre', 'average', 'decent', 'good'][i]
        SS[label] = gaussmf(SS.universe, center, sigma)
        SF[label] = gaussmf(SF.universe, center, sigma)
        CR[label] = gaussmf(CR.universe, center, sigma)
        CC[label] = gaussmf(CC.universe, center, sigma)

    rules = [
        ctrl.Rule(SS['poor'], CC['poor']),
        ctrl.Rule(SS['mediocre'], CC['mediocre']),
        ctrl.Rule(SS['average'], CC['average']),
        ctrl.Rule(SS['decent'], CC['decent']),
        ctrl.Rule(SS['good'], CC['good']),

        ctrl.Rule(CR['poor'], CC['poor']),
        ctrl.Rule(CR['mediocre'], CC['mediocre']),
        ctrl.Rule(CR['average'], CC['average']),
        ctrl.Rule(CR['decent'], CC['decent']),
        ctrl.Rule(CR['good'], CC['good']),

        ctrl.Rule(SF['poor'], CC['poor']),
        ctrl.Rule(SF['mediocre'], CC['mediocre']),
        ctrl.Rule(SF['average'], CC['average']),
        ctrl.Rule(SF['decent'], CC['decent']),
        ctrl.Rule(SF['good'], CC['good'])
    ]

    activity_ctrl = ctrl.ControlSystem(rules)
    activity_simulation = ctrl.ControlSystemSimulation(activity_ctrl)
    return activity_simulation


class FCCIIdentify(BaseCCPipeline):
    def __init__(self, project_dir, configs, cita, way,K):
        super().__init__(project_dir, configs, way)
        self.config_list = []
        self.test_part_list=[]
        self.cc_target = None

    def _find_cc_index(self):
        program = self.configs['-p']
        info_list = cc_info[program]
        program_len = len(info_list)
        program_method = self.configs['-m']
        model = build_expert_model()
        for i in range(program_len):
            config = {'-d': 'd4j', '-p': program, '-i': str(info_list[i]), '-m': program_method,
                      '-e': 'origin'}
            self.config_list.append(config)
        # pool = Pool(8)
        train_rtd = ReadTrainData(self.project_dir, self.config_list, self.way)
        for ccpl in train_rtd.ccpls:
            self.ssp, self.cr, self.sf = self.get_feature_tests(ccpl.data_df)

            for i in range(len(self.ssp)):
                ss = self.ssp[i][0]
                cr = self.cr[i][0]
                sf = self.sf[i][0]
                model.input['SS'] = ss
                model.input['SF'] = cr
                model.input['CR'] = sf
                model.compute()
                prob = model.output['CC']
                if prob > 0.5:
                    ccpl.cc_index.iloc[i] = True

            ccpl.evaluation()
            ccpl.calRes("relabel")
            ccpl.calRes("trim")


    def get_feature_tests(self, data_df):
        features = Features(data_df)
        cr = features.covRatio()
        ssp = features.suspScore()
        sf = features.similarityFactor()

        # 创建三个不同的 MinMaxScaler 实例
        scaler_ssp = MinMaxScaler(feature_range=(0, 1))
        scaler_cr = MinMaxScaler(feature_range=(0, 1))
        scaler_sf = MinMaxScaler(feature_range=(0, 1))

        # 分别对每个特征进行拟合和转换
        ssp_standard = scaler_ssp.fit_transform(ssp)
        cr_standard = scaler_cr.fit_transform(cr)
        sf_standard = scaler_sf.fit_transform(sf)

        return ssp_standard, cr_standard, sf_standard

if __name__ == "__main__":
    program_list=["Chart", "Lang", "Math", "Mockito", "Time.csv"]
    for program in program_list:
        configs = {'-d': 'd4j', '-p': program, '-i': '1', '-m': method_para, '-e': 'origin'}
        sys.argv = os.path.basename(__file__)
        cbccpl = FCCIIdentify(project_dir, configs, 1, "MLCCI", 5)
        cbccpl.find_cc_index()
    a = 1