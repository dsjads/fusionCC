from cc.cc_baselines.BaseCCPipeline import BaseCCPipeline


class BaseIdentify(BaseCCPipeline):
    def __init__(self, project_dir, configs, args_dict, way):
        super().__init__(project_dir, configs, way)
        self.CCT = None
        self.CCE = None
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
