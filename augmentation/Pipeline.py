import os
import sys

from read_data.ManyBugsDataLoader import ManyBugsDataLoader
from read_data.Defects4JDataLoader import Defects4JDataLoader
from read_data.SIRDataLoader import SIRDataLoader
from augmentation.data_systhesis.resampling import ResamplingData
from augmentation.data_systhesis.smote import SMOTEData
from augmentation.data_systhesis.cvae_synthesis import CVAESynthesisData
from augmentation.dimensional_reduciton.PCA import PCAData
from augmentation.data_undersampling.undersampling import UndersamplingData
from fl_evaluation.calculate_suspiciousness.CalculateSuspiciousness import CalculateSuspiciousness


class Pipeline:
    def __init__(self, project_dir, configs):
        self.configs = configs
        self.project_dir = project_dir
        self.dataset = configs["-d"]
        self.program = configs["-p"]
        self.bug_id = configs["-i"]
        self.experiment = configs["-e"]
        self.method = configs["-m"].split(",")
        self.dataloader = self._choose_dataloader_obj()     # dataloader为原始数据
        self.data_obj = None    # data_obj 为最终使用的数据

    def run(self):
        self._run_task()

    def _choose_dataloader_obj(self):
        if self.dataset == "d4j":
            return self._dynamic_choose(Defects4JDataLoader)
        if self.dataset == "manybugs" or self.dataset == "motivation":
            return self._dynamic_choose(ManyBugsDataLoader)
        if self.dataset == "SIR":
            return self._dynamic_choose(SIRDataLoader)

    def _dynamic_choose(self, loader):
        self.dataset_dir = os.path.join(self.project_dir, "..", "data")
        data_obj = loader(self.dataset_dir, self.program, self.bug_id)
        data_obj.load()
        return data_obj

    def _run_task(self):
        if self.experiment == "origin":
            self.data_obj = self.dataloader
        elif self.experiment == "resampling":
            self.data_obj = ResamplingData(self.dataloader)
            self.data_obj.process()
        elif self.experiment == "undersampling":
            self.data_obj = UndersamplingData(self.dataloader)
            self.data_obj.process()
        elif self.experiment == "smote":
            self.data_obj = SMOTEData(self.dataloader)
            self.data_obj.process()
        elif self.experiment == "cvae":
            self.data_obj = CVAESynthesisData(self.dataloader)
            self.data_obj.process()
        elif self.experiment == "fs":
            cp = float(self.configs["-cp"])
            ep = float(self.configs["-ep"])
            self.data_obj = PCAData(self.dataloader)
            self.data_obj.process(cp, ep)
        elif self.experiment == "fs_cvae":
            cp = float(self.configs["-cp"])
            ep = float(self.configs["-ep"])
            self.data_obj = PCAData(self.dataloader)
            self.data_obj.process(cp, ep)
            self.data_obj = CVAESynthesisData(self.data_obj)
            self.data_obj.process()

        save_rank_path = os.path.join(self.project_dir, "results")
        cc = CalculateSuspiciousness(self.data_obj, self.method, save_rank_path, self.experiment)
        cc.run()


if __name__ == "__main__":
    # main()
    # task_complete("Triplet CC end")
    from CONFIG import *
    configs = {'-d': 'd4j', '-p': 'Chart', '-i': '0', '-m': 'dstar', '-e': 'origin'}
    # configs = {'-d': 'd4j', '-p': 'Chart', '-i': '0', '-m': 'dstar', '-e': 'fs', '-cp': 0.5, '-ep': 0.5}
    sys.argv = [os.path.basename(__file__)]
    # CCSurveyPipeline 继承了ReadData，所以，最开始的时候，就加载了原始数据，和原始数据的备份
    ppl = Pipeline(project_dir, configs)
    ppl.run()
    # task_complete("precision end")