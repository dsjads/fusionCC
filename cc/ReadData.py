import os
import copy
import sys
from read_data.ManyBugsDataLoader import ManyBugsDataLoader
from read_data.Defects4JDataLoader import Defects4JDataLoader
from read_data.SIRDataLoader import SIRDataLoader
from CONFIG import *

class ReadData:
    def __init__(self, project_dir, configs):
        self.configs = configs
        self.project_dir = project_dir
        self.dataset = configs["-d"]
        self.program = configs["-p"]
        self.bug_id = configs["-i"]
        self.method = configs["-m"].split(",")
        self.dataloader = self._choose_dataloader_obj()
        self.data_obj = copy.deepcopy(self.dataloader)

        # 复制数据，修改数据时，只使用data.obj，dataloader作为源数据备份
        self.data_df = self.load_data()

    def load_data(self):
        self.data_obj = copy.deepcopy(self.dataloader)
        return self.data_obj.data_df

    def _dynamic_choose(self, loader):
        self.dataset_dir = os.path.join(self.project_dir, "..", "data")
        data_obj = loader(self.dataset_dir, self.program, self.bug_id)
        data_obj.load()
        return data_obj

    def _choose_dataloader_obj(self):
        if self.dataset == "d4j":
            return self._dynamic_choose(Defects4JDataLoader)
        if self.dataset == "manybugs" or self.dataset == "motivation":
            return self._dynamic_choose(ManyBugsDataLoader)
        if self.dataset == "SIR":
            return self._dynamic_choose(SIRDataLoader)

if __name__ == "__main__":

    configs = {'-d': 'd4j', '-p': 'Chart', '-i': '0', '-m': 'dstar', '-e': 'origin'}
    sys.argv = ["ReadData.py"]
    gpl = ReadData(project_dir, configs)
    gpl.load_data()
    a = 1
