from cc.cc_baselines.BaseCCPipeline import BaseCCPipeline


class ReadTrainData:
    def __init__(self, project_dir, configs, way):
        self.ccpls=[]
        for config in configs:
            ccpl=BaseCCPipeline(project_dir, config, way)
            ccpl.init_cc_index()
            self.ccpls.append(ccpl)

class ReadTestData:
    def __init__(self, project_dir, config, way):
        self.ccpl = BaseCCPipeline(project_dir, config, way)
        self.ccpl.init_cc_index()