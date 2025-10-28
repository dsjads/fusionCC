from CONFIG import program_list
from cc.cc_baselines.BaseCCPipeline import BaseCCPipeline
from cc.core import run
from cc.fusion_cc_identify.BaseIdentify import BaseIdentify
from cc.fusion_cc_identify.ContrastiveIdentify import ContrastiveIdentify
from cc.fusion_cc_identify.FusionIdentifyOversampling import FusionIdentifyOversampling
from cc.fusion_cc_identify.FusionIdentifyWithoutCov import FusionIdentifyWithoutCovInfo
from cc.fusion_cc_identify.FusionIdentifyWithoutHF import FusionIdentifyWithoutExpertFeature
from cc.fusion_cc_identify.FusionIdentify import FusionIdentify


def main():
    # program_list = ["Chart", "Lang", "Math", "Mockito", "Time", "Closure"]
    program_list = ["gzip","libtiff","python","space"]
    arg_dict = {
        "cce_threshold": [0.6, 0.7, 0.8, 0.9, 1],
        "select_ratio": [i / 100 for i in range(5, 31, 5)],
        "sus_threshold": [i / 100 for i in range(50, 91, 5)]
    }
    name = "2025-fusion-concat"

    run(program_list, "gzip", 2 , BaseIdentify, name, arg_dict)


if __name__ == "__main__":
    main()
