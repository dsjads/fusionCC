from cc.core import run
from cc.fusion_cc_identify.BaseIdentify import BaseIdentify
from cc.fusion_cc_identify.FusionIdentify import FusionIdentify
from cc.fusion_cc_identify.FusionIdentifyWithoutCovInfo import FusionIdentifyWithoutCovInfo
from cc.fusion_cc_identify.FusionIdentifyWithoutExpertFeature import FusionIdentifyWithoutExpertFeature
from cc.fusion_cc_identify.SupContraCCIdentify import SupContraCCIdentify
from cc.fusion_cc_identify.SupContraCCIdentify2 import SupContraCCIdentify2


def main():
    # program_list = [
    #     "Chart",
    # "Lang",
    # "Math",
    # "Mockito",
    # "Time"
    # ]
    # program_list = ["Chart"]
    program_list = ["Chart", "Lang", "Math", "Mockito", "Time"]

    arg_dict = {
        # "cce_threshold":[i/100 for i in range(60, 91, 5)],
        "cce_threshold": [0.6, 0.7, 0.8, 0.9, 1],
        "select_ratio": [i / 100 for i in range(5, 31, 5)],
        "sus_threshold": [i / 100 for i in range(50, 91, 5)]
    }
    name = "2025_3-13-fusion-withoutCBAM"
    run(program_list, "Chart", 1 , SupContraCCIdentify2 , name, arg_dict)


if __name__ == "__main__":
    main()
