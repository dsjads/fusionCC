from cc.core import run
from cc.fusion_cc_identify.FusionIdentify import FusionIdentify
from cc.fusion_cc_identify.FusionIdentifyAddSus import FusionIdentifyAddSus
from cc.fusion_cc_identify.FusionIdentifyWithoutCovInfo import FusionIdentifyWithoutCovInfo
from cc.fusion_cc_identify.FusionIdentifyWithoutExpertFeature import FusionIdentifyWithoutExpertFeature
from cc.fusion_cc_identify.SupContraCCIdentify import SupContraCCIdentify


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
        "cce_threshold": 1,
        "select_ratio": [i / 100 for i in range(5, 31, 5)],
        "sus_threshold": [i / 100 for i in range(50, 91, 5)]
    }
    name = "Fusion_2025_3-1-cov"
    run(program_list, "Chart", 1, SupContraCCIdentify, name, arg_dict)


if __name__ == "__main__":
    main()
