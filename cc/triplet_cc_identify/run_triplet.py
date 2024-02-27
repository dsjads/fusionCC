from cc.triplet_cc_identify.TripletCCIdentify import TripletCCIdentify
from cc.core import *


def main():
    program_list = [
        "Chart",
        # "Closure-2023-12-6-1",
        # "Lang",
        # "Math",
        # "Mockito",
        # "Time"
    ]
    name = "triplet"
    # thresholds=
    run(program_list, "Chart", 1, TripletCCIdentify, name, [[0.2, 0.2, 0.2], 0.56])


if __name__ == "__main__":
    main()
