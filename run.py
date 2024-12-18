import os
import sys

import torch

from utils.args_util import parse_args
from augmentation.Pipeline import Pipeline


def main():
    project_dir = os.path.dirname(__file__)
    configs = parse_args(sys.argv)
    sys.argv = ["run.py"]
    pl = Pipeline(project_dir, configs)
    pl.run()


if __name__ == "__main__":
    main()
