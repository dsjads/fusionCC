import time
from contextlib import nullcontext
from multiprocessing import Pool

import numpy as np
import pandas as pd


from CONFIG import *
from Features import Features
from read_data.Defects4JDataLoader import Defects4JDataLoader
from read_data.ManyBugsDataLoader import ManyBugsDataLoader


def find_CCE(data_df, cita):
    failing_df = data_df[data_df["error"] == 1]
    passing_df = data_df[data_df["error"] == 0]
    CCE = []
    for i in failing_df.columns:
        if i != "error":
            if _is_CCE(failing_df[i], passing_df[i], cita):
                CCE.append(i)
    return CCE


def _is_CCE(fail_data, pass_data, cita):
    fT = _getfT(fail_data)
    pT = _getpT(pass_data)
    if fT == 1.0 and pT < cita:
        return True
    else:
        return False


def _getfT(data):
    uncover = sum(data == 0)
    cover = sum(data == 1)
    fT = cover / (uncover + cover)
    return fT


def _getpT(data):
    uncover = sum(data == 0)
    cover = sum(data == 1)
    pT = cover / (uncover + cover)
    return pT

def run2(program_list, start_program, start_program_id):
    save_path = os.path.join(project_dir, "feature")
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    flag = False
    # pool = Pool(processes=32)
    for program in program_list:
        for i in cc_info[program]:
            if program == start_program and i == start_program_id:
                flag = True
            if flag:
                # pool.apply_async(featureExtract, (save_path, program, i), error_callback=error_callback)
                featureExtract(save_path, program, i)
    # pool.close()
    # pool.join()
    # print("Finished")


def error_callback(error):
    print(f"Error info: {error}")

def featureExtract(save_path, program, index):
    print(f"program {program}-{index} is processing")
    # data = Defects4JDataLoader(os.path.join(project_dir, '..', 'data'), program, index)
    data = ManyBugsDataLoader(os.path.join(project_dir, '..','MANYBUGS_DATA'), program, index)
    data.load()
    CCE = find_CCE(data.data_df, 1)
    if len(CCE) == 0:
        print("No CCE")
        return
    CCE.append("error")
    new_data_df = data.data_df[CCE]
    data.data_df = None
    data.feature_df = None
    # new_data_df = data.data_df
    features = Features(new_data_df)
    ssp = features.suspScore()
    cr = features.covRatio()
    sf = features.similarityFactor()
    # cr = features.covRatio()
    merged_matrix = np.concatenate((ssp,cr,sf), axis=1)
    passing_data_df = new_data_df[new_data_df["error"] == 0]
    df = pd.DataFrame(merged_matrix, index=passing_data_df.index)
    df_file_path = f"{save_path}/Ours/{program}-csv/features-{program}-{index}.csv"
    df.to_csv(df_file_path, index=True)
    print("successfully save:")


def main():
    # program_list = [
    #     "Chart",
    #     "Lang",
    #     "Math",
    #     "Mockito",
    #     "Time"
    #     # "Closure"
    # ]
    program_list = ["space"]
    run2(program_list, "space", 33)


if __name__ == "__main__":
    main()
