from contextlib import nullcontext
from multiprocessing import Pool

import numpy as np
import pandas as pd


from CONFIG import *
from Features import Features
from read_data.Defects4JDataLoader import Defects4JDataLoader


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
    pool = Pool(processes=8)
    for program in program_list:
        for i in cc_info[program]:
            if program == start_program and i == start_program_id:
                flag = True
            if flag:
                pool.apply_async(featureExtract, (save_path, program, i), error_callback=error_callback)
                # featureExtract(save_path, program, i)
    pool.close()
    pool.join()
    print("Finished")


def error_callback(error):
    print(f"Error info: {error}")

def featureExtract(save_path, program, index):
    print(f"program {program}-{index} is processing")
    data = Defects4JDataLoader(os.path.join(project_dir, '..', 'data'), program, index)
    data.load()
    # CCE = find_CCE(data.data_df, 1)
    # CCE.append("error")
    # new_data_df = data.data_df[CCE]
    new_data_df = data.data_df
    features = Features(new_data_df)
    ssp = features.suspScore()
    cr = features.covRatio()
    sf = features.similarityFactor()
    merged_matrix = np.concatenate((ssp, cr, sf), axis=1)
    passing_data_df = new_data_df[new_data_df["error"] == 0]
    df = pd.DataFrame(merged_matrix, index=passing_data_df.index)
    # npy_file_path = f"{save_path}/MLCCI/{program}-{cita}/features-{program}-{i}.npy"
    df_file_path = f"{save_path}/MLCCI2/{program}-csv/features-{program}-{index}.csv"
    # np.save(npy_file_path, merged_matrix)
    df.to_csv(df_file_path, index=True)
    print("successfully save:", df_file_path)


def run(program_list, start_program, start_program_id, cita):
    save_path = os.path.join(project_dir, "feature")
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    flag = False
    for program in program_list:
        for i in cc_info[program]:
            print(program, i)
            if program == start_program and i == start_program_id:
                flag = True
            if flag:
                data = Defects4JDataLoader(os.path.join(project_dir, '..', 'data'), program, i)
                data.load()
                CCE = find_CCE(data.data_df, cita)
                CCE.append("error")
                new_data_df = data.data_df[CCE]
                # new_data_df = data.data_df
                # features = Features(data.data_df)

                features = Features(new_data_df)
                ssp = features.suspScore()
                cr = features.covRatio()
                sf = features.similarityFactor()
                # load_path = f"{save_path}/Expert/{program}-{cita}/features-{program}-{i}.npy"

                # merged_matrix = np.load(load_path)
                merged_matrix = np.concatenate((ssp, cr, sf), axis=1)

                passing_data_df = new_data_df[new_data_df["error"] == 0]
                df = pd.DataFrame(merged_matrix, index=passing_data_df.index)

                # npy_file_path = f"{save_path}/MLCCI/{program}-{cita}/features-{program}-{i}.npy"
                df_file_path = f"{save_path}/Expert/{program}-csv/features-{program}-{i}.csv"

                # np.save(npy_file_path, merged_matrix)
                df.to_csv(df_file_path, index=True)
                print("successfully save:", df_file_path)


def main():
    program_list = [
        "Chart",
        "Lang",
        "Math",
        "Mockito",
        "Time"
        # "Closure"
    ]
    run2(program_list, "Chart", 1)


if __name__ == "__main__":
    main()
