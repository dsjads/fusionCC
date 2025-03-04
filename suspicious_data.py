import os

from CONFIG import project_dir, cc_info
from augmente_data import Features
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
                features = Features(new_data_df)
                df = features.get_suspicious_df()
                features.get_augmentation_data()

                df_file_path = f"{save_path}/Sus/{program}/sus-{program}-{i}.csv"
                df.to_csv(df_file_path, index=True)
                print("successfully save:", df_file_path)


def main():
    program_list = [
        "Chart",
        "Lang",
        "Math",
        "Time",
        "Mockito"
    ]
    run(program_list, "Chart", 1, 1)


if __name__ == "__main__":
    main()
