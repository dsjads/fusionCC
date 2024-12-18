import sys
import time

from CONFIG import *
from utils.postprocess import parse
from utils.write_util import write_rank_to_txt


def run(program_list, start_program, start_program_id, identifyMethod, way, n):
    flag = False
    for program in program_list:
        for i in cc_info[program]:
            print(program, i)
            if program == start_program and i == start_program_id:
                flag = True
            if flag:
                configs = {'-d': 'd4j', '-p': program, '-i': i, '-m': method_para, '-e': 'origin'}
                # configs = {'-d': 'd4j', '-p': "Closure-2023-12-6-1", '-i': 36, '-m': method_para, '-e': 'origin'}
                sys.argv = os.path.basename(__file__)
                pl = identifyMethod(project_dir, configs, n, way)
                start = time.time()
                pl.find_cc_index()
                end = time.time()
                time_ = dict()
                time_["time"] = end - start
                save_path = os.path.join(project_dir, "results", way, "time.txt")
                write_rank_to_txt(time_, save_path, program, i)
                pl.evaluation()
                pl.calRes("trim")
                pl.calRes("relabel")

    parse(os.path.join(project_dir, "results", way), "origin_record.txt", "precision_recall.xlsx")
    for operation in ["trim", "relabel"]:
        op_way = way+"-"+operation
        parse(os.path.join(project_dir, "results", op_way), op_way+"_MFR.txt", "FL-1.xlsx")
