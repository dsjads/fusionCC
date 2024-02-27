from cc.cc_baselines.tech_1_2010.Tech1_2010_Identification import Tech12010Identification
from cc.core import run, multi_process_run


def main():
    program_list = [
        "Chart",
        "Closure-2023-12-6-1",
        "Lang",
        "Math",
        "Mockito",
        "Time"
    ]
    name = "2022-11-4-Tech-1-2010-80-50"
    # run(program_list, "Chart", 0, Tech12010Identification, name, 0.6)
    multi_process_run(program_list, Tech12010Identification, name, 0.8)



if __name__ == "__main__":
    main()
    # task_complete("Triplet CC end")