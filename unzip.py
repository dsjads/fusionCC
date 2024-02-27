import os
import tarfile

base_dir = r'C:\Users\zhangwentao\Desktop\fault-localization.cs.washington.edu\data\Chart'  # Chart目录路径

for subdir in sorted(os.listdir(base_dir)):
    sub_dir_path = os.path.join(base_dir, subdir)
    if os.path.isdir(sub_dir_path):
        dict_name = subdir  # 将子目录名赋值给dict_name
        tar_file_path = os.path.join(sub_dir_path, 'gzoltar-files.tar.gz')

        # 解压gzoltar-files.tar.gz文件
        with tarfile.open(tar_file_path, 'r:gz') as tar:
            tar.extractall(path=os.path.join(base_dir, dict_name))

        print(f"解压完毕: {tar_file_path}")