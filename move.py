import os
import shutil

base_dir = r'C:\Users\zhangwentao\PycharmProjects\merit\data\d4j\data\Chart'  # Closure目录路径

for subdir in sorted(os.listdir(base_dir)):
    sub_dir_path = os.path.join(base_dir, subdir)
    if os.path.isdir(sub_dir_path):
        dict_name = subdir  # 将子目录名赋值给dict_name

        gzoltars_path = os.path.join(sub_dir_path, 'gzoltars')
        source_path = os.path.join(gzoltars_path, 'Chart', dict_name, 'matrix')
        target_path = os.path.join(base_dir, dict_name, 'matrix')

        # 移动文件
        shutil.move(source_path, target_path)

        print(f"移动完毕: {source_path} -> {target_path}")