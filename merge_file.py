def merge_matching_rows(file1_path, file2_path, output_file):
    # 使用字典存储每份文件中各Chart的第7列数据
    data_dict = {}

    # 读取第一个文件
    with open(file1_path, 'r') as file1:
        for line in file1:
            parts = line.strip().split('\t')  # 分割行数据
            name = parts[0]  # 名称
            value = parts[6]  # 第7列的值
            if name not in data_dict:
                data_dict[name] = [value]
            else:
                data_dict[name].append(value)

    # 读取第二个文件，并合并相同名称的第7列数据
    with open(file2_path, 'r') as file2:
        for line in file2:
            parts = line.strip().split('\t')
            name = parts[0]
            print(name)
            value = parts[1]
            if name in data_dict:
                data_dict[name].append(value)
            else:
                data_dict[name] = [value]

    # 写入合并后的数据到新文件
    with open(output_file, 'w') as output:
        for name, values in data_dict.items():
            # 确保每组数据至少有两个值（来自两个文件），然后写入
            if len(values) >= 2:
                output.write(f"{name}\t{values[0]}\t{values[1]}\n")


def find_common_elements(file1_path, file2_path, output_file_path):
    # 读取第一个文件的第一列构建集合
    with open(file1_path, 'r') as file1:
        set_file1 = {line.split()[0] for line in file1}

    # 读取第二个文件，检查并保存交集元素所在的行到新文件
    with open(file2_path, 'r') as file2, open(output_file_path, 'w') as output_file:
        for line in file2:
            element = line.split()[0]
            if element in set_file1:
                output_file.write(line)


# 使用函数，替换下面的路径为您的实际文件路径
file2_path = 'comp/refine_data.txt'
file1_path = 'new_results/Fusion_2024-10-13_pointnet_v2-relabel/Fusion_2024-10-13_pointnet_v2-relabel_MFR.txt'
output_file_path = 'comp/Fusion vs ContraCC.txt'

merge_matching_rows(file1_path, file2_path, output_file_path)
print("Intersection processed and saved to", output_file_path)