# 定义排序函数
def chart_sort_key(chart):
    chart_number = int(chart.split('-')[1].split('\t')[0])
    return chart_number

def custom_sort_key(string):
    prefix, rest = string.split('-', 1)
    number = rest.split('\t')[0]
    return prefix, int(number)


# 文件路径
file_path = 'new_results/2024-1-5-CBNN-k10-30-2/origin_record.txt'

output_file_path = 'new_results/2024-1-5-CBNN-k10-30-2/sorted_record.txt'

# 读取文件内容
with open(file_path, 'r') as file:
    data = file.readlines()

# 按照排序函数进行排序
sorted_data = sorted(data, key=custom_sort_key)

# for item in sorted_data:
#     print(item.strip())  # 使用strip()函数去除末尾的换行符
# 将排序后的结果写入新文件
with open(output_file_path, 'w') as output_file:
    for item in sorted_data:
        output_file.write(item)
