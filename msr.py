import pandas as pd
from scipy.stats import wilcoxon


# 定义一个函数来读取txt文件并转换为DataFrame
def load_data_from_txt(file_path):
    return pd.read_csv(file_path, sep='\t', names=['Category', 'Value1', 'Value2'])


# 加载数据
file_path = 'comp/Fusion vs ContraCC.txt'  # 替换为你的txt文件路径
df = load_data_from_txt(file_path)


# 定义一个函数来执行威尔科克森符号秩检验并打印结果
def perform_wilcoxon_test(group_name, values1, values2):
    statistic_2_tailed, pvalue_2_tailed = wilcoxon(values1, values2)
    print(f"{group_name} - 2-tailed:")
    print(f"Statistic: {statistic_2_tailed}, p-value: {pvalue_2_tailed}")

    statistic_1_tailed_left, pvalue_1_tailed_left = wilcoxon(values1, values2, alternative='less')
    print(f"{group_name} - 1-tailed (left):")
    print(f"Statistic: {statistic_1_tailed_left}, p-value: {pvalue_1_tailed_left}")

    statistic_1_tailed_right, pvalue_1_tailed_right = wilcoxon(values1, values2, alternative='greater')
    print(f"{group_name} - 1-tailed (right):")
    print(f"Statistic: {statistic_1_tailed_right}, p-value: {pvalue_1_tailed_right}")


# 对每个类别执行威尔科克森符号秩检验
categories = ['Chart', 'Lang', 'Math','Mockito','Time']
for category in categories:
    group_data = df[df['Category'].str.startswith(category)]
    values1 = group_data['Value1'].astype(float)
    values2 = group_data['Value2'].astype(float)
    perform_wilcoxon_test(category, values1, values2)

# 执行所有类别的威尔科克森符号秩检验
all_values1 = df['Value1'].astype(float)
all_values2 = df['Value2'].astype(float)
perform_wilcoxon_test('All Categories', all_values1, all_values2)