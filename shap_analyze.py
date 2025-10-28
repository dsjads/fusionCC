import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sympy.printing.pretty.pretty_symbology import line_width

# 读取数据
df = pd.read_csv('shap_results.csv', header=None, names=['program_version', 'coverage_shap', 'handcrafted_shap'])

# 过滤掉存在NaN的行
df = df.dropna()

df = df[(df['coverage_shap'] != 0) | (df['handcrafted_shap'] != 0)]

# 从程序版本中提取程序名称
df['program'] = df['program_version'].str.split('-').str[0]

# 将gzip、libtiff和space合并为C
df['program'] = df['program'].replace({'gzip': 'gzip\nlibtiff\nspace', 'libtiff': 'gzip\nlibtiff\nspace', 'space': 'gzip\nlibtiff\nspace'})

# 将数据从宽格式转换为长格式
df_long = pd.melt(df, id_vars=['program', 'program_version'],
                  value_vars=['coverage_shap', 'handcrafted_shap'],
                  var_name='feature_type', value_name='shap_value')

# 清理特征类型名称
df_long['feature_type'] = df_long['feature_type'].str.replace('_shap', '')

# 创建图形
plt.figure(figsize=(14, 8))

# 使用seaborn绘制箱线图，不显示异常值
sns.boxplot(data=df_long, x='program', y='shap_value', hue='feature_type',
            palette={'coverage': 'lightblue', 'handcrafted': 'lightcoral'},
            showfliers=False,
            linewidth = 2.5)  # 不显示异常值

# 设置标签和标题
plt.xlabel('Programs', fontsize=24)
plt.ylabel('SHAP Values', fontsize=24)
# plt.title('SHAP Values Comparison: Coverage Features vs Handcrafted Features')

# 调整图例
plt.legend(title='Feature Type', title_fontsize=20, fontsize=20)

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

# 调整布局
plt.tight_layout()

plt.savefig('shap_results.pdf', bbox_inches='tight')  #
plt.savefig('shap_results.png', bbox_inches='tight')  #
# 显示图形
plt.show()
