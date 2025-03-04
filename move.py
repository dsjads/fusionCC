import pandas as pd

# 定义Excel文件路径
excel_file = 'comp/context_change_data.xlsx'
# 定义输出的txt文件路径
output_txt_file = 'extracted_data.txt'

# 使用pandas读取Excel文件
df = pd.read_excel(excel_file)

# 提取第一列和第四列，这里假设列标签是从0开始计数，即第0列和第3列
# 如果你的列是命名的（比如'A', 'B', 'C'...），可以使用列名代替，如 `df[['Column1Name', 'Column4Name']]`
extracted_data = df.iloc[:, [0, 3]]

# 将提取的数据保存为txt文件，每一列数据之间用制表符分隔
with open(output_txt_file, 'w') as file:
    for index, row in extracted_data.iterrows():
        # 写入每行数据，列之间用\t分隔，末尾不加换行符，除非你想每条记录独占一行
        file.write('\t'.join(map(str, row)) + '\n')

print(f"数据已成功提取并保存至{output_txt_file}")