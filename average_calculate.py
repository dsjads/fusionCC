def merge_averages(old_num, old_values, new_num, new_values):
    """
    合并新旧数据的平均值

    参数:
        old_num: 旧数据的总样本数
        old_values: 旧数据的n个平均值列表
        new_num: 新数据的样本数
        new_values: 新数据的n个平均值列表

    返回:
        list: 合并后的n个平均值列表

    异常:
        ValueError: 当输入参数不符合要求时抛出
    """
    # 验证输入参数
    if not isinstance(old_num, int) or old_num < 0:
        raise ValueError("old_num必须是非负整数")
    if not isinstance(new_num, int) or new_num < 0:
        raise ValueError("new_num必须是非负整数")
    if not isinstance(old_values, list) or not isinstance(new_values, list):
        raise ValueError("old_values和new_values必须是列表")
    if len(old_values) != len(new_values):
        raise ValueError("old_values和new_values的长度必须相同")

    total_num = old_num + new_num
    if total_num == 0:
        raise ValueError("总样本数不能为0")

    # 计算合并后的平均值
    merged_values = []
    for old_avg, new_avg in zip(old_values, new_values):
        # 计算总和：旧数据总和 + 新数据总和
        total_sum = old_avg * old_num + new_avg * new_num
        # 计算新的平均值
        merged_avg = total_sum / total_num
        merged_values.append(merged_avg)

    return merged_values


# 示例用法
if __name__ == "__main__":
    # 示例数据
    old_num = 256  # 旧数据总样本数
    old_values = [299.59, 975.66, 332.12, 255.85,243.25, 225.60, 793.58, 1429.26, 702.86, 698.30, 706.94, 680.23]  # 旧数据的3个平均值

    new_num = 30  # 新数据样本数
    new_values = [158.03, 209.63, 181.6, 83.97, 85.5, 131.5, 412.09, 475.99, 457.65, 368.74, 343.43, 333.55]  # 新数据的3个平均值




    try:
        # 计算合并后的平均值
        result = merge_averages(old_num, old_values, new_num, new_values)

        # 输出结果
        print(f"旧数据样本数: {old_num}")
        print(f"旧数据平均值: {old_values}")
        print(f"新数据样本数: {new_num}")
        print(f"新数据平均值: {new_values}")
        print(f"合并后总样本数: {old_num + new_num}")
        print(f"合并后平均值: {[round(v, 2) for v in result]}")  # 保留两位小数显示
    except ValueError as e:
        print(f"错误: {e}")
