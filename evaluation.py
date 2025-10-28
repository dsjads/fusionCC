# 定义函数计算指标
def calculate_metrics(true_positive, detected, ground_truth):
    # 避免除以零错误（新增保护逻辑）
    recall = true_positive / ground_truth if ground_truth != 0 else 0.0
    precision = true_positive / detected if detected != 0 else 0.0

    # 处理F1分数计算时的边界情况
    if recall + precision == 0:
        f1_score = 0.0
    else:
        f1_score = (2 * recall * precision) / (recall + precision)

    return recall, precision, f1_score


def calculate_f1(recall, precision):
    # 检查输入是否为有效数值
    if not isinstance(precision, (int, float)) or not isinstance(recall, (int, float)):
        raise TypeError("精确率和召回率必须是数字类型")

    # 检查数值范围是否合理（0到1之间）
    if not (0 <= precision <= 1) or not (0 <= recall <= 1):
        raise ValueError("精确率和召回率必须在0到1之间")

    # 处理分母为0的情况（避免除零错误）
    if precision + recall == 0:
        return 0.0

    # 计算F1分数
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

f1 = calculate_f1(0.9383,0.9275)
print("f1:", f1)

# 文件路径
file_path = './results/2025-1-6-ms/origin_record.txt'
# file_path = './results/kernel_size/2025-9-25-Fusion-C-cov/origin_record.txt'
# file_path = './results/kernel_size/2025-10-2-mlp-c/origin_record.txt'
# file_path = './results/2025-1-6-ms/origin_record.txt'

# 读取文件内容
with open(file_path, 'r') as file:
    data = file.readlines()

# 创建字典，用于保存每个组的指标结果
metrics = {}
# 用于保存所有组的汇总数据
total_metrics = {
    'true_positive': 0,
    'detected': 0,
    'ground_truth': 0
}

# 处理数据并计算指标
for line in data:
    line = line.strip().split('\t')
    group = line[0].split('-')[0]
    ground_truth = int(line[1])
    detected = int(line[2])
    true_positive = int(line[3])

    if group not in metrics:
        metrics[group] = {
            'true_positive': 0,
            'detected': 0,
            'ground_truth': 0
        }

    metrics[group]['true_positive'] += true_positive
    metrics[group]['detected'] += detected
    metrics[group]['ground_truth'] += ground_truth

    # 汇总所有组的数据
    total_metrics['true_positive'] += true_positive
    total_metrics['detected'] += detected
    total_metrics['ground_truth'] += ground_truth

# 输出每个组的指标结果（保留6位小数）
for group, metric in metrics.items():
    true_positive = metric['true_positive']
    detected = metric['detected']
    ground_truth = metric['ground_truth']

    recall, precision, f1_score = calculate_metrics(true_positive, detected, ground_truth)

    print(f"Group: {group}")
    print(f"Recall: {recall:.6f}")  # 6位小数
    print(f"Precision: {precision:.6f}")  # 6位小数
    print(f"F1 Score: {f1_score:.6f}")  # 6位小数
    print(f"True Positive: {true_positive}")  # 新增原始计数
    print(f"Detected: {detected}")
    print(f"Ground Truth: {ground_truth}")
    print()

# 计算并输出 total 版本的指标结果
total_true_positive = total_metrics['true_positive']
total_detected = total_metrics['detected']
total_ground_truth = total_metrics['ground_truth']

total_recall, total_precision, total_f1_score = calculate_metrics(
    total_true_positive, total_detected, total_ground_truth)

print("Group: Total")
print(f"Recall: {total_recall:.6f}")
print(f"Precision: {total_precision:.6f}")
print(f"F1 Score: {total_f1_score:.6f}")
print(f"Total True Positive: {total_true_positive}")
print(f"Total Detected: {total_detected}")
print(f"Total Ground Truth: {total_ground_truth}")
