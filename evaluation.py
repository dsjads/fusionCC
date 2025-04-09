# 定义函数计算指标
def calculate_metrics(true_positive, detected, ground_truth):
    recall = true_positive / ground_truth
    precision = true_positive / detected
    f1_score = (2 * recall * precision) / (recall + precision)
    return recall, precision, f1_score


# 文件路径
file_path = './new_results/MLCCI/origin_record.txt'

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

# 输出每个组的指标结果
for group, metric in metrics.items():
    true_positive = metric['true_positive']
    detected = metric['detected']
    ground_truth = metric['ground_truth']

    recall, precision, f1_score = calculate_metrics(true_positive, detected, ground_truth)

    print(f"Group: {group}")
    print(f"Recall: {recall:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"F1 Score: {f1_score:.2f}")
    print()

# 计算并输出 total 版本的指标结果
total_true_positive = total_metrics['true_positive']
total_detected = total_metrics['detected']
total_ground_truth = total_metrics['ground_truth']

total_recall, total_precision, total_f1_score = calculate_metrics(
    total_true_positive, total_detected, total_ground_truth)

print("Group: Total")
print(f"Recall: {total_recall:.2f}")
print(f"Precision: {total_precision:.2f}")
print(f"F1 Score: {total_f1_score:.2f}")
