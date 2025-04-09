import os
import torch
import torch.nn as nn

import numpy as np
from sklearn.preprocessing import StandardScaler

import numpy as np
from sklearn.preprocessing import StandardScaler

# 示例数据
ssp_array = np.array([[1, 2], [3, 4], [5, 6]])
cr_array = np.array([[7, 8], [9, 10], [11, 12]])
sf_array = np.array([[13, 14], [15, 16], [17, 18]])

# 方式一：分别标准化再拼接
standardScaler1 = StandardScaler()
standardScaler2 = StandardScaler()
standardScaler3 = StandardScaler()
standardScaler1.fit(ssp_array)
standardScaler2.fit(cr_array)
standardScaler3.fit(sf_array)
ssp_standard = standardScaler1.transform(ssp_array)
cr_standard = standardScaler2.transform(cr_array)
sf_standard = standardScaler3.transform(sf_array)
result1 = np.hstack((ssp_standard, cr_standard, sf_standard))

# 方式二：先拼接再标准化
combined_array = np.hstack((ssp_array, cr_array, sf_array))
standardScaler = StandardScaler()
standardScaler.fit(combined_array)
result2 = standardScaler.transform(combined_array)

print("分别标准化再拼接的结果：")
print(result1)
print("先拼接再标准化的结果：")
print(result2)
