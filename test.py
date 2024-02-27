import pandas as pd
import os

base_dir = "feature/Triplet/Chart-2023-12-6-1/"
for root, dirs, files in os.walk(base_dir):
    for file in files:
        file_path = os.path.join(root, file)
        feature_matrix = pd.read_csv(file_path, index_col=0)
        print(feature_matrix.shape)
