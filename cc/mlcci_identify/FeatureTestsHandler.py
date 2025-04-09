import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class FeatureTestsHandler:
    def __init__(self):
        pass

    @staticmethod
    def get_feature_from_file(project_dir, program, bug_id):
        save_path = os.path.join(project_dir, "feature", "MLCCI", f"{program}-csv")
        # save_path = os.path.join(project_dir, "feature", "Expert", f"{program}-passing-csv-1")
        file_path = f"{save_path}/features-{program}-{bug_id}.csv"
        feature_matrix = pd.read_csv(file_path, index_col=0)
        ssp = feature_matrix.iloc[:, 0:20]
        cr = feature_matrix.iloc[:, 20:40]
        sf = feature_matrix.iloc[:, 40:60]
        ssp_array = ssp.to_numpy()
        cr_array = cr.to_numpy()
        sf_array = sf.to_numpy()

        standardScaler1 = StandardScaler()
        standardScaler2 = StandardScaler()
        standardScaler3 = StandardScaler()
        standardScaler1.fit(ssp_array)
        standardScaler2.fit(cr_array)
        standardScaler3.fit(sf_array)
        ssp_standard = standardScaler1.transform(ssp_array)
        cr_standard = standardScaler2.transform(cr_array)
        sf_standard = standardScaler3.transform(sf_array)

        return np.hstack((ssp_standard, cr_standard, sf_standard))

    @staticmethod
    def standard(features):
        standardScaler = StandardScaler()
        standardScaler.fit(features)
        features_standard = standardScaler.transform(features)
        return features_standard

    @staticmethod
    def get_sus_data_from_file(project_dir,program,bug_id):
        load_path = os.path.join(project_dir, "feature", "Sus", f"{program}")
        file_path = f"{load_path}/sus-{program}-{bug_id}.csv"

        feature_matrix = pd.read_csv(file_path, index_col=0)

        return feature_matrix