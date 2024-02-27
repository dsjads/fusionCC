import numpy as np
import pandas as pd
from augmentation.ProcessedData import ProcessedData

class PearsonData(ProcessedData):

    def __init__(self, raw_data):
        super().__init__(raw_data)
        self.rest_columns = None

    def process(self, components_percent=0.7, place_holder=0.7):
        if len(self.label_df) > 1:
            features_list = list(self.data_df.columns)[:-1]
            label_name = list(self.data_df.columns)[-1]
            corr_dict = {}
            for feature in features_list:
                corr_dict[feature] = self.pearson(self.data_df[feature], self.data_df[label_name])
            dict = sorted(corr_dict.items(), key=lambda d: d[1], reverse=True)
            num_of_features = int(len(features_list)*components_percent)
            selected_features = dict[:num_of_features]
            rest_columns = dict[num_of_features:]
            selected_features = [tup[0] for tup in selected_features]
            self.rest_columns = [tup[0] for tup in rest_columns]
            low_features = self.feature_df[selected_features]
            low_data = pd.concat([low_features, self.label_df], axis=1)

            self.feature_df = low_features
            self.label_df = self.label_df
            self.data_df = low_data

    def pearson(self, feature, label):
        cov = feature.cov(label)
        std_x = feature.std()
        std_y = label.std()
        if abs(std_x * std_y) < 1e-5:
            return np.nan
        else:
            pearson_corr = cov / (std_x * std_y)
            return pearson_corr