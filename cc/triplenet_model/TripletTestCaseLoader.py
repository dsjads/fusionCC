import numpy as np
import torch.utils.data as data

class TripletTestCaseLoader(data.Dataset):

    def __init__(self, passing_df, failing_df):
        # problem 1: how to determine the
        self.true_passing_np = np.array(passing_df)
        self.failing_np = np.array(failing_df)

    def __getitem__(self, item):
        random_index = np.random.randint(self.true_passing_np.shape[0])
        random_fail_index = np.random.randint(self.failing_np.shape[0])
        return self.true_passing_np[item], self.true_passing_np[random_index], self.failing_np[random_fail_index]

    def __len__(self):
        return self.true_passing_np.shape[0]
