import numpy as np
from torch.utils import data


class CombinedInfoTestCaseLoader(data.Dataset):
    def __init__(self, passing_df, true_passing_test_expert, failing_df, failing_test_expert):
        self.true_passing_np = np.array(passing_df)
        self.failing_np = np.array(failing_df)
        self.true_passing_expert_np = np.array(true_passing_test_expert)
        self.failing_test_expert_np = np.array(failing_test_expert)

    def __getitem__(self, item):
        random_index = np.random.randint(self.true_passing_np.shape[0])
        random_fail_index = np.random.randint(self.failing_np.shape[0])
        return self.true_passing_np[item], self.true_passing_expert_np[item], \
               self.true_passing_np[random_index], self.true_passing_expert_np[random_index], \
               self.failing_np[random_fail_index], self.failing_test_expert_np[random_fail_index]

    def __len__(self):
        return self.true_passing_np.shape[0]