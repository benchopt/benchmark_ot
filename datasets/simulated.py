from benchopt import BaseDataset, safe_import_context


# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import numpy as np


# All datasets must be named `Dataset` and inherit from `BaseDataset`
class Dataset(BaseDataset):

    # Name to select the dataset in the CLI and to display the results.
    name = "Simulated"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    # Any parameters 'param' defined here is available as `self.param`.
    parameters = {
        'n_samples': [1000, 10000],
        'random_state': [27],
    }

    def get_data(self):
        # The return arguments of this function are passed as keyword arguments
        # to `Objective.set_data`. This defines the benchmark's
        # API to pass data. It is customizable for each benchmark.

        rng = np.random.RandomState(self.random_state)
        n = self.n_samples

        mu_x = np.array([0, 0])
        cov_x = np.array([[1, 0], [0, 1]])

        mu_y = np.array([10, 10])
        cov_y = np.array([[1, .8], [.8, 1]])

        x = rng.randn(n, 2) @ cov_x + mu_x
        y = rng.randn(n + 1, 2) @ cov_y + mu_y

        # uniform distribution on samples
        a, b = np.ones(n) / n, np.ones(n + 1) / (n + 1)

        # The dictionary defines the keyword arguments for `Objective.set_data`
        return dict(x=x, a=a, y=y, b=b)
