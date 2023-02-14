from benchopt import BaseObjective, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import numpy as np
    from sklearn.metrics import pairwise_distances


class Objective(BaseObjective):

    name = "Optimal Transport"

    # List of parameters for the objective. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    parameters = {
    }

    # Minimal version of benchopt required to run this benchmark.
    # Bump it up if the benchmark depends on a new feature of benchopt.
    min_benchopt_version = "1.3"

    def set_data(self, x, a, y, b):
        # The keyword arguments of this function are the keys of the dictionary
        # returned by `Dataset.get_data`. This defines the benchmark's
        # API to pass data.
        self.x, self.a = x, a
        self.y, self.b = y, b
        self.M = pairwise_distances(self.x, self.y)

    def compute(self, P):

        P_a, P_b = P.sum(axis=1), P.sum(axis=0)
        violation = 0.5 * ((P_a - self.a) ** 2).sum()
        violation += 0.5 * ((P_b - self.b) ** 2).sum()

        obj = (P*self.M).sum()
        P_supp = P[P > 0]
        neg_entropy = (P_supp*np.log(P_supp)).sum()

        # benchopt tries to early stop solvers based on value.
        # Set the objective value to be large as long as violation is higher
        # than a threshold.
        obj_violation = (1+violation) * np.diag(self.M).mean()

        # This method can return many metrics in a dictionary. One of these
        # metrics needs to be `value` for convergence detection purposes.
        return dict(
            cost=obj,
            violation=violation,
            neg_entropy=neg_entropy,
            value=obj if violation < 1e-9 else obj_violation,
        )

    def get_one_solution(self):
        # Return one solution. The return value should be an object compatible
        # with `self.compute`. This is mainly for testing purposes.
        return np.eye(self.a.shape[0], self.b.shape[0])

    def get_objective(self):
        # Define the information to pass to each solver to run the benchmark.
        # The output of this function are the keyword arguments
        # for `Solver.set_objective`. This defines the
        # benchmark's API for passing the objective to the solver.
        # It is customizable for each benchmark.
        return dict(
            x=self.x, a=self.a, y=self.y, b=self.b
        )
