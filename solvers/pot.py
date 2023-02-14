from benchopt import BaseSolver, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import ot


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class Solver(BaseSolver):

    # Name to select the solver in the CLI and to display the results.
    name = 'POT'

    install_cmd = 'conda'
    requirements = ['pot']

    # List of parameters for the solver. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    parameters = {
        'reg': [0, 1e-4, 1e-1],
    }

    def set_objective(self, x, a, y, b):
        # Define the information received by each solver from the objective.
        # The arguments of this function are the results of the
        # `Objective.get_objective`. This defines the benchmark's API for
        # passing the objective to the solver.
        # It is customizable for each benchmark.
        self.x, self.y = x, y
        self.a, self.b = a, b

    def run(self, n_iter):
        # This is the function that is called to evaluate the solver.
        # It runs the algorithm for a given a number of iterations `n_iter`.

        M = ot.dist(self.x, self.y)

        if self.reg == 0:
            self.P = ot.emd(self.a, self.b, M, numItermax=n_iter * 10)
        else:
            self.P = ot.sinkhorn(
                self.a, self.b, M, self.reg, numItermax=n_iter * 10 + 1,
                stopThr=0, method="sinkhorn_log",
            )

    def get_result(self):
        # Return the result from one optimization run.
        # The outputs of this function are the arguments of `Objective.compute`
        # This defines the benchmark's API for solvers' results.
        # it is customizable for each benchmark.
        return self.P
