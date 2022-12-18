from benchopt import BaseSolver, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import numpy as np

    import jax.numpy as jnp
    from ott.geometry import pointcloud
    from ott.solvers.linear import sinkhorn


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class Solver(BaseSolver):

    # Name to select the solver in the CLI and to display the results.
    name = 'OTT'

    install_cmd = 'conda'
    requirements = ['pip:git+https://github.com/ott-jax/ott']

    # List of parameters for the solver. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    parameters = {
        'reg': [1e-4, 1e-1],
    }

    def set_objective(self, x, a, y, b):
        # Define the information received by each solver from the objective.
        # The arguments of this function are the results of the
        # `Objective.get_objective`. This defines the benchmark's API for
        # passing the objective to the solver.
        # It is customizable for each benchmark.
        self.x, self.y, self.a, self.b = map(jnp.array, (x, y, a, b))

        # Call the solver once to make sure to precompile.
        self.run(1)

    def run(self, n_iter):
        # This is the function that is called to evaluate the solver.
        # It runs the algorithm for a given a number of iterations `n_iter`.

        self.out = sinkhorn.sinkhorn(
            pointcloud.PointCloud(self.x, self.y, epsilon=self.reg),
            self.a,
            self.b,
            threshold=0,
            lse_mode=True,
            max_iterations=n_iter + 1,
        )

    def get_result(self):
        # Return the result from one optimization run.
        # The outputs of this function are the arguments of `Objective.compute`
        # This defines the benchmark's API for solvers' results.
        # it is customizable for each benchmark.
        return np.array(self.out.matrix)
