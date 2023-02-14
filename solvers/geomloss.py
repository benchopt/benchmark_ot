from benchopt import BaseSolver, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from geomloss.sinkhorn_divergence import log_weights, sinkhorn_loop
    from geomloss.sinkhorn_samples import cost_routines, softmin_tensorized
    import jax.numpy as jnp
    import numpy as np
    from ott.geometry import pointcloud
    from ott.problems.linear import linear_problem
    from ott.solvers.linear.sinkhorn import SinkhornOutput
    import torch


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class Solver(BaseSolver):

    # Name to select the solver in the CLI and to display the results.
    name = 'GeomLoss'

    install_cmd = 'conda'
    requirements = ['torch', 'pykeops', 'pip:geomloss']

    # List of parameters for the solver. The benchmark will consider
    # the cross product for each key in the dictionary.
    parameters = {
        'reg': [1e-2, 1e-1],
    }

    def set_objective(self, x, a, y, b):
        # Convert problem into jax array with int32 for jitted computations.
        x_jax, y_jax, a_jax, b_jax = map(
            lambda x: jnp.array(x), (x, y, a, b)
        )
        self.ot_prob = linear_problem.LinearProblem(
                pointcloud.PointCloud(x_jax, y_jax), a_jax, b_jax
            )
        self.x, self.a, self.y, self.b = [
            torch.from_numpy(t).float()[None] for t in (x, a, y, b)
        ]
        # put all tensors on GPU if available
        if torch.cuda.is_available():
            self.x, self.a, self.y, self.b = [
                t.cuda() for t in (self.x, self.a, self.y, self.b)
            ]

    def run(self, n_iter):
        # content of `sinkhorn_tensorized` from
        # https://github.com/jeanfeydy/geomloss/blob/main/geomloss/sinkhorn_samples.py
        x, y, a, b = self.x, self.y, self.a, self.b
        # Retrieve the batch size B, the numbers of samples N, M
        # and the size of the ambient space D:
        B, N, D = x.shape
        _, M, _ = y.shape
        p = 2
        eps = self.reg
        eps_list = [eps for _ in range(max(10*n_iter, 1))]
        cost = None

        # By default, our cost function :math:`C(x_i,y_j)` is a halved,
        # squared Euclidean distance (p=2) or a simple Euclidean distance (p=1):
        if cost is None:
            cost = cost_routines[p]

        # Compute the relevant cost matrices C(x_i, y_j), C(y_j, x_i), etc.
        # Note that we "detach" the gradients of the "right-hand sides":
        # this is coherent with the way we compute our gradients
        # in the `sinkhorn_loop(...)` routine, in the `sinkhorn_divergence.py` file.
        # Please refer to the comments in this file for more details.
        C_xy = cost(x, y.detach())  # (B,N,M) torch Tensor
        C_yx = cost(y, x.detach())  # (B,M,N) torch Tensor

        # N.B.: The "auto-correlation" matrices C(x_i, x_j) and C(y_i, y_j)
        #       are only used by the "debiased" Sinkhorn algorithm.
        C_xx = None  # (B,N,N) torch Tensor
        C_yy = None  # (B,M,M) torch Tensor

        # Use an optimal transport solver to retrieve the dual potentials:
        f_aa, g_bb, g_ab, f_ba = sinkhorn_loop(
            softmin_tensorized,
            log_weights(a),
            log_weights(b),
            C_xx,
            C_yy,
            C_xy,
            C_yx,
            eps_list,
            None,
            debias=False,
        )
        self.f_ba = f_ba.view_as(a)
        self.g_ab = g_ab.view_as(b)

    def get_result(self):
        # Return the result from one optimization run.
        out = SinkhornOutput(
            f=jnp.array(self.f_ba.detach().cpu().numpy()[0]),
            g=jnp.array(self.g_ab.detach().cpu().numpy()[0]),
            ot_prob=self.ot_prob,
        )
        return np.array(out.matrix)
