from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import SufficientProgressCriterion

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import jax.numpy as jnp
    import numpy as np

    # Import Geomloss which is based on pytorch.
    import torch
    from geomloss.sinkhorn_divergence import log_weights, sinkhorn_loop
    from geomloss.sinkhorn_samples import cost_routines, softmin_tensorized

    # Using OTT to get an output compatible with the objective.
    import ott
    from ott.geometry import pointcloud
    from ott.problems.linear import linear_problem
    from ott.solvers.linear.sinkhorn import SinkhornOutput


class Solver(BaseSolver):

    # Name to select the solver in the CLI and to display the results.
    name = 'GeomLoss'

    install_cmd = 'conda'
    requirements = [
        'pytorch:pytorch', 'pip:pykeops', 'pip:geomloss', 'ott-jax'
    ]

    # List of parameters for the solver. The benchmark will consider
    # the cross product for each key in the dictionary.
    parameters = {
        'reg': [1e-2, 1e-1],
        'use_gpu': [False, True],
    }

    stopping_criterion = SufficientProgressCriterion(patience=50)

    def skip(self, **kwargs):
        # we skip the solver if use_gpu is True and no GPU is available
        if self.use_gpu and not torch.cuda.is_available():
            return True, "No GPU available"
        return False, None

    def set_objective(self, x, a, y, b):
        # Create a ott problem based on jax to compute the output \
        # of the solver.
        x_jax, y_jax, a_jax, b_jax = map(
            lambda x: jnp.array(x), (x, y, a, b)
        )
        self.ot_prob = linear_problem.LinearProblem(
            pointcloud.PointCloud(
                x_jax, y_jax, epsilon=self.reg,
                cost_fn=ott.geometry.costs.SqPNorm(p=2)
            ), a_jax, b_jax,
        )

        # Store the problem in torch to use GeomLoss.
        # Use the GPU when it is available.
        device = 'cuda' if self.use_gpu else 'cpu'

        self.x, self.a, self.y, self.b = [
            torch.from_numpy(t).float().to(device=device)[None]
            for t in (x, a, y, b)
        ]

    def run(self, n_iter):
        # content of `sinkhorn_tensorized` from
        # https://github.com/jeanfeydy/geomloss/blob/main/geomloss/sinkhorn_samples.py
        x, y, a, b = self.x, self.y, self.a, self.b
        # Retrieve the batch size B, the numbers of samples N, M
        # and the size of the ambient space D:
        B, N, D = x.shape
        _, M, _ = y.shape

        # Geomloss use the eps schedule to specify the number of iterations.
        # We use a fix schedule here.
        eps = self.reg
        eps_list = [eps for _ in range(10*n_iter + 1)]
        # Select the squared euclidean cost
        cost = cost_routines[2]
        # Compute the relevant cost matrices C(x_i, y_j), C(y_j, x_i), etc.
        C_xy = cost(x, y)  # (B,N,M) torch Tensor
        C_yx = C_xy.transpose(1, 2)  # (B,M,N) torch Tensor
        self.C = C_xy

        # Use an optimal transport solver to retrieve the dual potentials:
        f_aa, g_bb, g_ab, f_ba = sinkhorn_loop(
            softmin_tensorized,
            log_weights(a),
            log_weights(b),
            None,
            None,
            C_xy,
            C_yx,
            eps_list,
            rho=None,
            debias=False,
        )
        self.f_ba = f_ba.view_as(a)
        self.g_ab = g_ab.view_as(b)

    def get_result(self):
        # Return the result from one optimization run.
        f = self.f_ba + self.reg*log_weights(self.a)
        g = self.g_ab + self.reg*log_weights(self.b)
        out = SinkhornOutput(
            f=jnp.array(f.detach().cpu().numpy()[0]),
            g=jnp.array(g.detach().cpu().numpy()[0]),
            ot_prob=self.ot_prob,
        )
        return dict(P=np.array(out.matrix))
