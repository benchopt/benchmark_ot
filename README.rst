
Benchopt benchmark for exact Optimal Transport
==============================================
|Build Status| |Python 3.6+|

Benchopt is a package to simplify and make more transparent and
reproducible the comparisons of optimization algorithms.
This benchmark is dedicated to solvers of discrete exact optimal transport:


$$\\min_{x \\in \\mathbb{R}^{n_1 \\times n_2}} \\langle C, x \\rangle \\quad \\mathrm{subject \, to} \\quad a^\\top x = 1_{n_2}, \\, x b = 1_{n_1} $$


where
$$C \\in \\mathbb{R}^{n_1 \\times n_2}, a \\in \\mathbb{R}_+^{n_1} \\ , b \\in \\mathbb{R}_+^{n_2}$$


Install
-------

This benchmark can be run using the following commands:

.. code-block::

   $ pip install -U benchopt
   $ git clone https://github.com/tomMoral/benchmark_ot
   $ benchopt run benchmark_ot

Apart from the problem, options can be passed to ``benchopt run``, to restrict the benchmarks to some solvers or datasets, e.g.:

.. code-block::

	$ benchopt run benchmark_ot -s solver1 -d dataset2 --max-runs 10 --n-repetitions 10


Use ``benchopt run -h`` for more details about these options, or visit https://benchopt.github.io/api.html.

.. |Build Status| image:: https://github.com/tomMoral/benchmark_ot/workflows/Tests/badge.svg
   :target: https://github.com/tomMoral/benchmark_ot/actions
.. |Python 3.6+| image:: https://img.shields.io/badge/python-3.6%2B-blue
   :target: https://www.python.org/downloads/release/python-360/
