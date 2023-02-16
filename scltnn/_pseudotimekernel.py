# -*- coding: utf-8 -*-


import scvelo as scv

"""# setting"""

import logging
from copy import copy
from typing import Optional
from scanpy import settings

from typing import Optional

import logging
from logging import INFO, DEBUG, ERROR, WARNING, CRITICAL
from datetime import datetime, timezone, timedelta
from functools import partial, update_wrapper

class _RootLogger(logging.RootLogger):
    def __init__(self, level):
        super().__init__(level)
        self.propagate = False
        _RootLogger.manager = logging.Manager(self)

    def log(
        self,
        level: int,
        msg: str,
        *,
        extra: Optional[dict] = None,
        time: datetime = None,
        deep: Optional[str] = None,
    ) -> datetime:
     #   from cellrank import settings

        # this will correctly initialize the handles if doing
        # just from cellrank import logging
        settings.verbosity = settings.verbosity

        now = datetime.now(timezone.utc)
        time_passed: timedelta = None if time is None else now - time
        extra = {
            **(extra or {}),
           # "deep": deep if settings.verbosity.level < level else None,
            "deep": None,
            "time_passed": time_passed,
        }
        super().log(level, msg, extra=extra)
        return now

    def critical(self, msg, *, time=None, deep=None, extra=None) -> datetime:
        return self.log(CRITICAL, msg, time=time, deep=deep, extra=extra)

    def error(self, msg, *, time=None, deep=None, extra=None) -> datetime:
        return self.log(ERROR, msg, time=time, deep=deep, extra=extra)

    def warning(self, msg, *, time=None, deep=None, extra=None) -> datetime:
        return self.log(WARNING, msg, time=time, deep=deep, extra=extra)

    def info(self, msg, *, time=None, deep=None, extra=None) -> datetime:
        return self.log(INFO, msg, time=time, deep=deep, extra=extra)

    def hint(self, msg, *, time=None, deep=None, extra=None) -> datetime:
        return self.log(HINT, msg, time=time, deep=deep, extra=extra)

    def debug(self, msg, *, time=None, deep=None, extra=None) -> datetime:
        return self.log(DEBUG, msg, time=time, deep=deep, extra=extra)


class _LogFormatter(logging.Formatter):
    def __init__(
        self, fmt="{levelname}: {message}", datefmt="%Y-%m-%d %H:%M", style="{"
    ):
        super().__init__(fmt, datefmt, style)

    def format(self, record: logging.LogRecord):
        format_orig = self._style._fmt
        if record.levelno == INFO:
            self._style._fmt = "{message}"
        elif record.levelno == HINT:
            self._style._fmt = "--> {message}"
        elif record.levelno == DEBUG:
            self._style._fmt = "DEBUG: {message}"
        if record.time_passed:
            # strip microseconds
            if record.time_passed.microseconds:
                record.time_passed = timedelta(
                    seconds=int(record.time_passed.total_seconds())
                )
            if "{time_passed}" in record.msg:
                record.msg = record.msg.replace(
                    "{time_passed}", str(record.time_passed)
                )
            else:
                self._style._fmt += " ({time_passed})"
        if record.deep:
            record.msg = f"{record.msg}: {record.deep}"
        result = logging.Formatter.format(self, record)
        self._style._fmt = format_orig
        return result

def _set_log_file(settings):
    file = settings.logfile
    name = settings.logpath
    root = settings._root_logger
    h = logging.StreamHandler(file) if name is None else logging.FileHandler(name)
    h.setFormatter(_LogFormatter())
    h.setLevel(root.level)

    if len(root.handlers) == 1:
        root.removeHandler(root.handlers[0])
    elif len(root.handlers) > 1:
        raise RuntimeError("CellRank's root logger somehow got more than one handler.")

    root.addHandler(h)


settings = copy(settings)
settings._root_logger = _RootLogger(settings.verbosity)
# these 2 lines are necessary to get it working (otherwise no logger is found)
# this is a hacky way of modifying the logging, in the future, use our own
_set_log_file(settings)
settings.verbosity = settings.verbosity

"""# logging"""

from typing import Optional

import logging
from logging import INFO, DEBUG, ERROR, WARNING, CRITICAL
from datetime import datetime, timezone, timedelta
from functools import partial, update_wrapper


HINT = (INFO + DEBUG) // 2
logging.addLevelName(HINT, "HINT")



_DEPENDENCIES_NUMERICS = [
    "scanpy",
    "anndata",
    "numpy",
    "numba",
    "scipy",
    "pandas",
    "pygpcca",
    ("sklearn", "scikit-learn"),
    "statsmodels",
    ("igraph", "python-igraph"),
    "scvelo",
    "pygam",
]


_DEPENDENCIES_PLOTTING = ["matplotlib", "seaborn"]


def _versions_dependencies(dependencies):
  pass
    # this is not the same as the requirements!
    #for mod in dependencies:
    #    mod_name, dist_name = mod if isinstance(mod, tuple) else (mod, mod)
    #    try:
    #        imp = __import__(mod_name)
    #        if mod == "cellrank":
    #            yield dist_name, imp.__full_version__
    #        else:
    #            yield dist_name, imp.__version__
    #    except (ImportError, AttributeError):
    #        pass


def print_versions():
    """Print package versions that might influence the numerical and plotting results."""

    modules = ["cellrank"] + _DEPENDENCIES_NUMERICS + _DEPENDENCIES_PLOTTING
    print(
        " ".join(f"{mod}=={ver}" for mod, ver in _versions_dependencies(modules)),
        file=settings.logfile,
    )


def print_version_and_date():
    """
    Print version and date.
    Useful for starting a notebook so you see when you started working.
    """

   # from cellrank import settings, __full_version__

    print(
        f"Running CellRank , on {datetime.now():%Y-%m-%d %H:%M}.",
        file=settings.logfile,
    )


def _copy_docs_and_signature(fn):
    return partial(update_wrapper, wrapped=fn, assigned=["__doc__", "__annotations__"])


def error(
    msg: str,
    *,
    time: datetime = None,
    deep: Optional[str] = None,
    extra: Optional[dict] = None,
) -> datetime:
    """
    Log message with specific level and return current time.
    Parameters
    ----------
    msg
        Message to display.
    time
        A time in the past. If this is passed, the time difference from then
        to now is appended to `msg` as ` (HH:MM:SS)`.
        If `msg` contains `{time_passed}`, the time difference is instead
        inserted at that position.
    deep
        If the current verbosity is higher than the log function’s level,
        this gets displayed as well
    extra
        Additional values you can specify in `msg` like `{time_passed}`.
    Returns
    -------
    :class:`datetime.datetime`
        The current time.
    """

    return settings._root_logger.error(msg, time=time, deep=deep, extra=extra)


@_copy_docs_and_signature(error)
def warning(msg: str, *, time=None, deep=None, extra=None) -> datetime:  # noqa

    return settings._root_logger.warning(msg, time=time, deep=deep, extra=extra)


@_copy_docs_and_signature(error)
def info(msg: str, *, time=None, deep=None, extra=None) -> datetime:  # noqa

    return settings._root_logger.info(msg, time=time, deep=deep, extra=extra)


@_copy_docs_and_signature(error)
def hint(msg: str, *, time=None, deep=None, extra=None) -> datetime:  # noqa

    return settings._root_logger.hint(msg, time=time, deep=deep, extra=extra)


@_copy_docs_and_signature(error)
def debug(msg: str, *, time=None, deep=None, extra=None) -> datetime:  # noqa

    return settings._root_logger.debug(msg, time=time, deep=deep, extra=extra)

"""# _docs"""

from typing import Any

from docrep import DocstringProcessor
from textwrap import dedent


_adata = """\
adata : :class:`anndata.AnnData`
    Annotated data object."""
_adata_ret = """\
:class:`anndata.AnnData`
    Annotated data object."""
_plotting = """\
figsize
    Size of the figure.
dpi
    Dots per inch.
save
    Filename where to save the plot."""
_n_jobs = """\
n_jobs
    Number of parallel jobs. If `-1`, use all available cores. If `None` or `1`, the execution is sequential."""
_parallel = f"""\
show_progress_bar
    Whether to show a progress bar. Disabling it may slightly improve performance.
{_n_jobs}
backend
    Which backend to use for parallelization. See :class:`joblib.Parallel` for valid options."""
_model = """\
model
    Model based on :class:`cellrank.models.BaseModel` to fit.
    If a :class:`dict`, gene and lineage specific models can be specified. Use ``'*'`` to indicate
    all genes or lineages, for example ``{'gene_1': {'*': ...}, 'gene_2': {'lineage_1': ..., '*': ...}}``."""
_just_plots = """\
Nothing, just plots the figure. Optionally saves it based on ``save``."""
_plots_or_returns_models = """\
None
    If ``return_models = False``, just plots the figure and optionally saves it based on ``save``.
Dict[str, Dict[str, :class:`cellrank.models.BaseModel`]]
    Otherwise returns the fitted models as ``{'gene_1': {'lineage_1': <model_11>, ...}, ...}``.
    Models which have failed will be instances of :class:`cellrank.models.FailedModel`."""
_backward = """\
backward
    Direction of the process."""
_eigen = """\
which
    How to sort the eigenvalues. Valid option are:
        - `'LR'` - the largest real part.
        - `'LM'` - the largest magnitude.
alpha
    Used to compute the *eigengap*. ``alpha`` is the weight given to the deviation of an eigenvalue from one."""
_n_cells = """\
n_cells
    Number of most likely cells from each macrostate to select."""
_fit = """\
n_lineages
    Number of lineages. If `None`, it will be determined automatically.
cluster_key
    Match computed states against pre-computed clusters to annotate the states.
    For this, provide a key from :attr:`adata` ``.obs`` where cluster labels have been computed.
keys
    Determines which %(initial_or_terminal)s states to use by passing their names.
    Further, %(initial_or_terminal)s states can be combined. If e.g. the %(terminal)s states are
    ['Neuronal_1', 'Neuronal_1', 'Astrocytes', 'OPC'], then passing ``keys=['Neuronal_1, Neuronal_2', 'OPC']``
    means that the two neuronal %(terminal)s states are treated as one and the 'Astrocyte' state is excluded."""
_density_correction = (
    "Optionally, we apply a density correction as described in :cite:`coifman:05`, "
    "where we use the implementation of :cite:`haghverdi:16`."
)
_time_range = """\
time_range
    Specify start and end times:
        - If a :class:`tuple`, it specifies the minimum and maximum pseudotime. Both values can be `None`,
          in which case the minimum is the earliest pseudotime and the maximum is automatically determined.
        - If a :class:`float`, it specifies the maximum pseudotime."""

_velocity_mode = """\
mode
    How to compute transition probabilities. Valid options are:
        - `{m.DETERMINISTIC!r}` - deterministic computation that doesn't propagate uncertainty.
        - `{m.MONTE_CARLO!r}` - Monte Carlo average of randomly sampled velocity vectors.
        - `{m.STOCHASTIC!r}` - second order approximation, only available when :mod:`jax` is installed."""
_velocity_backward_mode = """\
backward_mode
    Only matters if initialized as :attr:`backward` ``= True``.  Valid options are:
        - `{b.TRANSPOSE!r}` - compute transitions from neighboring cells :math:`j` to cell :math:`i`.
        - `{b.NEGATE!r}` - negate the velocity vector."""
_velocity_backward_mode_high_lvl = """\
backward_mode
    How to compute the backward transitions. Valid options are:
        - `{b.TRANSPOSE!r}` - compute transitions from neighboring cells :math:`j` to cell :math:`i`.
        - `{b.NEGATE!r}` - negate the velocity vector."""
_copy = """Return a copy of self."""
_initial = "initial"
_terminal = "terminal"
_model_callback = """\
callback
    Function which takes a :class:`cellrank.models.BaseModel` and some keyword arguments
    for :meth:`cellrank.models.BaseModel.prepare` and returns the prepared model.
    Can be specified in gene- and lineage-specific manner, similarly to :attr:`model`."""
_genes = """\
genes
    Genes in :attr:`anndata.AnnData.var_names` or in :attr:`anndata.AnnData.raw.var_names`, if ``use_raw = True``."""
_softmax_scale = """\
softmax_scale
    Scaling parameter for the softmax. If `None`, it will be estimated using ``1 / median(correlations)``.
    The idea behind this is to scale the softmax to counter everything tending to orthogonality in high dimensions."""
_time_mode = """\
mode
    Valid options are:
        - `'embedding'` - plot the embedding while coloring in continuous or categorical observations.
        - `'time'` - plot the pseudotime on x-axis and the probabilities/memberships on y-axis."""
_write_to_adata = """\
Updates the :attr:`adata` with the following fields:
        - ``.obsp['{{key}}']`` - the transition matrix.
        - ``.uns['{{key}}_params']`` - parameters used for the calculation."""
_en_cutoff_p_thresh = """\
en_cutoff
    If ``cluster_key`` is given, this parameter determines when an approximate recurrent class will
    be labeled as *'Unknown'*, based on the entropy of the distribution of cells over transcriptomic clusters.
p_thresh
    If cell cycle scores were provided, a *Wilcoxon rank-sum test* is conducted to identify cell-cycle states.
    If the test returns a positive statistic and a p-value smaller than ``p_thresh``, a warning will be issued."""
_return_models = """\
return_models
    If `True`, return the fitted models for each gene in ``genes`` and lineage in ``lineages``."""
_basis = """\
basis
    Basis to use when ``mode = 'embedding'``. If `None`, use `'umap'`."""
_velocity_scheme = """\
scheme
    Similarity measure between cells as described in :cite:`li:20`. Can be one of the following:
        - `{s.CORRELATION!r}` - :class:`cellrank.kernels.utils.Correlation`.
        - `{s.COSINE!r}` - :class:`cellrank.kernels.utils.Cosine`.
        - `{s.DOT_PRODUCT!r}` - :class:`cellrank.kernels.utils.DotProduct`.
    Alternatively, any function can be passed as long as it follows the signature of
    :meth:`cellrank.kernels.utils.SimilarityABC.__call__`."""
_cond_num = """\
compute_cond_num
    Whether to compute condition number of the transition matrix. Note that this might be costly,
    since it does not use sparse implementation."""
_soft_scheme_fmt = """\
b
    The growth rate of generalized logistic function.{}
nu
    Affects near which asymptote maximum growth occurs.{}"""
_rw_ixs = """\
Can be specified as:
        - :class:`dict` - dictionary with 1 key in :attr:`anndata.AnnData.obs` with values corresponding
          to either 1 or more clusters (if the column is categorical) or a :class:`tuple` specifying
          `[min, max]` interval from which to select the indices.
        - :class:`typing.Sequence` - sequence of cell ids in :attr:`anndata.AnnData.obs_names`.
"""
_gene_symbols = """\
gene_symbols
    Key in :attr:`anndata.AnnData.var` to use instead of :attr:`anndata.AnnData.var_names`."""
_absorption_utils = """\
solver
    Solver to use for the linear problem. Options are `'direct', 'gmres', 'lgmres', 'bicgstab' or 'gcrotmk'`
    when ``use_petsc = False`` or one of :class:`petsc4py.PETSc.KPS.Type` otherwise.
    Information on the :mod:`scipy` iterative solvers can be found in :func:`scipy.sparse.linalg` or for
    :mod:`petsc4py` solver `here <https://petsc.org/release/overview/linear_solve_table/>`__.
use_petsc
    Whether to use solvers from :mod:`petsc4py` or :mod:`scipy`. Recommended for large problems.
    If no installation is found, defaults to :func:`scipy.sparse.linalg.gmres`.
n_jobs
    Number of parallel jobs to use when using an iterative solver.
backend
    Which backend to use for multiprocessing. See :class:`joblib.Parallel` for valid options.
show_progress_bar
    Whether to show progress bar. Only used when ``solver != 'direct'``.
tol
    Convergence tolerance for the iterative solver. The default is fine for most cases, only consider
    decreasing this for severely ill-conditioned matrices.
preconditioner
    Preconditioner to use, only available when ``use_petsc = True``. For valid options, see
    `here <https://petsc.org/release/docs/manual/ksp/?highlight=pctype#preconditioners>`__.
    We recommend the `'ilu'` preconditioner for badly conditioned problems."""


def inject_docs(**kwargs: Any):  # noqa
    def decorator(obj):
        obj.__doc__ = dedent(obj.__doc__).format(**kwargs)
        return obj

    def decorator2(obj):
        obj.__doc__ = dedent(kwargs["__doc__"])
        return obj

    if isinstance(kwargs.get("__doc__", None), str) and len(kwargs) == 1:
        return decorator2

    return decorator


d = DocstringProcessor(
    plotting=_plotting,
    n_jobs=_n_jobs,
    parallel=_parallel,
    model=_model,
    adata=_adata,
    adata_ret=_adata_ret,
    just_plots=_just_plots,
    backward=_backward,
    initial=_initial,
    terminal=_terminal,
    eigen=_eigen,
    initial_or_terminal=f"{_initial} or {_terminal}",
    n_cells=_n_cells,
    fit=_fit,
    copy=_copy,
    density_correction=_density_correction,
    time_range=_time_range,
    velocity_mode=_velocity_mode,
    velocity_backward_mode=_velocity_backward_mode,
    velocity_backward_mode_high_lvl=_velocity_backward_mode_high_lvl,
    model_callback=_model_callback,
    genes=_genes,
    softmax_scale=_softmax_scale,
    time_mode=_time_mode,
    write_to_adata=_write_to_adata,
    en_cutoff_p_thresh=_en_cutoff_p_thresh,
    return_models=_return_models,
    plots_or_returns_models=_plots_or_returns_models,
    basis=_basis,
    velocity_scheme=_velocity_scheme,
    cond_num=_cond_num,
    soft_scheme=_soft_scheme_fmt.format("", "", ""),
    soft_scheme_kernel=_soft_scheme_fmt.format(
        *([" Only used when ``threshold_scheme = 'soft'``."] * 3)
    ),
    rw_ixs=_rw_ixs,
    gene_symbols=_gene_symbols,
    absorption_utils=_absorption_utils,
)

"""# _parallelize"""

"""Module used to parallelize model fitting."""

from typing import Any, Union, Callable, Optional, Sequence

import joblib as jl
from threading import Thread
from multiprocessing import Manager, cpu_count

import numpy as np
from scipy.sparse import issparse, spmatrix


def parallelize(
    callback: Callable[[Any], Any],
    collection: Union[spmatrix, Sequence[Any]],
    n_jobs: Optional[int] = None,
    n_split: Optional[int] = None,
    unit: str = "",
    as_array: bool = True,
    use_ixs: bool = False,
    backend: str = "loky",
    extractor: Optional[Callable[[Any], Any]] = None,
    show_progress_bar: bool = True,
) -> Any:
    """
    Parallelize function call over a collection of elements.
    Parameters
    ----------
    callback
        Function to parallelize.
    collection
        Sequence of items which to chunkify or an already .
    n_jobs
        Number of parallel jobs.
    n_split
        Split ``collection`` into ``n_split`` chunks. If `None`, split into ``n_jobs`` chunks.
    unit
        Unit of the progress bar.
    as_array
        Whether to convert the results not :class:`numpy.ndarray`.
    use_ixs
        Whether to pass indices to the callback.
    backend
        Which backend to use for multiprocessing. See :class:`joblib.Parallel` for valid options.
    extractor
        Function to apply to the result after all jobs have finished.
    show_progress_bar
        Whether to show a progress bar.
    Returns
    -------
    The result depending on ``callable``, ``extractor`` and ``as_array``.
    """

    if show_progress_bar:
        try:
            import ipywidgets
            from tqdm.auto import tqdm
        except ImportError:
            try:
                from tqdm.std import tqdm
            except ImportError:
                tqdm = None
    else:
        tqdm = None

    def update(pbar, queue, n_total):
        n_finished = 0
        while n_finished < n_total:
            try:
                res = queue.get()
            except EOFError as e:
                if not n_finished != n_total:
                    raise RuntimeError(
                        f"Finished only `{n_finished}` out of `{n_total}` tasks.`"
                    ) from e
                break
            assert res in (None, (1, None), 1)  # (None, 1) means only 1 job
            if res == (1, None):
                n_finished += 1
                if pbar is not None:
                    pbar.update()
            elif res is None:
                n_finished += 1
            elif pbar is not None:
                pbar.update()

        if pbar is not None:
            pbar.close()

    def wrapper(*args, **kwargs):
        if pass_queue and show_progress_bar:
            pbar = (
                None
                if tqdm is None
                else tqdm(total=col_len, unit=unit, mininterval=0.125)
            )
            queue = Manager().Queue()
            thread = Thread(target=update, args=(pbar, queue, len(collections)))
            thread.start()
        else:
            pbar, queue, thread = None, None, None

        res = jl.Parallel(n_jobs=n_jobs, backend=backend)(
            jl.delayed(callback)(
                *((i, cs) if use_ixs else (cs,)),
                *args,
                **kwargs,
                queue=queue,
            )
            for i, cs in enumerate(collections)
        )

        res = np.array(res) if as_array else res
        if thread is not None:
            thread.join()

        return res if extractor is None else extractor(res)

    col_len = collection.shape[0] if issparse(collection) else len(collection)
    n_jobs = _get_n_cores(n_jobs, col_len)
    if n_split is None:
        n_split = n_jobs

    if issparse(collection):
        n_split = max(1, min(n_split, collection.shape[0]))
        if n_split == collection.shape[0]:
            collections = [collection[[ix], :] for ix in range(collection.shape[0])]
        else:
            step = collection.shape[0] // n_split
            ixs = [
                np.arange(i * step, min((i + 1) * step, collection.shape[0]))
                for i in range(n_split)
            ]
            ixs[-1] = np.append(
                ixs[-1], np.arange(ixs[-1][-1] + 1, collection.shape[0])
            )

            collections = [collection[ix, :] for ix in filter(len, ixs)]
    else:
        collections = list(filter(len, np.array_split(collection, n_split)))

    n_split = len(collections)
    n_jobs = min(n_jobs, n_split)
    pass_queue = not hasattr(callback, "py_func")  # we'd be inside a numba function

    return wrapper


def _get_n_cores(n_cores: Optional[int], n_jobs: Optional[int]) -> int:
    """
    Make number of cores a positive integer.
    Parameters
    ----------
    n_cores
        Number of cores to use.
    n_jobs.
        Number of jobs. This is just used to determine if the collection is a singleton.
        If `1`, always returns `1`.
    Returns
    -------
    Positive integer corresponding to how many cores to use.
    """
    if n_cores == 0:
        raise ValueError("Number of cores cannot be `0`.")
    if n_jobs == 1 or n_cores is None:
        return 1
    if n_cores < 0:
        return cpu_count() + 1 + n_cores

    return n_cores

"""# from cellrank.kernels.utils._pseudotime_scheme"""

# Commented out IPython magic to ensure Python compatibility.
#from cellrank.kernels.utils._pseudotime_scheme import (
#    ThresholdSchemeABC,
#    HardThresholdScheme,
#    SoftThresholdScheme,
#    CustomThresholdScheme,
#)

from typing import Any, Tuple, Callable, Optional

from abc import ABC, abstractmethod

import numpy as np
from scipy.sparse import csr_matrix

class ThresholdSchemeABC(ABC):
    """Base class for all connectivity biasing schemes."""

    @d.get_summary(base="pt_scheme")
    @d.get_sections(base="pt_scheme", sections=["Parameters", "Returns"])
    @abstractmethod
    def __call__(
        self,
        cell_pseudotime: float,
        neigh_pseudotime: np.ndarray,
        neigh_conn: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Calculate biased connections for a given cell.
        Parameters
        ----------
        cell_pseudotime
            Pseudotime of the current cell.
        neigh_pseudotime
            Array of shape ``(n_neighbors,)`` containing pseudotime of neighbors.
        neigh_conn
            Array of shape ``(n_neighbors,)`` containing connectivities of the current cell and its neighbors.
        Returns
        -------
        Array of shape ``(n_neighbors,)`` containing the biased connectivities.
        """

    def _bias_knn_helper(
        self,
        ixs: np.ndarray,
        conn: csr_matrix,
        pseudotime: np.ndarray,
        queue=None,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        indices, indptr, data = [], [], []

        for i in ixs:
            row = conn[i]
            biased_row = self(
                pseudotime[i], pseudotime[row.indices], row.data, **kwargs
            )
            if np.shape(biased_row) != row.data.shape:
                raise ValueError(
                    f"Expected row of shape `{row.data.shape}`, found `{np.shape(biased_row)}`."
                )

            data.extend(biased_row)
            indices.extend(row.indices)
            indptr.append(conn.indptr[i])

            if queue is not None:
                queue.put(1)

        if i == conn.shape[0] - 1:
            indptr.append(conn.indptr[-1])
        if queue is not None:
            queue.put(None)

        return np.array(data), np.array(indices), np.array(indptr)

    @d.dedent
    def bias_knn(
        self,
        conn: csr_matrix,
        pseudotime: np.ndarray,
        n_jobs: Optional[int] = None,
        backend: str = "loky",
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> csr_matrix:
        """
        Bias cell-cell connectivities of a KNN graph.
        Parameters
        ----------
        conn
            Sparse matrix of shape ``(n_cells, n_cells)`` containing the nearest neighbor connectivities.
        pseudotime
            Pseudotemporal ordering of cells.
#         %(parallel)s
        Returns
        -------
        The biased connectivities.
        """
        res = parallelize(
            self._bias_knn_helper,
            np.arange(conn.shape[0]),
            as_array=False,
            unit="cell",
            n_jobs=n_jobs,
            backend=backend,
            show_progress_bar=show_progress_bar,
        )(conn, pseudotime, **kwargs)
        data, indices, indptr = zip(*res)

        conn = csr_matrix(
            (np.concatenate(data), np.concatenate(indices), np.concatenate(indptr))
        )
        conn.eliminate_zeros()

        return conn

    def __repr__(self):
        return f"<{self.__class__.__name__}>"

    def __str__(self):
        return repr(self)


class HardThresholdScheme(ThresholdSchemeABC):
    """
    Thresholding scheme inspired by *Palantir* :cite:`setty:19`.
    Note that this won't exactly reproduce the original *Palantir* results, for three reasons:
        - *Palantir* computes the KNN graph in a scaled space of diffusion components.
        - *Palantir* uses its own pseudotime to bias the KNN graph which is not implemented here.
        - *Palantir* uses a slightly different mechanism to ensure the graph remains connected when removing edges
          that point into the "pseudotime past".
    """

    @d.dedent
    def __call__(
        self,
        cell_pseudotime: float,
        neigh_pseudotime: np.ndarray,
        neigh_conn: np.ndarray,
        frac_to_keep: float = 0.3,
    ) -> np.ndarray:
        """
        Convert the undirected graph of cell-cell similarities into a directed one by removing "past" edges.
        This uses a pseudotemporal measure to remove graph-edges that point into the pseudotime-past. For each cell,
        it keeps the closest neighbors, even if they are in the pseudotime past, to make sure the graph remains
        connected.
        Parameters
        ----------
#         %(pt_scheme.parameters)s
        frac_to_keep
            The `frac_to_keep` * n_neighbors closest neighbors (according to graph connectivities) are kept, no matter
            whether they lie in the pseudotemporal past or future. `frac_to_keep` needs to fall within the
            interval `[0, 1]`.
        Returns
        -------
#         %(pt_scheme.returns)s
        """
        if not (0 <= frac_to_keep <= 1):
            raise ValueError(
                f"Expected `frac_to_keep` to be in `[0, 1]`, found `{frac_to_keep}`."
            )

        k_thresh = max(0, min(30, int(np.floor(len(neigh_conn) * frac_to_keep))))
        ixs = np.flip(np.argsort(neigh_conn))
        close_ixs, far_ixs = ixs[:k_thresh], ixs[k_thresh:]

        mask_keep = cell_pseudotime <= neigh_pseudotime[far_ixs]
        far_ixs_keep = far_ixs[mask_keep]

        biased_conn = np.zeros_like(neigh_conn)
        biased_conn[close_ixs] = neigh_conn[close_ixs]
        biased_conn[far_ixs_keep] = neigh_conn[far_ixs_keep]

        return biased_conn


class SoftThresholdScheme(ThresholdSchemeABC):
    """
    Thresholding scheme inspired by :cite:`stassen:21`.
    The idea is to downweight edges that points against the direction of increasing pseudotime. Essentially, the
    further "behind" a query cell is in pseudotime with respect to the current reference cell, the more penalized will
    be its graph-connectivity.
    """

    @d.dedent
    def __call__(
        self,
        cell_pseudotime: float,
        neigh_pseudotime: np.ndarray,
        neigh_conn: np.ndarray,
        b: float = 10.0,
        nu: float = 0.5,
    ) -> np.ndarray:
        """
        Bias the connectivities by downweighting ones to past cells.
        This function uses `generalized logistic regression
        <https://en.wikipedia.org/wiki/Generalized_logistic_function>`_ to weight the past connectivities.
        Parameters
        ----------
#         %(pt_scheme.parameters)s
#         %(soft_scheme)s
        Returns
        -------
#         %(pt_scheme.returns)s
        """
        past_ixs = np.where(neigh_pseudotime < cell_pseudotime)[0]
        if not len(past_ixs):
            return neigh_conn

        weights = np.ones_like(neigh_conn)

        dt = cell_pseudotime - neigh_pseudotime[past_ixs]
        weights[past_ixs] = 2.0 / ((1.0 + np.exp(b * dt)) ** (1.0 / nu))

        return neigh_conn * weights


class CustomThresholdScheme(ThresholdSchemeABC):
    """
    Class that wraps a user supplied scheme.
    Parameters
    ----------
    callback
        Function which returns the biased connectivities.
    """

    def __init__(
        self,
        callback: Callable[
            [float, np.ndarray, np.ndarray, np.ndarray, Any], np.ndarray
        ],
    ):
        super().__init__()
        self._callback = callback

    @d.dedent
    def __call__(
        self,
        cell_pseudotime: float,
        neigh_pseudotime: np.ndarray,
        neigh_conn: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        """
#         %(pt_scheme.summary)s
        Parameters
        ----------
#         %(pt_scheme.parameters)s
        kwargs
            Additional keyword arguments.
        Returns
        -------
#         %(pt_scheme.returns)s
        """  # noqa: D400
        return self._callback(cell_pseudotime, neigh_pseudotime, neigh_conn, **kwargs)

"""# _enum."""

from typing import Any, Dict, Type, Tuple, Callable
from typing_extensions import Literal

from abc import ABC, ABCMeta
from enum import Enum, EnumMeta
from functools import wraps

_DEFAULT_BACKEND = "loky"
Backend_t = Literal["loky", "multiprocessing", "threading"]


class PrettyEnum(Enum):
    """Enum with a pretty :meth:`__str__` and :meth:`__repr__`."""

    @property
    def v(self) -> Any:
        """Alias for :attr`value`."""
        return self.value

    def __repr__(self) -> str:
        return f"{self.value!r}"

    def __str__(self) -> str:
        return f"{self.value!s}"


def _pretty_raise_enum(cls: Type["ErrorFormatterABC"], func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> "ErrorFormatterABC":
        try:
            return func(*args, **kwargs)
        except ValueError as e:
            _cls, value, *_ = args
            e.args = (cls._format(value),)
            raise e

    if not issubclass(cls, ErrorFormatterABC):
        raise TypeError(f"Class `{cls}` must be subtype of `ErrorFormatterABC`.")
    elif not len(cls.__members__):
        # empty enum, for class hierarchy
        return func

    return wrapper


class ABCEnumMeta(EnumMeta, ABCMeta):  # noqa: D101
    def __call__(cls, *args, **kwargs):  # noqa
        if getattr(cls, "__error_format__", None) is None:
            raise TypeError(
                f"Can't instantiate class `{cls.__name__}` "
                f"without `__error_format__` class attribute."
            )
        return super().__call__(*args, **kwargs)

    def __new__(  # noqa: D102
        cls, clsname: str, superclasses: Tuple[type], attributedict: Dict[str, Any]
    ):
        res = super().__new__(cls, clsname, superclasses, attributedict)
        res.__new__ = _pretty_raise_enum(res, res.__new__)
        return res


class ErrorFormatterABC(ABC):  # noqa: D101
    __error_format__ = "Invalid option `{!r}` for `{}`. Valid options are: `{}`."

    @classmethod
    def _format(cls, value) -> str:
        return cls.__error_format__.format(
            value, cls.__name__, [m.value for m in cls.__members__.values()]
        )


class ModeEnum(str, ErrorFormatterABC, PrettyEnum, metaclass=ABCEnumMeta):  # noqa: D101
    def _generate_next_value_(self, start, count, last_values):
        return str(self).lower()

"""# _utils"""

from scipy.sparse import diags, issparse, spmatrix, csr_matrix, isspmatrix_csr
import numpy as np

def _connected(c: Union[spmatrix, np.ndarray]) -> bool:
    """Check whether the undirected graph encoded by c is connected."""

    import networkx as nx

    G = nx.from_scipy_sparse_matrix(c) if issparse(c) else nx.from_numpy_array(c)

    return nx.is_connected(G)
def _irreducible(d: Union[spmatrix, np.ndarray]) -> bool:
    """Check whether the undirected graph encoded by d is irreducible."""

    import networkx as nx

    G = nx.DiGraph(d) if not isinstance(d, nx.DiGraph) else d
    try:
        it = iter(nx.strongly_connected_components(G))
        _ = next(it)
        _ = next(it)
        return False
    except StopIteration:
        return True

"""# mixins"""

from scipy.sparse.linalg import norm as sparse_norm
from numpy.linalg import norm as d_norm
from anndata import AnnData

def _read_graph_data(adata: AnnData, key: str) -> Union[np.ndarray, spmatrix]:
    """
    Read graph data from :mod:`anndata`.
    Parameters
    ----------
    adata
        Annotated data object.
    key
        Key in :attr:`anndata.AnnData.obsp`.
    Returns
    -------
    The graph data.
    """
    if key in adata.obsp:
        return adata.obsp[key]

    raise KeyError(f"Unable to find data in `adata.obsp[{key!r}]`.")
def _modify_neigh_key(key: Optional[str]) -> str:
    if key in (None, "connectivities", "distances"):
        return "neighbors"

    if key.endswith("_connectivities"):
        key = key[:-15]
    elif key.endswith("_distances"):
        key = key[:-10]
    return key

def _symmetric(
    matrix: Union[spmatrix, np.ndarray],
    ord: str = "fro",
    eps: float = 1e-4,
    only_check_sparsity_pattern: bool = False,
) -> bool:
    """Check whether the graph encoded by `matrix` is symmetric."""
    if only_check_sparsity_pattern:
        if issparse(matrix):
            return len(((matrix != 0) - (matrix != 0).T).data) == 0
        return ((matrix != 0) == (matrix != 0).T).all()

    if issparse(matrix):
        return sparse_norm((matrix - matrix.T), ord=ord) < eps
    return d_norm((matrix - matrix.T), ord=ord) < eps
def _get_neighs(
    adata: AnnData, mode: str = "distances", key: Optional[str] = None
) -> Union[np.ndarray, spmatrix]:
    if key is None:
        res = _read_graph_data(adata, mode)  # legacy behavior
    else:
        try:
            res = _read_graph_data(adata, key)
            assert isinstance(res, (np.ndarray, spmatrix))
        except (KeyError, AssertionError):
            res = _read_graph_data(adata, f"{_modify_neigh_key(key)}_{mode}")

    if not isinstance(res, (np.ndarray, spmatrix)):
        raise TypeError(
            f"Expected to find `numpy.ndarray` or `scipy.sparse.spmatrix`, found `{type(res)}`."
        )

    return res

class UnidirectionalMixin:
    """Mixin specifying that its kernel doesn't is directionless."""

    @property
    def backward(self) -> None:
        """None."""
        return None

from typing import Any, Union

from abc import ABC, abstractmethod

import numpy as np
from scipy.sparse import spdiags, spmatrix, csr_matrix

class ConnectivityMixin:
    """Mixin class that reads kNN connectivities and allows for density normalization."""

    def _read_from_adata(
        self,
        conn_key: str = "connectivities",
        check_connectivity: bool = False,
        check_symmetric: bool = True,
        **kwargs: Any,
    ) -> None:
        super()._read_from_adata(**kwargs)
        self._conn_key = conn_key
        conn = _get_neighs(self.adata, mode="connectivities", key=conn_key)
        self._conn = csr_matrix(conn).astype(np.float64, copy=False)


    def _density_normalize(
        self, matrix: Union[np.ndarray, spmatrix]
    ) -> Union[np.ndarray, spmatrix]:
        """
        Density normalization by the underlying kNN graph.
        Parameters
        ----------
        matrix
            Matrix to normalize.
        Returns
        -------
        Density normalized matrix.
        """

        q = np.asarray(self.connectivities.sum(axis=0)).squeeze()
        Q = spdiags(1.0 / q, 0, matrix.shape[0], matrix.shape[0])

        return Q @ matrix @ Q

    @property
    def connectivities(self) -> csr_matrix:
        """Underlying connectivity matrix."""
        return self._conn

"""# FlowPlotter"""

from typing import Iterable, Hashable, List, Sequence
from matplotlib import cm
from matplotlib import colors as mcolors
from pandas.core.dtypes.common import is_numeric_dtype, is_categorical_dtype
import pandas as pd 

def _convert_to_hex_colors(colors: Sequence[Any]) -> List[str]:
    if not all(mcolors.is_color_like(c) for c in colors):
        raise ValueError("Not all values are color-like.")

    return [mcolors.to_hex(c) for c in colors]

def _unique_order_preserving(iterable: Iterable[Hashable]) -> List[Hashable]:
    """Remove items from an iterable while preserving the order."""
    seen = set()
    return [i for i in iterable if i not in seen and not seen.add(i)]

def _create_categorical_colors(n_categories: Optional[int] = None):
    from scanpy.plotting.palettes import vega_20_scanpy

    cmaps = [
        mcolors.ListedColormap(vega_20_scanpy),
        cm.Accent,
        mcolors.ListedColormap(np.array(cm.Dark2.colors)[[1, 2, 4, 5, 6]]),
        cm.Set1,
        cm.Set2,
        cm.Set3,
    ]
    max_cats = sum(c.N for c in cmaps)

    if n_categories is None:
        n_categories = max_cats
    if n_categories > max_cats:
        raise ValueError(
            f"Number of categories `{n_categories}` exceeded the maximum number of colors `{max_cats}`."
        )

    colors = []
    for cmap in cmaps:
        colors += [cmap(i) for i in range(cmap.N)][: n_categories - len(colors)]
        if len(colors) == n_categories:
            return _convert_to_hex_colors(colors)

    raise RuntimeError(f"Unable to create `{n_categories}` colors.")


def _ensure_numeric_ordered(adata: AnnData, key: str) -> pd.Series:
    if key not in adata.obs.keys():
        raise KeyError(f"Unable to find data in `adata.obs[{key!r}]`.")

    exp_time = adata.obs[key].copy()
    if not is_numeric_dtype(np.asarray(exp_time)):
        try:
            exp_time = np.asarray(exp_time).astype(float)
        except ValueError as e:
            raise TypeError(
                f"Unable to convert `adata.obs[{key!r}]` of type `{infer_dtype(adata.obs[key])}` to `float`."
            ) from e

    if not is_categorical_dtype(exp_time):
        debug(f"Converting `adata.obs[{key!r}]` to `categorical`")
        exp_time = np.asarray(exp_time)
        categories = sorted(set(exp_time[~np.isnan(exp_time)]))
        if len(categories) > 100:
            raise ValueError(
                f"Unable to convert `adata.obs[{key!r}]` to `categorical` since it "
                f"would create `{len(categories)}` categories."
            )
        exp_time = pd.Series(
            pd.Categorical(
                exp_time,
                categories=categories,
                ordered=True,
            )
        )

    if not exp_time.cat.ordered:
        warning("Categories are not ordered. Using ascending order")
        exp_time.cat = exp_time.cat.as_ordered()

    exp_time = pd.Series(pd.Categorical(exp_time, ordered=True), index=adata.obs_names)
    if exp_time.isnull().any():
        raise ValueError("Series contains NaN value(s).")

    n_cats = len(exp_time.cat.categories)
    if n_cats < 2:
        raise ValueError(f"Expected to find at least `2` categories, found `{n_cats}`.")

    return exp_time

import matplotlib as mpl
def _position_legend(ax: mpl.axes.Axes, legend_loc: str, **kwargs) -> mpl.legend.Legend:
    """
    Position legend in- or outside the figure.
    Parameters
    ----------
    ax
        Ax where to position the legend.
    legend_loc
        Position of legend.
    kwargs
        Keyword arguments for :func:`matplotlib.pyplot.legend`.
    Returns
    -------
    The created legend.
    """

    if legend_loc == "center center out":
        raise ValueError("Invalid option: `'center center out'`.")
    if legend_loc == "best":
        return ax.legend(loc="best", **kwargs)

    tmp, loc = legend_loc.split(" "), ""

    if len(tmp) == 1:
        height, rest = tmp[0], []
        width = "right" if height in ("upper", "top", "center") else "left"
    else:
        height, width, *rest = legend_loc.split(" ")
        if rest:
            if len(rest) != 1:
                raise ValueError(
                    f"Expected only 1 additional modifier ('in' or 'out'), found `{list(rest)}`."
                )
            elif rest[0] not in ("in", "out"):
                raise ValueError(
                    f"Invalid modifier `{rest[0]!r}`. Valid options are: `'in', 'out'`."
                )
            if rest[0] == "in":  # ignore in, it's default
                rest = []

    if height in ("upper", "top"):
        y = 1.55 if width == "center" else 1.025
        loc += "upper"
    elif height == "center":
        y = 0.5
        loc += "center"
    elif height in ("lower", "bottom"):
        y = -0.55 if width == "center" else -0.025
        loc += "lower"
    else:
        raise ValueError(
            f"Invalid legend position on y-axis: `{height!r}`. "
            f"Valid options are: `'upper', 'top', 'center', 'lower', 'bottom'`."
        )

    if width == "left":
        x = -0.05
        loc += " right" if rest else " left"
    elif width == "center":
        x = 0.5
        if height != "center":  # causes to be like top center
            loc += " center"
    elif width == "right":
        x = 1.05
        loc += " left" if rest else " right"
    else:
        raise ValueError(
            f"Invalid legend position on x-axis: `{width!r}`. "
            f"Valid options are: `'left', 'center', 'right'`."
        )

    if rest:
        kwargs["bbox_to_anchor"] = (x, y)

    return ax.legend(loc=loc, **kwargs)

# Commented out IPython magic to ensure Python compatibility.
from typing import Any, List, Tuple, Union, Mapping, Optional, Sequence

from dataclasses import dataclass
from statsmodels.nonparametric.smoothers_lowess import lowess

from anndata import AnnData

import numpy as np
import pandas as pd
from scipy.stats import logistic
from scipy.sparse import issparse, spmatrix
from pandas.api.types import infer_dtype
from scipy.interpolate import interp1d
from pandas.core.dtypes.common import is_categorical_dtype

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.collections import PolyCollection

__all__ = ["FlowPlotter"]

Numeric_t = Union[float, int]


@dataclass(frozen=True)
class Point:
    x: float
    xt: float


@d.dedent
class FlowPlotter:
    """
    Class that plots outgoing flow for a specific cluster :cite:`mittnenzweig:21`.
    It should be able to recreate (to a high degree) figures such as Fig. 4a in the above mentioned paper.
    Parameters
    ----------
#     %(adata)s
    tmat
        Matrix of shape ``(adata.n_obs, adata.n_obs)``.
    cluster_key
        Key in :attr:`anndata.AnnData.obs` where clustering is stored.
    time_key
        Key in :attr:`anndata.AnnData.obs` where experimental time is stored.
    """

    TIME_KEY = "time"

    def __init__(
        self,
        adata: AnnData,
        tmat: Union[np.ndarray, spmatrix],
        cluster_key: str,
        time_key: str,
    ):
        self._adata = adata
        self._tmat = tmat
        self._ckey = cluster_key
        self._tkey = time_key

        self._cluster: Optional[str] = None
        self._clusters: Optional[Sequence[Any]] = None

        self._flow: Optional[pd.DataFrame] = None
        self._cmat: Optional[pd.DataFrame] = None

        if self._ckey not in self._adata.obs:
            raise KeyError(f"Unable to find clusters in `adata.obs[{self._ckey!r}]`.")
        if not is_categorical_dtype(self._adata.obs[self._ckey]):
            raise TypeError(
                f"Expected `adata.obs[{self._ckey!r}]` to be categorical, "
                f"found `{infer_dtype(self._adata.obs[self._ckey])}`."
            )
        self._adata.obs[self._tkey] = _ensure_numeric_ordered(self._adata, self._tkey)

    def prepare(
        self,
        cluster: str,
        clusters: Optional[Sequence[Any]] = None,
        time_points: Optional[Sequence[Numeric_t]] = None,
    ) -> "FlowPlotter":
        """
        Prepare itself for plotting by computing flow and contingency matrix.
        Parameters
        ----------
        cluster
            Source cluster for flow calculation.
        clusters
            Target clusters for flow calculation. If `None`, use all clusters.
        time_points
            Restrict flow calculation only to these time points. If `None`, use all time points.
        Returns
        -------
        Returns self and modifies internal internal attributes.
        """
        if clusters is None:
            self._clusters = self.clusters.cat.categories
        else:
            clusters = _unique_order_preserving([cluster] + list(clusters))
            mask = self.clusters.isin(clusters).values

            self._adata = self._adata[mask]
            if not self._adata.n_obs:
                raise ValueError("No valid clusters have been selected.")
            self._tmat = self._tmat[mask, :][:, mask]
            self._clusters = [c for c in clusters if c in self.clusters.cat.categories]

        if cluster not in self._clusters:
            raise ValueError(f"Invalid source cluster `{cluster!r}`.")

        if len(self._clusters) < 2:
            raise ValueError(
                f"Expected at least `2` clusters, found `{len(clusters)}`."
            )

        if time_points is not None:
            time_points = _unique_order_preserving(time_points)
            if len(time_points) < 2:
                raise ValueError(
                    f"Expected at least `2` time points, found `{len(time_points)}`."
                )

            mask = self.time.isin(time_points)

            self._adata = self._adata[mask]
            if not self._adata.n_obs:
                raise ValueError("No valid time points have been selected.")
            self._tmat = self._tmat[mask, :][:, mask]

        time_points = list(
            zip(self.time.cat.categories[:-1], self.time.cat.categories[1:])
        )

        logg.info(
            f"Computing flow from `{cluster}` into `{len(self._clusters) - 1}` cluster(s) "
            f"in `{len(time_points)}` time points"
        )
        self._cluster = cluster
        self._cmat = self.compute_contingency_matrix()
        self._flow = self.compute_flow(time_points, cluster)

        return self

    def compute_flow(
        self,
        time_points: Sequence[Tuple[Numeric_t, Numeric_t]],
        cluster: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Compute outgoing flow.
        Parameters
        ----------
        time_points
            Time point pair for which to calculate the flow.
        cluster
            Cluster for which to calculate the outgoing flow. If `None`, calculate the flow for all clusters.
        Returns
        -------
        Dataframe of shape ``(n_time_points, n_clusters)`` if ``cluster != None`` or
        a dataframe of shape ``(n_time_points * n_clusters, n_clusters)`` otherwise.
        The dataframe's index is a multi-index and the 1st level corresponds to time, the 2nd level to source clusters.
        """

        def default_helper(t1: Numeric_t, t2: Numeric_t) -> pd.DataFrame:
            subset, row_cls, col_cls = self._get_time_subset(t1, t2)

            df = pd.DataFrame(subset.A if issparse(subset) else subset)
            df = df.groupby(row_cls).sum().T.groupby(col_cls).sum().T

            res = pd.DataFrame(np.zeros((n, n)), index=categories, columns=categories)
            res.loc[df.index, df.columns] = df
            res.fillna(0, inplace=True)

            return res

        def cluster_helper(t1: Numeric_t, t2: Numeric_t) -> pd.DataFrame:
            subset, row_cls, col_cls = self._get_time_subset(t1, t2, cluster=cluster)

            df = pd.DataFrame(subset.A if issparse(subset) else subset).sum(0)
            df = df.groupby(col_cls).sum()
            df = pd.DataFrame([df], index=[cluster], columns=df.index)

            res = pd.DataFrame(np.zeros((1, n)), index=[cluster], columns=categories)
            res.loc[df.index, df.columns] = df
            res.fillna(0, inplace=True)

            return res

        categories = self.clusters.cat.categories
        n = len(categories)
        callback = cluster_helper if cluster is not None else default_helper
        flows, times = [], []

        for t1, t2 in time_points:
            flow = callback(t1, t2)
            times.extend([t1] * len(flow))
            flows.append(flow)

        flow = pd.concat(flows)
        flow.set_index([times, flow.index], inplace=True)
        flow /= flow.sum(1).values[:, None]
        flow.fillna(0, inplace=True)

        return flow

    def compute_contingency_matrix(self) -> pd.DataFrame:
        """Row-normalized contingency matrix of shape ``(n_clusters, n_time_points)``."""
        cmat = pd.crosstab(self.clusters, self.time)
        return (cmat / cmat.sum(0).values[None, :]).fillna(0)

    @d.get_sections(base="flow", sections=["Parameters"])
    def plot(
        self,
        min_flow: float = 0,
        remove_empty_clusters: bool = True,
        ascending: Optional[bool] = False,
        alpha: float = 0.8,
        xticks_step_size: Optional[int] = 1,
        legend_loc: Optional[str] = "upper right out",
        figsize: Optional[Tuple[float, float]] = None,
        dpi: Optional[int] = None,
    ) -> plt.Axes:
        """
        Plot outgoing flow.
        Parameters
        ----------
        min_flow
            Only show flow edges with flow greater than this value. Flow values are always in `[0, 1]`.
        remove_empty_clusters
            Whether to remove clusters with no incoming flow edges.
        ascending
            Whether to sort the cluster by ascending or descending incoming flow.
            If `None`, use the order as in defined by ``clusters``.
        alpha
            Alpha value for cell proportions.
        xticks_step_size
            Show only every other *n-th* tick on the x-axis. If `None`, don't show any ticks.
        legend_loc
            Position of the legend. If `None`, do not show the legend.
        Returns
        -------
        The axes object.
        """
        if self._flow is None or self._cmat is None:
            raise RuntimeError(
                "Compute flow and contingency matrix first as `.prepare()`."
            )

        flow, cmat = self._flow, self._cmat
        try:
            if remove_empty_clusters:
                self._remove_min_clusters(min_flow)
            logg.info(
                f"Plotting flow from `{self._cluster}` into `{len(self._flow.columns) - 1}` cluster(s) "
                f"in `{len(self._cmat.columns) - 1}` time points"
            )
            return self._plot(
                self._rename_times(),
                ascending=ascending,
                min_flow=min_flow,
                alpha=alpha,
                xticks_step_size=xticks_step_size,
                legend_loc=legend_loc,
                figsize=figsize,
                dpi=dpi,
            )
        finally:
            self._flow = flow
            self._cmat = cmat

    def _get_time_subset(
        self, t1: Numeric_t, t2: Numeric_t, cluster: Optional[str] = None
    ) -> Tuple[Union[np.ndarray, spmatrix], pd.Series, pd.Series]:
        if cluster is None:
            row_ixs = np.where(self.time == t1)[0]
        else:
            row_ixs = np.where((self.time == t1) & (self.clusters == cluster))[0]

        col_ixs = np.where(self.time == t2)[0]
        row_cls = self.clusters.values[row_ixs]
        col_cls = self.clusters.values[col_ixs]

        return self._tmat[row_ixs, :][:, col_ixs], row_cls, col_cls

    def _remove_min_clusters(self, min_flow: float) -> None:
        logg.debug("Removing clusters with no incoming flow edges")
        columns = (self._flow.loc[(slice(None), self._cluster), :] > min_flow).any()
        columns = columns[columns].index
        if not len(columns):
            raise ValueError(
                "After removing clusters with no incoming flow edges, none remain."
            )
        self._flow = self._flow[columns]

    def _rename_times(self) -> Sequence[Numeric_t]:
        # make sure we have enough horizontal space to draw the flow (i.e. time points are at least 1 unit apart)
        old_times = self._cmat.columns
        tmp = np.array(old_times)
        tmp = (tmp - tmp.min()) / (tmp.max() - tmp.min())
        tmp /= np.min(tmp[1:] - tmp[:-1])
        time_mapper = dict(zip(old_times, tmp))
        self._flow.index = pd.MultiIndex.from_tuples(
            [(time_mapper[t], c) for t, c in self._flow.index]
        )
        self._cmat.columns = tmp
        return old_times

    def _order_clusters(
        self, cluster: str, ascending: Optional[bool] = False
    ) -> Tuple[List[Any], List[Any]]:
        if ascending is not None:
            tmp = [[], []]
            total_flow = (
                self._flow.loc[(slice(None), cluster), :]
                .sum()
                .sort_values(ascending=ascending)
            )
            for i, c in enumerate(c for c in total_flow.index if c != cluster):
                tmp[i % 2].append(c)
            return tmp[0][::-1], tmp[1]

        clusters = [c for c in self._clusters if c != cluster]
        return clusters[: len(clusters) // 2], clusters[len(clusters) // 2 :]

    def _calculate_y_offsets(
        self, clusters: Sequence[Any], delta: float = 0.2
    ) -> Mapping[Any, float]:
        offset = [0]
        for i in range(1, len(clusters)):
            offset.append(
                offset[-1]
                + delta
                + np.max(self._cmat.loc[clusters[i]] + self._cmat.loc[clusters[i - 1]])
            )
        return dict(zip(clusters, offset))

    def _plot_smoothed_proportion(
        self,
        ax: plt.Axes,
        clusters: Sequence[Any],
        y_offset: Mapping[Any, float],
        alpha: float = 0.8,
    ) -> Tuple[Mapping[Any, np.ndarray], Mapping[Any, PolyCollection]]:
        start_t, end_t = self._cmat.columns.min(), self._cmat.columns.max()
        x = np.array(self._cmat.columns)  # fitting
        # extrapolation
        e = np.linspace(start_t, end_t, int(1 + (end_t - start_t) * 100))

        smoothed_proportion, handles = {}, {}
        for clust in clusters:
            y = self._cmat.loc[clust]
            f = interp1d(x, y)
            fe = f(e)
            lo = lowess(fe, e, frac=0.3, is_sorted=True, return_sorted=False)
            smoothed_proportion[clust] = lo

            handles[clust] = ax.fill_between(
                e,
                y_offset[clust] + lo,
                y_offset[clust] - lo,
                color=self.cmap[clust],
                label=clust,
                alpha=alpha,
                edgecolor=None,
            )

        return smoothed_proportion, handles

    def _draw_flow_edge(
        self,
        ax,
        x1: Point,
        x2: Point,
        y1: Point,
        y2: Point,
        start_color: Tuple[float, float, float],
        end_color: Tuple[float, float, float],
        flow: float,
        alpha: float = 0.8,
    ) -> None:
        # transcribed from: https://github.com/tanaylab/embflow/blob/main/scripts/generate_paper_figures/plot_vein.r
        dx = x2.xt - x1.x
        dy = y2.xt - y1.x
        dxt = x2.x - x1.x
        dyt = y2.x - y1.xt

        start_color = np.asarray(to_rgb(start_color))
        end_color = np.asarray(to_rgb(end_color))
        delta = 0.05

        beta0 = _lcdf(0)
        beta_f = _lcdf(1) - _lcdf(0)

        rs = np.arange(0, 1, delta)
        beta = (_lcdf(rs) - beta0) / beta_f
        beta5 = (_lcdf(rs + delta) - beta0) / beta_f

        sx1 = x1.x + rs * dx
        sy1 = y1.x + beta * dy
        sx2 = x1.x + (rs + delta) * dx
        sy2 = y1.x + beta5 * dy

        sx1t = x1.x + flow + rs * dxt
        sy1t = y1.xt + beta * dyt
        sx2t = x1.x + flow + (rs + delta) * dxt
        sy2t = y1.xt + beta5 * dyt

        xs = np.c_[sx1, sx2, sx2t, sx1t]
        ys = np.c_[sy1, sy2, sy2t, sy1t]

        start_alpha, end_alpha = 0.2, alpha
        if start_alpha > end_alpha:
            start_alpha, end_alpha = end_alpha, start_alpha
        col = np.c_[
            (start_color * (1 - rs[:, None])) + (end_color * rs[:, None]),
            np.linspace(start_alpha, end_alpha, len(rs)),
        ]

        for x, y, c in zip(xs, ys, col):
            ax.fill(x, y, c=c, edgecolor=None)

    def _plot(
        self,
        old_times: Sequence[Numeric_t],
        ascending: Optional[bool],
        min_flow: float = 0,
        alpha: float = 0.8,
        xticks_step_size: Optional[int] = 1,
        legend_loc: Optional[str] = "upper right out",
        figsize: Optional[Tuple[float, float]] = None,
        dpi: Optional[int] = None,
    ) -> plt.Axes:

        def r(num: float) -> int:
            return max(0, int(round(num, 2) * 100) - 1)

        def draw_edges(
            curr_t: Numeric_t,
            next_t: Numeric_t,
            clusters: Sequence[Any],
            *,
            bottom: bool,
        ):
            smooth_cluster = float(smoothed_proportions[self._cluster][r(curr_t)])
            flow = self._flow.loc[curr_t]
            for clust in clusters:
                fl = flow.loc[self._cluster, clust]
                if fl > min_flow:
                    fl = np.clip(fl, 0, 0.95)
                    smooth_cluster_fl = smoothed_proportions[self._cluster][
                        r(curr_t + fl)
                    ]

                    if bottom:
                        self._draw_flow_edge(
                            ax,
                            x1=Point(curr_t, 0),
                            x2=Point(next_t - fl, next_t - fl - 0.05),
                            y1=Point(
                                cluster_offset - smooth_cluster,
                                cluster_offset - smooth_cluster_fl,
                            ),
                            y2=Point(
                                y_offset[clust]
                                + smoothed_proportions[clust][r(next_t)],
                                y_offset[clust]
                                + smoothed_proportions[clust][r(next_t - fl - 0.05)],
                            ),
                            flow=fl,
                            start_color=self.cmap[self._cluster],
                            end_color=self.cmap[clust],
                            alpha=alpha,
                        )
                    else:
                        self._draw_flow_edge(
                            ax,
                            x1=Point(curr_t + fl, 0),
                            x2=Point(next_t - 0.05, next_t),
                            y1=Point(
                                cluster_offset + smooth_cluster_fl,
                                cluster_offset + smooth_cluster,
                            ),
                            y2=Point(
                                y_offset[clust]
                                - smoothed_proportions[clust][r(next_t - fl - 0.05)],
                                y_offset[clust]
                                - smoothed_proportions[clust][r(next_t)],
                            ),
                            flow=-fl,
                            start_color=self.cmap[self._cluster],
                            end_color=self.cmap[clust],
                            alpha=alpha,
                        )

        if xticks_step_size is not None:
            xticks_step_size = max(1, xticks_step_size)
        times = self._cmat.columns
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        clusters_bottom, clusters_top = self._order_clusters(self._cluster, ascending)
        all_clusters = clusters_bottom + [self._cluster] + clusters_top

        y_offset = self._calculate_y_offsets(all_clusters)
        cluster_offset = y_offset[self._cluster]

        smoothed_proportions, handles = self._plot_smoothed_proportion(
            ax, all_clusters, y_offset, alpha=alpha
        )

        for curr_t, next_t in zip(times[:-1], times[1:]):
            draw_edges(curr_t, next_t, clusters_bottom, bottom=True)
            draw_edges(curr_t, next_t, clusters_top, bottom=False)

        ax.margins(0.025)
        ax.set_title(self._cluster)
        ax.set_xlabel(self._tkey)
        ax.set_ylabel(self._ckey)
        if xticks_step_size is None:
            ax.set_xticks([])
        else:
            ax.set_xticks(times[::xticks_step_size])
            ax.set_xticklabels(old_times[::xticks_step_size])
        ax.set_yticks([])

        if legend_loc not in (None, "none"):
            _position_legend(
                ax,
                legend_loc=legend_loc,
                handles=[handles[c] for c in all_clusters[::-1]],
            )

        return ax

    @property
    def clusters(self) -> pd.Series:
        """Clusters."""
        return self._adata.obs[self._ckey]

    @property
    def time(self) -> pd.Series:
        """Time points."""
        return self._adata.obs[self._tkey]

    @property
    def cmap(self) -> Mapping[str, Any]:
        """Colormap for :attr:`clusters`."""
        return dict(
            zip(
                self.clusters.cat.categories,
                self._adata.uns.get(
                    f"{self._ckey}_colors",
                    _create_categorical_colors(len(self.clusters.cat.categories)),
                ),
            )
        )

def _lcdf(
    x: Union[int, float, np.ndarray], loc: float = 0.5, scale: float = 0.2
) -> float:
    return logistic.cdf(x, loc=loc, scale=scale)

"""# Rand walk"""

# Commented out IPython magic to ensure Python compatibility.
from typing import Any, List, Tuple, Union, Mapping, Optional, Sequence
from typing_extensions import Literal

from pathlib import Path
from itertools import chain

import scvelo as scv
from anndata import AnnData
import numpy as np
from scipy.sparse import issparse, spmatrix
from pandas.api.types import infer_dtype, is_numeric_dtype, is_categorical_dtype

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_hex
from matplotlib.collections import LineCollection

def _get_basis(adata: AnnData, basis: str) -> np.ndarray:
    try:
        return adata.obsm[f"X_{basis}"]
    except KeyError:
        try:
            return adata.obsm[basis]  # e.g. 'spatial'
        except KeyError:
            raise KeyError(
                f"Unable to find a basis in `adata.obsm['X_{basis}']` or `adata.obsm[{basis!r}]`."
            ) from None


__all__ = ["RandomWalk"]

Indices_t = Optional[
    Union[Sequence[str], Mapping[str, Union[str, Sequence[str], Tuple[float, float]]]]
]
@d.dedent
class RandomWalk:
    """
    Class that simulates a random walk on a Markov chain.
    Parameters
    ----------
#     %(adata)s
    transition_matrix
        Row-stochastic transition matrix.
    start_ixs
        Indices from which to uniformly sample the starting points. If `None`, use all points.
    stop_ixs
        Indices which when hit, the random walk is terminated.
    """

    def __init__(
        self,
        adata: AnnData,
        transition_matrix: Union[np.ndarray, spmatrix],
        start_ixs: Optional[Sequence[int]] = None,
        stop_ixs: Optional[Sequence[int]] = None,
    ):
        if transition_matrix.ndim != 2 or (
            transition_matrix.shape[0] != transition_matrix.shape[1]
        ):
            raise ValueError(
                f"Expected transition matrix to be a square matrix, found `{transition_matrix.ndim}`."
            )
        if transition_matrix.shape[0] != adata.n_obs:
            raise ValueError(
                f"Expected transition matrix to be of shape `{adata.n_obs, adata.n_obs}`,"
                f"found `{transition_matrix.shape}`."
            )
        if not np.allclose(transition_matrix.sum(1), 1.0):
            raise ValueError("Transition matrix is not row-stochastic.")

        self._adata = adata
        self._tmat = transition_matrix
        self._ixs = np.arange(self._tmat.shape[0])
        self._is_sparse = issparse(self._tmat)

        start_ixs = self._normalize_ixs(start_ixs, kind="start")
        stop_ixs = self._normalize_ixs(stop_ixs, kind="stop")
        self._stop_ixs = set([] if stop_ixs is None or not len(stop_ixs) else stop_ixs)
        self._starting_dist = (
            np.ones_like(self._ixs)
            if start_ixs is None
            else np.isin(self._ixs, start_ixs)
        )
        _sum = np.sum(self._starting_dist)
        if _sum == 0:
            raise ValueError("No starting indices have been selected.")

        self._starting_dist = self._starting_dist.astype(transition_matrix.dtype) / _sum

    @d.get_sections(base="rw_sim", sections=["Parameters"])
    def simulate_one(
        self,
        max_iter: Union[int, float] = 0.25,
        seed: Optional[int] = None,
        successive_hits: int = 0,
    ) -> np.ndarray:
        """
        Simulate one random walk.
        Parameters
        ----------
        max_iter
            Maximum number of steps of a random walk. If a :class:`float`, it can be specified
            as a fraction of the number of cells.
        seed
            Random seed.
        successive_hits
            Number of successive hits in the ``stop_ixs`` required to stop prematurely.
        Returns
        -------
        Array of shape ``(max_iter + 1,)`` of states that have been visited. If ``stop_ixs`` was specified, the array
        may have smaller shape.
        """
        max_iter = self._max_iter(max_iter)
        if successive_hits < 0:
            raise ValueError(
                f"Expected number of successive hits to be positive, found `{successive_hits}`."
            )

        rs = np.random.RandomState(seed)
        ix = rs.choice(self._ixs, p=self._starting_dist)
        sim, cnt = [ix], -1

        for _ in range(max_iter):
            ix = self._sample(ix, rs=rs)
            sim.append(ix)
            cnt = (cnt + 1) if self._should_stop(ix) else -1
            if cnt >= successive_hits:
                break

        return np.array(sim)

    def _simulate_many(
        self,
        sims: np.ndarray,
        max_iter: Union[int, float] = 0.25,
        seed: Optional[int] = None,
        successive_hits: int = 0,
        queue: Optional[Any] = None,
    ) -> List[np.ndarray]:
        res = []
        for s in sims:
            sim = self.simulate_one(
                max_iter=max_iter,
                seed=None if seed is None else seed + s,
                successive_hits=successive_hits,
            )
            res.append(sim)
            if queue is not None:
                queue.put(1)

        if queue is not None:
            queue.put(None)

        return res

    @d.dedent
    def simulate_many(
        self,
        n_sims: int,
        max_iter: Union[int, float] = 0.25,
        seed: Optional[int] = None,
        successive_hits: int = 0,
        n_jobs: Optional[int] = None,
        backend: str = "loky",
        show_progress_bar: bool = True,
    ) -> List[np.ndarray]:
        """
        Simulate many random walks.
        Parameters
        ----------
        n_sims
            Number of random walks to simulate.
#         %(rw_sim.params)s
#         %(parallel)s
        Returns
        -------
        List of arrays of shape ``(max_iter + 1,)`` of states that have been visited.
        If ``stop_ixs`` was specified, the arrays may have smaller shape.
        """
        if n_sims <= 0:
            raise ValueError(
                f"Expected number of simulations to be positive, found `{n_sims}`."
            )
        max_iter = self._max_iter(max_iter)
        start = info(
            f"Simulating `{n_sims}` random walks of maximum length `{max_iter}`"
        )

        simss = parallelize(
            self._simulate_many,
            collection=np.arange(n_sims),
            n_jobs=n_jobs,
            backend=backend,
            show_progress_bar=show_progress_bar,
            as_array=False,
            unit="sim",
        )(max_iter=max_iter, seed=seed, successive_hits=successive_hits)
        simss = list(chain.from_iterable(simss))

        info("    Finish", time=start)

        return simss

    @d.dedent
    def plot(
        self,
        sims: List[np.ndarray],
        basis: str = "umap",
        cmap: Union[str, LinearSegmentedColormap] = "gnuplot",
        linewidth: float = 1.0,
        linealpha: float = 0.3,
        ixs_legend_loc: Optional[str] = None,
        figsize: Optional[Tuple[float, float]] = None,
        dpi: Optional[int] = None,
        save: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Plot simulated random walks.
        Parameters
        ----------
        sims
            The simulated random walks.
        basis
            Basis used for plotting.
        cmap
            Colormap for the random walks.
        linewidth
            Line width for the random walks.
        linealpha
            Line alpha.
        ixs_legend_loc
            Position of the legend describing start- and endpoints.
#         %(plotting)s
        kwargs
            Keyword arguments for :func:`scvelo.pl.scatter`.
        Returns
        -------
#         %(just_plots)s
        """
        emb = _get_basis(self._adata, basis)
        if isinstance(cmap, str):
            cmap = plt.get_cmap(cmap)
        if not isinstance(cmap, LinearSegmentedColormap):
            if not hasattr(cmap, "colors"):
                raise AttributeError(
                    "Unable to create a colormap, `cmap` does not have attribute `colors`."
                )
            cmap = LinearSegmentedColormap.from_list(
                "random_walk",
                colors=cmap.colors,
                N=max(map(len, sims)),
            )

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        scv.pl.scatter(self._adata, basis=basis, show=False, ax=ax, **kwargs)

        info("Plotting random walks")
        for sim in sims:
            x = emb[sim][:, 0]
            y = emb[sim][:, 1]
            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            n_seg = len(segments)

            lc = LineCollection(
                segments,
                linewidths=linewidth,
                colors=[cmap(float(i) / n_seg) for i in range(n_seg)],
                alpha=linealpha,
                zorder=2,
            )
            ax.add_collection(lc)

        for ix in [0, -1]:
            ixs = [sim[ix] for sim in sims]
            from scvelo.plotting.utils import default_size, plot_outline

            plot_outline(
                x=emb[ixs][:, 0],
                y=emb[ixs][:, 1],
                outline_color=("black", to_hex(cmap(float(abs(ix))))),
                kwargs={
                    "s": kwargs.get("size", default_size(self._adata)) * 1.1,
                    "alpha": 0.9,
                },
                ax=ax,
                zorder=4,
            )

        if ixs_legend_loc not in (None, "none"):
            from cellrank.pl._utils import _position_legend

            h1 = ax.scatter([], [], color=cmap(0.0), label="start")
            h2 = ax.scatter([], [], color=cmap(1.0), label="stop")
            legend = ax.get_legend()
            if legend is not None:
                ax.add_artist(legend)
            _position_legend(ax, legend_loc=ixs_legend_loc, handles=[h1, h2])

        if save is not None:
            save_fig(fig, save)

    def _normalize_ixs(
        self, ixs: Indices_t, *, kind: Literal["start", "stop"]
    ) -> Optional[np.ndarray]:
        if ixs is None:
            return None

        if isinstance(ixs, dict):
            # fmt: off
            if len(ixs) != 1:
                raise ValueError(f"Expected to find only 1 cluster key, found `{len(ixs)}`.")
            key = next(iter(ixs.keys()))
            if key not in self._adata.obs:
                raise KeyError(f"Unable to find data in `adata.obs[{key!r}]`.")

            vals = self._adata.obs[key]
            if is_categorical_dtype(vals):
                ixs = np.where(np.isin(vals, ixs[key]))[0]
            elif is_numeric_dtype(vals):
                if len(ixs[key]) != 2:
                    raise ValueError(f"Expected range to be of length `2`, found `{len(ixs[key])}`")
                minn, maxx = sorted(ixs[key])
                ixs = np.where((vals >= minn) & (vals <= maxx))[0]
            else:
                raise TypeError(f"Expected `adata.obs[{key!r}]` to be numeric or categorical, "
                                f"found `{infer_dtype(vals)}`.")
            # fmt: on
        elif isinstance(ixs, str):
            ixs = np.where(self._adata.obs_names == ixs)[0]
        elif isinstance(ixs[0], str):
            ixs = np.where(np.isin(self._adata.obs_names, ixs))[0]
        elif isinstance(ixs[0], bool):
            if len(ixs) != self._adata.n_obs:
                raise ValueError(
                    f"Expected `bool` {kind} indices of length"
                    f"`{self._adata.n_obs}`, found `{len(ixs)}`."
                )
            ixs = np.where(ixs)[0]
        elif isinstance(ixs[0], int):
            ixs = list(set(ixs))
            if max(ixs) >= self._adata.n_obs:
                raise IndexError(max(ixs))
            if min(ixs) < -self._adata.n_obs:
                raise IndexError(min(ixs))
        else:
            raise TypeError(
                f"Expected {kind} indices to be either `dict` or a sequence of "
                f"`int`, `str`, `bool`, found `{type(ixs).__nam__}`."
            )

        if not len(ixs):
            raise ValueError(f"No {kind} indices have been selected.")

        return ixs

    def _should_stop(self, ix: int) -> bool:
        return ix in self._stop_ixs

    def _sample(self, ix: int, *, rs: np.random.RandomState) -> int:
        return rs.choice(
            self._ixs,
            p=self._tmat[ix].A.squeeze() if self._is_sparse else self._tmat[ix],
        )

    def _max_iter(self, max_iter: Union[int, float]) -> int:
        if isinstance(max_iter, float):
            max_iter = int(np.ceil(max_iter * len(self._ixs)))
        if max_iter <= 1:
            raise ValueError(
                f"Expected number of iterations to be > 1, found `{max_iter}`."
            )
        return max_iter

"""# BidirectionalKernel"""

from typing import Any, Tuple, Union, Optional
from typing_extensions import Protocol

import pickle
from pathlib import Path
from contextlib import contextmanager

from anndata import AnnData

class IOMixinProtocol(Protocol):
    @property
    def shape(self) -> Tuple[int, ...]:
        ...

    @property
    def adata(self) -> AnnData:
        ...

    @adata.setter
    def adata(self, adata: AnnData) -> None:
        ...

class IOMixin:
    """Mixin that allows for serialization from/to files using :mod:`pickle`."""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    @property
    @contextmanager
    def _remove_adata(self) -> None:
        """Temporarily remove :attr:`adata`, if present."""
        adata = getattr(self, "adata", None)

        try:
            if adata is not None:
                self.adata = None
            yield
        finally:
            if adata is not None:
                self.adata = adata

    def write(
        self,
        fname: Union[str, Path],
        write_adata: bool = True,
        ext: Optional[str] = "pickle",
    ) -> None:
        """
        Serialize self to a file.
        Parameters
        ----------
        fname
            Filename where to save the object.
        write_adata
            Whether to save :attr:`adata` object or not, if present.
        ext
            Filename extension to use. If `None`, don't append any extension.
        Returns
        -------
        Nothing, just writes itself to a file using :mod:`pickle`.
        """

        fname = str(fname)
        if ext is not None:
            if not ext.startswith("."):
                ext = "." + ext
            if not fname.endswith(ext):
                fname += ext

        info(f"Writing `{self}` to `{fname}`")

        if write_adata:
            with open(fname, "wb") as fout:
                pickle.dump(self, fout)
            return

        with self._remove_adata:
            with open(fname, "wb") as fout:
                pickle.dump(self, fout)

    @staticmethod
    def read(
        fname: Union[str, Path], adata: Optional[AnnData] = None, copy: bool = False
    ) -> "IOMixin":
        """
        De-serialize self from a file.
        Parameters
        ----------
        fname
            Filename from which to read the object.
        adata
            :class:`anndata.AnnData` object to assign to the saved object.
            Only used when the saved object has :attr:`adata` and it was saved without it.
        copy
            Whether to copy ``adata`` before assigning it or not. If ``adata`` is a view, it is always copied.
        Returns
        -------
        The de-serialized object.
        """

        with open(fname, "rb") as fin:
            obj: IOMixinProtocol = pickle.load(fin)

        if hasattr(obj, "adata"):
            if isinstance(obj.adata, AnnData):
                if adata is not None:
                    pass
                return obj

            if not isinstance(adata, AnnData):
                raise TypeError(
                    "This object was saved without its `adata` object. "
                    "Please supply one as `adata=...`."
                )

            if obj.shape[0] != len(adata):
                raise ValueError(
                    f"Expected `adata` to be of length `{len(adata)}`, found `{obj.shape[0]}`."
                )
            if copy or adata.is_view:
                adata = adata.copy()

            obj.adata = adata
            return obj

        return obj

class BidirectionalMixin(ABC):
    """Mixin specifying that its kernel has forward or backward directions."""

    def __init__(self, *args: Any, backward: bool = False, **kwargs: Any):
        super().__init__(*args, **kwargs)
        if not isinstance(backward, (bool, np.bool_)):
            raise TypeError(
                f"Expected `backward` to be `bool`, found `{type(backward).__name__}`."
            )
        self._backward = bool(backward)
        self._init_kwargs["backward"] = backward

    @abstractmethod
    def __invert__(self) -> "BidirectionalMixin":
        pass

    @property
    def backward(self) -> bool:
        """Direction of the process."""
        return self._backward

Indices_t = Optional[
    Union[Sequence[str], Mapping[str, Union[str, Sequence[str], Tuple[float, float]]]]
]

from typing import Any, Callable, Optional


class cprop:
    """Class property."""

    def __init__(self, f: Callable[..., str]):
        self.f = f

    def __get__(self, obj: Any, owner: Any) -> str:
        return self.f(owner)


class Key:
    """Class which manages keys in :class:`anndata.AnnData`."""

    @classmethod
    def backward(cls, bwd: Optional[bool]) -> str:
        return "bwd" if bwd else "fwd"

    @classmethod
    def where(cls, bwd: Optional[bool]) -> str:
        return "from" if bwd else "to"

    @classmethod
    def initial(cls, bwd: Optional[bool]) -> str:
        return "initial" if bwd else "terminal"

    @classmethod
    def cytotrace(cls, key: str) -> str:
        return f"ct_{key}"

    class obs:
        @classmethod
        def probs(cls, key: str) -> str:
            return f"{key}_probs"

        @classmethod
        def macrostates(cls, bwd: Optional[bool]) -> str:
            return f"macrostates_{Key.backward(bwd)}"

        @classmethod
        def term_states(cls, bwd: Optional[bool]) -> str:
            return f"{Key.initial(bwd)}_states"

        @classmethod
        def priming_degree(cls, bwd: Optional[bool]) -> str:
            return f"priming_degree_{Key.backward(bwd)}"

    class obsm:
        @classmethod
        def memberships(cls, key: str) -> str:
            return f"{key}_memberships"

        @classmethod
        def schur_vectors(cls, bwd: Optional[bool]) -> str:
            return f"schur_vectors_{Key.backward(bwd)}"

        @classmethod
        def macrostates(cls, bwd: Optional[bool]) -> str:
            return f"macrostates_{Key.backward(bwd)}"

        @classmethod
        def abs_probs(cls, bwd: Optional[bool]) -> str:
            return Key.where(bwd) + "_" + Key.obs.term_states(bwd)

        @classmethod
        def abs_times(cls, bwd: Optional[bool]) -> str:
            return f"absorption_times_{Key.backward(bwd)}"

    class varm:
        @classmethod
        def lineage_drivers(cls, bwd: Optional[bool]):
            return Key.initial(bwd) + "_lineage_drivers"

    class uns:
        @classmethod
        def kernel(cls, bwd: Optional[bool], key: Optional[str] = None) -> str:
            return key if key is not None else f"T_{Key.backward(bwd)}"

        @classmethod
        def estimator(cls, bwd: Optional[bool], key: Optional[str] = None) -> str:
            return key if key is not None else f"{Key.backward(bwd)}_estimator"

        @classmethod
        def names(cls, key: str) -> str:
            return f"{key}_names"

        @classmethod
        def colors(cls, key: str) -> str:
            return f"{key}_colors"

        @classmethod
        def eigen(cls, bwd: Optional[bool]) -> str:
            return f"eigendecomposition_{Key.backward(bwd)}"

        @classmethod
        def schur_matrix(cls, bwd: Optional[bool]) -> str:
            return f"schur_matrix_{Key.backward(bwd)}"

        @classmethod
        def coarse(cls, bwd: Optional[bool]) -> str:
            return f"coarse_{Key.backward(bwd)}"

# Commented out IPython magic to ensure Python compatibility.
from typing import Any, Optional

import warnings

from scvelo.tools.velocity_embedding import quiver_autoscale

import numpy as np
from scipy.sparse import issparse, spmatrix, isspmatrix_csr


class TmatProjection:
    """
    Project transition matrix onto a low-dimensional embedding.
    Should be used for visualization purposes.
    Parameters
    ----------
    kexpr
        Kernel that contains a transition matrix.
    basis
        Key in :attr:`anndata.AnnData.obsm` where the basis is stored.
    """

    def __init__(self, kexpr: "ExpernelExpression", basis: str = "umap"):  # noqa: F821

        for kernel in kexpr.kernels:
            if not isinstance(kernel, ConnectivityMixin):
                warning(
                    f"{kernel!r} is not a kNN based kernel. "
                    f"The embedding projection works best for kNN-based kernels"
                )
                break
        self._kexpr = kexpr
        self._basis = basis[2:] if basis.startswith("X_") else basis
        self._key: Optional[str] = None

    def project(
        self,
        key_added: Optional[str] = None,
        recompute: bool = False,
        connectivities: Optional[spmatrix] = None,
    ) -> None:
        """
        Project transition matrix onto an embedding.
        This function has been adapted from :func:`scvelo.tl.velocity_embedding`.
        Parameters
        ----------
        key_added
            Key in :attr:`anndata.AnnData.obsm` where to store the projection.
        recompute
            Whether to recompute the projection if it already exists.
        connectivities
            Connectivity matrix to use for projection. If ``None``, use ones from the underlying kernel, is possible.
        Returns
        -------
        Nothing, just updates :attr:`anndata.AnnData` with the projection and the parameters used for computation.
        """

        self._key = Key.uns.kernel(self._kexpr.backward, key=key_added)
        ukey = f"{self._key}_params"
        key = f"{self._key}_{self._basis}"

        if not recompute and key in self._kexpr.adata.obsm:
            info(f"Using precomputed projection `adata.obsm[{key!r}]`")
            return

        start = info(f"Projecting transition matrix onto `{self._basis}`")
        if connectivities is None:
            try:
                connectivities, *_ = (
                    c.connectivities
                    for c in self._kexpr.kernels
                    if isinstance(c, ConnectivityMixin)
                )
            except ValueError:
                raise RuntimeError(
                    "Unable to find connectivities in the kernel. "
                    "Please supply them explicitly as `connectivities=...`."
                ) from None
        if not isspmatrix_csr(connectivities):
            connectivities = connectivities.tocsr()

        emb = _get_basis(self._kexpr.adata, self._basis)
        T_emb = np.empty_like(emb)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for row_id, row in enumerate(self._kexpr.transition_matrix):
                conn_idxs = connectivities[row_id, :].indices
                dX = emb[conn_idxs] - emb[row_id, None]

                if np.any(np.isnan(dX)):
                    T_emb[row_id, :] = np.nan
                else:
                    probs = (
                        row[:, conn_idxs].A.squeeze()
                        if issparse(row)
                        else row[conn_idxs]
                    )
                    dX /= np.linalg.norm(dX, axis=1)[:, None]
                    dX = np.nan_to_num(dX)
                    T_emb[row_id, :] = probs.dot(dX) - dX.sum(0) / dX.shape[0]

        T_emb /= 3 * quiver_autoscale(np.nan_to_num(emb), T_emb)

        embs = self._kexpr.adata.uns.get(ukey, {}).get("embeddings", [])
        if self._basis not in embs:
            embs = list(embs) + [self._basis]
            self._kexpr.adata.uns[ukey] = self._kexpr.adata.uns.get(ukey, {})
            self._kexpr.adata.uns[ukey]["embeddings"] = embs

        info(
            f"Adding `adata.obsm[{key!r}]`\n    Finish",
            time=start,
        )
        self._kexpr.adata.obsm[key] = T_emb

    @d.dedent
    def plot(self, *args: Any, stream: bool = True, **kwargs: Any) -> None:
        """
        Plot projected transition matrix in a embedding.
        Parameters
        ----------
        args
            Positional argument for the plotting function.
        stream
            If ``True``, use :func:`scvelo.pl.velocity_embedding_stream`.
            Otherwise, use :func:`scvelo.pl.velocity_embedding_grid`.
        kwargs
            Keyword argument for the chosen plotting function.
        Returns
        -------
#         %(just_plots)s
        """
        if stream:
            return scv.pl.velocity_embedding_stream(
                self._kexpr.adata, *args, basis=self._basis, vkey=self._key, **kwargs
            )
        return scv.pl.velocity_embedding_grid(
            self._kexpr.adata, *args, basis=self._basis, vkey=self._key, **kwargs
        )

def _normalize(
    X: Union[np.ndarray, spmatrix],
) -> Union[np.ndarray, spmatrix]:
    """
    Row-normalizes an array to sum to 1.
    Parameters
    ----------
    X
        Array to be normalized.
    Returns
    -------
    :class:`numpy.ndarray` or :class:`scipy.sparse.spmatrix`
        The normalized array.
    """

    with np.errstate(divide="ignore"):
        if issparse(X):
            return X.multiply(csr_matrix(1.0 / np.abs(X).sum(1)))
        X = np.array(X)
        return X / (X.sum(1)[:, None])

from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import os
from matplotlib.colors import LinearSegmentedColormap, to_hex

def require_tmat(
    wrapped: Callable[..., Any],
    instance: "KernelExpression",  # noqa: F821
    args: Any,
    kwargs: Any,
) -> Any:
    """Require that the transition matrix is computed before calling the wrapped function."""
    # this can trigger combinations, but not individual kernels
    if instance.transition_matrix is None:
        raise RuntimeError(
            "Compute transition matrix first as `.compute_transition_matrix()`."
        )
    return wrapped(*args, **kwargs)

def _maybe_create_dir(dirpath: Union[str, os.PathLike]) -> None:
    """
    Try creating a directory if it does not already exist.
    Parameters
    ----------
    dirpath
        Path of the directory to create.
    Returns
    -------
    None
        Nothing, just creates a directory if it doesn't exist.
    """

    if not os.path.exists(dirpath) or not os.path.isdir(dirpath):
        try:
            os.makedirs(dirpath, exist_ok=True)
        except OSError:
            pass

def save_fig(
    fig, path: Union[str, os.PathLike], make_dir: bool = True, ext: str = "png"
) -> None:
    """
    Save a plot.
    Parameters
    ----------
    fig: :class:`matplotlib.figure.Figure`
        Figure to save.
    path:
        Path where to save the figure. If path is relative, save it under ``cellrank.settings.figdir``.
    make_dir:
        Whether to try making the directory if it does not exist.
    ext:
        Extension to use.
    Returns
    -------
    None
        Just saves the plot.
    """


    if os.path.splitext(path)[1] == "":
        path = f"{path}.{ext}"

    if not os.path.isabs(path):
        path = os.path.join(settings.figdir, path)

    if make_dir:
        _maybe_create_dir(os.path.split(path)[0])

    debug(f"Saving figure to `{path!r}`")

    fig.savefig(path, bbox_inches="tight", transparent=True)

# Commented out IPython magic to ensure Python compatibility.
Tmat_t = Union[np.ndarray, spmatrix]
import wrapt
@wrapt.decorator

def require_tmat(
    wrapped: Callable[..., Any],
    instance: "KernelExpression",  # noqa: F821
    args: Any,
    kwargs: Any,
) -> Any:
    """Require that the transition matrix is computed before calling the wrapped function."""
    # this can trigger combinations, but not individual kernels
    if instance.transition_matrix is None:
        raise RuntimeError(
            "Compute transition matrix first as `.compute_transition_matrix()`."
        )
    return wrapped(*args, **kwargs)




class KernelExpression(IOMixin, ABC):
    def __init__(
        self,
        parent: Optional["KernelExpression"] = None,
        **kwargs: Any,
    ):
        super().__init__()
        self._parent = parent
        self._normalize = parent is None
        self._transition_matrix = None
        self._params: Dict[str, Any] = {}
        self._init_kwargs = kwargs  # for `_read_from_adata`

    def __init_subclass__(cls, **_: Any) -> None:
        super().__init_subclass__()

    @abstractmethod
    def compute_transition_matrix(
        self, *args: Any, **kwargs: Any
    ) -> "KernelExpression":
        """
        Compute transition matrix.
        Parameters
        ----------
        args
            Positional arguments.
        kwargs
            Keyword arguments.
        Returns
        -------
        Modifies :attr:`transition_matrix` and returns self.
        """

    @abstractmethod
    def copy(self, *, deep: bool = False) -> "KernelExpression":
        """Return a copy of itself. The underlying :attr:`adata` object is not copied."""

    @property
    @abstractmethod
    def adata(self) -> AnnData:
        """Annotated data object."""

    @adata.setter
    @abstractmethod
    def adata(self, value: Optional[AnnData]) -> None:
        pass

    @property
    @abstractmethod
    def kernels(self) -> Tuple["KernelExpression", ...]:
        """Underlying base kernels."""

    @property
    @abstractmethod
    def shape(self) -> Tuple[int, int]:
        """``(n_cells, n_cells)``."""

    @property
    @abstractmethod
    def backward(self) -> Optional[bool]:
        """Direction of the process."""

    @abstractmethod
    def __getitem__(self, ix: int) -> "KernelExpression":
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @d.get_full_description(base="plot_single_flow")
    @d.get_sections(base="plot_single_flow", sections=["Parameters", "Returns"])
    @d.dedent
    @require_tmat
    def plot_single_flow(
        self,
        cluster: str,
        cluster_key: str,
        time_key: str,
        clusters: Optional[Sequence[Any]] = None,
        time_points: Optional[Sequence[Union[int, float]]] = None,
        min_flow: float = 0,
        remove_empty_clusters: bool = True,
        ascending: Optional[bool] = False,
        legend_loc: Optional[str] = "upper right out",
        alpha: Optional[float] = 0.8,
        xticks_step_size: Optional[int] = 1,
        figsize: Optional[Tuple[float, float]] = None,
        dpi: Optional[int] = None,
        save: Optional[Union[str, Path]] = None,
        show: bool = True,
    ) -> Optional[plt.Axes]:
        """
        Visualize outgoing flow from a cluster of cells :cite:`mittnenzweig:21`.
        Parameters
        ----------
        cluster
            Cluster for which to visualize outgoing flow.
        cluster_key
            Key in :attr:`anndata.AnnData.obs` where clustering is stored.
        time_key
            Key in :attr:`anndata.AnnData.obs` where experimental time is stored.
        clusters
            Visualize flow only for these clusters. If `None`, use all clusters.
        time_points
            Visualize flow only for these time points. If `None`, use all time points.
#         %(flow.parameters)s
#         %(plotting)s
        show
            If `False`, return :class:`matplotlib.pyplot.Axes`.
        Returns
        -------
        The axes object, if ``show = False``.
#         %(just_plots)s
        Notes
        -----
        This function is a Python re-implementation of the following
        `original R function <https://github.com/tanaylab/embflow/blob/main/scripts/generate_paper_figures/plot_vein.r>`_
        with some minor stylistic differences.
        This function will not recreate the results from :cite:`mittnenzweig:21`, because there, the *Metacell* model
        :cite:`baran:19` was used to compute the flow, whereas here the transition matrix is used.
        """  # noqa: E501
        fp = FlowPlotter(self.adata, self.transition_matrix, cluster_key, time_key)
        fp = fp.prepare(cluster, clusters, time_points)

        ax = fp.plot(
            min_flow=min_flow,
            remove_empty_clusters=remove_empty_clusters,
            ascending=ascending,
            alpha=alpha,
            xticks_step_size=xticks_step_size,
            legend_loc=legend_loc,
            figsize=figsize,
            dpi=dpi,
        )

        if save is not None:
            save_fig(ax.figure, save)

        if not show:
            return ax

    @d.dedent
    @require_tmat
    def plot_random_walks(
        self,
        n_sims: int = 100,
        max_iter: Union[int, float] = 0.25,
        seed: Optional[int] = None,
        successive_hits: int = 0,
        start_ixs: Indices_t = None,
        stop_ixs: Indices_t = None,
        basis: str = "umap",
        cmap: Union[str, LinearSegmentedColormap] = "gnuplot",
        linewidth: float = 1.0,
        linealpha: float = 0.3,
        ixs_legend_loc: Optional[str] = None,
        n_jobs: Optional[int] = None,
        backend: str = "loky",
        show_progress_bar: bool = True,
        figsize: Optional[Tuple[float, float]] = None,
        dpi: Optional[int] = None,
        save: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Plot random walks in an embedding.
        This method simulates random walks on the Markov chain defined though the corresponding transition matrix. The
        method is intended to give qualitative rather than quantitative insights into the transition matrix. Random
        walks are simulated by iteratively choosing the next cell based on the current cell's transition probabilities.
        Parameters
        ----------
        n_sims
            Number of random walks to simulate.
#         %(rw_sim.parameters)s
        start_ixs
            Cells from which to sample the starting points. If `None`, use all cells.
#             %(rw_ixs)s
            For example ``{'dpt_pseudotime': [0, 0.1]}`` means that starting points for random walks
            will be sampled uniformly from cells whose pseudotime is in `[0, 0.1]`.
        stop_ixs
            Cells which when hit, the random walk is terminated. If `None`, terminate after ``max_iters``.
#             %(rw_ixs)s
            For example ``{'clusters': ['Alpha', 'Beta']}`` and ``successive_hits = 3`` means that the random walk will
            stop prematurely after cells in the above specified clusters have been visited successively 3 times in a
            row.
        basis
            Basis in :attr:`anndata.AnnData.obsm` to use as an embedding.
        cmap
            Colormap for the random walk lines.
        linewidth
            Width of the random walk lines.
        linealpha
            Alpha value of the random walk lines.
        ixs_legend_loc
            Legend location for the start/top indices.
#         %(parallel)s
#         %(plotting)s
        kwargs
            Keyword arguments for :func:`scvelo.pl.scatter`.
        Returns
        -------
#         %(just_plots)s
        For each random walk, the first/last cell is marked by the start/end colors of ``cmap``.
        """
        rw = RandomWalk(
            self.adata, self.transition_matrix, start_ixs=start_ixs, stop_ixs=stop_ixs
        )
        sims = rw.simulate_many(
            n_sims=n_sims,
            max_iter=max_iter,
            seed=seed,
            n_jobs=n_jobs,
            backend=backend,
            successive_hits=successive_hits,
            show_progress_bar=show_progress_bar,
        )

        rw.plot(
            sims,
            basis=basis,
            cmap=cmap,
            linewidth=linewidth,
            linealpha=linealpha,
            ixs_legend_loc=ixs_legend_loc,
            figsize=figsize,
            dpi=dpi,
            save=save,
            **kwargs,
        )

    @require_tmat
    def plot_projection(
        self,
        basis: str = "umap",
        key_added: Optional[str] = None,
        recompute: bool = False,
        stream: bool = True,
        connectivities: Optional[spmatrix] = None,
        **kwargs: Any,
    ) -> None:
        """
        Plot :attr:`transition_matrix` as a stream or a grid plot.
        Parameters
        ----------
        basis
            Key in :attr:`anndata.AnnData.obsm` containing the basis.
        key_added
            If not `None`, save the result to :attr:`anndata.AnnData.obsm` ``['{key_added}']``.
            Otherwise, save the result to `'T_fwd_{basis}'` or `T_bwd_{basis}`, depending on the direction.
        recompute
            Whether to recompute the projection if it already exists.
        stream
            If ``True``, use :func:`scvelo.pl.velocity_embedding_stream`.
            Otherwise, use :func:`scvelo.pl.velocity_embedding_grid`.
        connectivities
            Connectivity matrix to use for projection. If ``None``, use ones from the underlying kernel, is possible.
        kwargs
            Keyword argument for the chosen plotting function.
        Returns
        -------
        Nothing, just plots and modifies :attr:`anndata.AnnData.obsm` with a key based on ``key_added``.
        """
        proj = TmatProjection(self, basis=basis)
        proj.project(
            key_added=key_added, recompute=recompute, connectivities=connectivities
        )
        proj.plot(stream=stream, **kwargs)

    def __add__(self, other: "KernelExpression") -> "KernelExpression":
        return self.__radd__(other)

    def __radd__(self, other: "KernelExpression") -> "KernelExpression":
        def same_level_add(k1: "KernelExpression", k2: "KernelExpression") -> bool:
            if not (isinstance(k1, KernelAdd) and isinstance(k2, KernelMul)):
                return False

            for kexpr in k1:
                if not isinstance(kexpr, KernelMul):
                    return False
                if not kexpr._bin_consts:
                    return False
            return True

        if not isinstance(other, KernelExpression):
            return NotImplemented

        s = self * 1.0 if isinstance(self, Kernel) else self
        o = other * 1.0 if isinstance(other, Kernel) else other

        # (c1 * x + c2 * y + ...) + (c3 * z) => (c1 * x + c2 * y + c3 + ... + * z)
        if same_level_add(s, o):
            return KernelAdd(*tuple(s) + (o,))
        if same_level_add(o, s):
            return KernelAdd(*tuple(o) + (s,))

        # add virtual constant
        if not isinstance(s, KernelMul):
            s = s * 1.0
        if not isinstance(o, KernelMul):
            o = o * 1.0

        return KernelAdd(s, o)

    def __mul__(
        self, other: Union[float, int, "KernelExpression"]
    ) -> "KernelExpression":
        return self.__rmul__(other)

    def __rmul__(
        self, other: Union[int, float, "KernelExpression"]
    ) -> "KernelExpression":
        def same_level_mul(k1: "KernelExpression", k2: "KernelExpression") -> bool:
            return (
                isinstance(k1, KernelMul)
                and isinstance(k2, Constant)
                and k1._bin_consts
            )

        if isinstance(other, (int, float, np.integer, np.floating)):
            other = Constant(self.adata, other)

        if not isinstance(other, KernelExpression):
            return NotImplemented

        # fmt: off
        s = self if isinstance(self, (KernelMul, Constant)) else KernelMul(Constant(self.adata, 1.0), self)
        o = other if isinstance(other, (KernelMul, Constant)) else KernelMul(Constant(other.adata, 1.0), other)
        # fmt: on

        # at this point, only KernelMul and Constant is possible (two constants are not possible)
        # (c1 * k) * c2 => (c1 * c2) * k
        if same_level_mul(s, o):
            c, expr = s._split_const
            return KernelMul(c * o, expr)
        if same_level_mul(o, s):
            c, expr = o._split_const
            return KernelMul(c * s, expr)

        return KernelMul(s, o)

    @d.get_sections(base="write_to_adata", sections=["Parameters"])
    @inject_docs()  # gets rid of {{}} in %(write_to_adata)s
    @d.dedent
    @require_tmat
    def write_to_adata(self, key: Optional[str] = None, copy: bool = False) -> None:
        """
        Write the transition matrix and parameters used for computation to the underlying :attr:`adata` object.
        Parameters
        ----------
        key
            Key used when writing transition matrix to :attr:`adata`.
            If `None`, the key will be determined automatically.
        Returns
        -------
#         %(write_to_adata)s
        """

        if self.adata is None:
            raise ValueError("Underlying annotated data object is not set.")

        key = Key.uns.kernel(self.backward, key=key)
        # retain the embedding info
        self.adata.uns[f"{key}_params"] = {
            **self.adata.uns.get(f"{key}_params", {}),
            **{"params": self.params},
            **{"init": self._init_kwargs},
        }
        self.adata.obsp[key] = (
            self.transition_matrix.copy() if copy else self.transition_matrix
        )

    @property
    def transition_matrix(self) -> Union[np.ndarray, csr_matrix]:
        """Row-normalized transition matrix."""
        if self._parent is None and self._transition_matrix is None:
            self.compute_transition_matrix()
        return self._transition_matrix

    @transition_matrix.setter
    def transition_matrix(self, matrix: Union[np.ndarray, spmatrix]) -> None:
        """
        Set the transition matrix.
        Parameters
        ----------
        matrix
            Transition matrix. The matrix is row-normalized if necessary.
        Returns
        -------
        Nothing, just updates the :attr:`transition_matrix` and optionally normalizes it.
        """
        # fmt: off
        if matrix.shape != self.shape:
            raise ValueError(
                f"Expected matrix to be of shape `{self.shape}`, found `{matrix.shape}`."
            )

        def should_norm(mat: Union[np.ndarray, spmatrix]) -> bool:
            return not np.all(np.isclose(np.asarray(mat.sum(1)).squeeze(), 1.0, rtol=1e-12))

        if issparse(matrix) and not isspmatrix_csr(matrix):
            matrix = csr_matrix(matrix)
        matrix = matrix.astype(np.float64, copy=False)

        force_normalize = (self._parent is None or self._normalize) and should_norm(matrix)
        if force_normalize:
            if np.any((matrix.data if issparse(matrix) else matrix) < 0):
                raise ValueError("Unable to normalize matrix with negative values.")
            matrix = _normalize(matrix)
            if should_norm(matrix):  # some rows are all 0s/contain invalid values
                n_inv = np.sum(~np.isclose(np.asarray(matrix.sum(1)).squeeze(), 1.0, rtol=1e-12))
                raise ValueError(f"Transition matrix is not row stochastic, {n_inv} rows do not sum to 1.")
        # fmt: on

        self._transition_matrix = matrix

    @property
    def params(self) -> Dict[str, Any]:
        """Parameters which are used to compute the transition matrix."""
        if len(self.kernels) == 1:
            return self._params
        return {f"{k!r}:{i}": k.params for i, k in enumerate(self.kernels)}

    def _reuse_cache(
        self, expected_params: Dict[str, Any], *, time: Optional[Any] = None
    ) -> bool:
        # fmt: off
        try:
            if expected_params == self._params:
                assert self.transition_matrix is not None
                debug("Using cached transition matrix")
                info("    Finish", time=time)
                return True
            return False
        except AssertionError:
            warning("Transition matrix does not exist for the given parameters. Recomputing")
            return False
        except Exception as e:  # noqa: B902
            # e.g. the dict is not comparable
            warning(f"Expected and actually parameters are not comparable, reason `{e}`. Recomputing")
            expected_params = {}  # clear the params
            return False
        finally:
            self._params = expected_params
        # fmt: on

from copy import deepcopy

class Kernel(KernelExpression, ABC):
    """Base kernel class."""

    def __init__(
        self, adata: AnnData, parent: Optional[KernelExpression] = None, **kwargs: Any
    ):
        super().__init__(parent=parent, **kwargs)
        self._adata = adata
        self._n_obs = adata.n_obs
        self._read_from_adata(**kwargs)

    def _read_from_adata(self, **kwargs: Any) -> None:
        pass

    @property
    def adata(self) -> AnnData:
        """Annotated data object."""
        return self._adata

    @adata.setter
    def adata(self, adata: Optional[AnnData]) -> None:
        if adata is None:
            self._adata = None
            return
        if not isinstance(adata, AnnData):
            raise TypeError(
                f"Expected `adata` to be of type `AnnData`, found `{type(adata).__name__}`."
            )
        shape = (adata.n_obs, adata.n_obs)
        if self.shape != shape:
            raise ValueError(
                f"Expected new `AnnData` object to have same shape as the previous `{self.shape}`, found `{shape}`."
            )
        self._adata = adata

    @classmethod
    @d.dedent
    def from_adata(
        cls,
        adata: AnnData,
        key: str,
        copy: bool = False,
    ) -> "Kernel":
        """
        Read kernel object saved using :meth:`write_to_adata`.
        Parameters
        ---------
#         %(adata)s
        key
            Key in :attr:`anndata.AnnData.obsp` where the transition matrix is stored.
            The parameters should be stored in :attr:`anndata.AnnData.uns` ``['{key}_params']``.
        copy
            Whether to copy the transition matrix.
        Returns
        -------
        The kernel with explicitly initialized properties:
            - :attr:`transition_matrix` - the transition matrix.
            - :attr:`params` - parameters used for computation.
        """
        transition_matrix = _read_graph_data(adata, key=key)
        try:
            params = adata.uns[f"{key}_params"]["params"].copy()
            init_params = adata.uns[f"{key}_params"]["init"].copy()
        except KeyError as e:
            raise KeyError(f"Unable to kernel parameters, reason: `{e}`") from e

        if copy:
            transition_matrix = transition_matrix.copy()

        kernel = cls(adata, **init_params)
        kernel.transition_matrix = transition_matrix
        kernel._params = params

        return kernel

    def copy(self, *, deep: bool = False) -> "Kernel":
        """Return a copy of self."""
        with self._remove_adata:
            k = deepcopy(self)
        k.adata = self.adata.copy() if deep else self.adata
        return k

    def _copy_ignore(self, *attrs: str) -> "Kernel":
        # prevent copying attributes that are not necessary, e.g. during inversion
        sentinel, attrs = object(), set(attrs)
        objects = [
            (attr, obj)
            for attr, obj in ((attr, getattr(self, attr, sentinel)) for attr in attrs)
            if obj is not sentinel
        ]
        try:
            for attr, _ in objects:
                setattr(self, attr, None)
            return self.copy(deep=False)
        finally:
            for attr, obj in objects:
                setattr(self, attr, obj)

    def _format_params(self) -> str:
        n, _ = self.shape
        params = ", ".join(
            f"{k}={round(v, 3) if isinstance(v, float) else v!r}"
            for k, v in self.params.items()
        )
        return f"n={n}, {params}" if params else f"n={n}"

    @property
    def transition_matrix(self) -> Union[np.ndarray, csr_matrix]:
        """Row-normalized transition matrix."""
        return self._transition_matrix

    @transition_matrix.setter
    def transition_matrix(self, matrix: Any) -> None:
        KernelExpression.transition_matrix.fset(self, matrix)

    @property
    def kernels(self) -> Tuple["KernelExpression", ...]:
        """Underlying base kernels."""
        return (self,)

    @property
    def shape(self) -> Tuple[int, int]:
        """``(n_cells, n_cells)``."""
        return self._n_obs, self._n_obs

    def __getitem__(self, ix: int) -> "Kernel":
        if ix != 0:
            raise IndexError(ix)
        return self

    def __len__(self) -> int:
        return 1

    def __repr__(self) -> str:
        params = self._format_params()
        prefix = "~" if self.backward and self._parent is None else ""
        return f"{prefix}{self.__class__.__name__}[{params}]"

    def __str__(self) -> str:
        prefix = "~" if self.backward and self._parent is None else ""
        return f"{prefix}{self.__class__.__name__}[n={self.shape[0]}]"
        

class NaryKernelExpression(BidirectionalMixin, KernelExpression):
    def __init__(
        self, *kexprs: KernelExpression, parent: Optional[KernelExpression] = None
    ):
        super().__init__(parent=parent)
        self._validate(kexprs)

    @abstractmethod
    def _combine_transition_matrices(self, t1: Tmat_t, t2: Tmat_t) -> Tmat_t:
        pass

    @property
    @abstractmethod
    def _initial_value(self) -> Union[int, float, Tmat_t]:
        pass

    @property
    @abstractmethod
    def _combiner(self) -> str:
        pass

    def compute_transition_matrix(self) -> "NaryKernelExpression":
        for kexpr in self:
            if kexpr.transition_matrix is None:
                if isinstance(kexpr, Kernel):
                    raise RuntimeError(
                        f"`{kexpr}` is uninitialized. Compute its "
                        f"transition matrix first as `.compute_transition_matrix()`."
                    )
                kexpr.compute_transition_matrix()

        tmat = self._initial_value
        for kexpr in self:
            tmat = self._combine_transition_matrices(tmat, kexpr.transition_matrix)
        self.transition_matrix = tmat

        return self

    def _validate(self, kexprs: Sequence[KernelExpression]) -> None:
        if not len(kexprs):
            raise ValueError("No kernels to combined.")

        shapes = {kexpr.shape for kexpr in kexprs}
        if len(shapes) > 1:
            raise ValueError(
                f"Expected all kernels to have the same shapes, found `{sorted(shapes)}`."
            )

        directions = {kexpr.backward for kexpr in kexprs}
        if True in directions and False in directions:
            raise ValueError("Unable to combine both forward and backward kernels.")

        self._backward = (
            True if True in directions else False if False in directions else None
        )
        self._kexprs = kexprs
        for kexpr in self:
            kexpr._parent = self

    @property
    def adata(self) -> AnnData:
        """Annotated data object."""
        return self[0].adata

    @adata.setter
    def adata(self, adata: Optional[AnnData]) -> None:
        # allow resetting (use for temp. pickling without adata)
        for kexpr in self:
            kexpr.adata = adata

    @property
    def kernels(self) -> Tuple["KernelExpression", ...]:
        """Underlying unique basic kernels."""
        kernels = []
        for kexpr in self:
            if isinstance(kexpr, Kernel) and not isinstance(kexpr, Constant):
                kernels.append(kexpr)
            elif isinstance(kexpr, NaryKernelExpression):  # recurse
                kernels.extend(kexpr.kernels)

        # return only unique kernels
        return tuple(set(kernels))

    def copy(self, *, deep: bool = False) -> "KernelExpression":
        kexprs = (k.copy(deep=deep) for k in self)
        return type(self)(*kexprs, parent=self._parent)

    @property
    def shape(self) -> Tuple[int, int]:
        """``(n_cells, n_cells)``."""
        # all kernels have the same shape
        return self[0].shape

    def _format(self, formatter: Callable[[KernelExpression], str]) -> str:
        return (
            f"{'~' if self.backward and self._parent is None else ''}("
            + f" {self._combiner} ".join(formatter(kexpr) for kexpr in self)
            + ")"
        )

    def __getitem__(self, ix: int) -> "KernelExpression":
        return self._kexprs[ix]

    def __len__(self) -> int:
        return len(self._kexprs)

    def __invert__(self) -> "KernelExpression":
        if self.backward is None:
            return self.copy()
        kexprs = tuple(
            ~k if isinstance(k, BidirectionalMixin) else k.copy() for k in self
        )
        kexpr = type(self)(*kexprs, parent=self._parent)
        kexpr._transition_matrix = None
        return kexpr

    def __repr__(self) -> str:
        return self._format(repr)

    def __str__(self) -> str:
        return self._format(str)

class UnidirectionalKernel(UnidirectionalMixin, Kernel, ABC):
    """Kernel with no directionality."""

class Constant(UnidirectionalKernel):
    def __init__(self, adata: AnnData, value: Union[int, float]):
        super().__init__(adata)
        self.transition_matrix = value

    def compute_transition_matrix(self, value: Union[int, float]) -> "Constant":
        self.transition_matrix = value
        return self

    @property
    def transition_matrix(self) -> Union[int, float]:
        return self._transition_matrix

    @transition_matrix.setter
    def transition_matrix(self, value: Union[int, float]) -> None:
        if not isinstance(value, (int, float, np.integer, np.floating)):
            raise TypeError(
                f"Value must be a `float` or `int`, found `{type(value).__name__}`."
            )
        if value <= 0:
            raise ValueError(f"Expected the scalar to be positive, found `{value}`.")

        self._transition_matrix = value
        self._params = {"value": value}

    def copy(self, *, deep: bool = False) -> "Constant":
        return Constant(self.adata, self.transition_matrix)

    # fmt: off
    def __radd__(self, other: Union[int, float, "KernelExpression"]) -> "Constant":
        if isinstance(other, (int, float, np.integer, np.floating)):
            other = Constant(self.adata, other)
        if isinstance(other, Constant):
            if self.shape != other.shape:
                raise ValueError(f"Expected kernel shape to be `{self.shape}`, found `{other.shape}`.")
            return Constant(self.adata, self.transition_matrix + other.transition_matrix)

        return super().__radd__(other)

    def __rmul__(self, other: Union[int, float, "KernelExpression"]) -> "Constant":
        if isinstance(other, (int, float, np.integer, np.floating)):
            other = Constant(self.adata, other)
        if isinstance(other, Constant):
            if self.shape != other.shape:
                raise ValueError(f"Expected kernel shape to be `{self.shape}`, found `{other.shape}`.")
            return Constant(self.adata, self.transition_matrix * other.transition_matrix)

        return super().__rmul__(other)
    # fmt: on

    def __repr__(self) -> str:
        return repr(round(self.transition_matrix, 3))

    def __str__(self) -> str:
        return str(round(self.transition_matrix, 3))



class KernelAdd(NaryKernelExpression):
    def compute_transition_matrix(self) -> "KernelAdd":
        self._maybe_recalculate_constants()
        return super().compute_transition_matrix()

    def _combine_transition_matrices(self, t1: Tmat_t, t2: Tmat_t) -> Tmat_t:
        return t1 + t2

    def _maybe_recalculate_constants(self) -> None:
        """Normalize constants to sum to 1."""
        constants = self._bin_consts
        if constants:
            total = sum((c.transition_matrix for c in constants), 0.0)
            for c in constants:
                c.transition_matrix = c.transition_matrix / total
            for kexpr in self:  # don't normalize  (c * x)
                kexpr._normalize = False

    @property
    def _bin_consts(self) -> List[KernelExpression]:
        """Return constant expressions for each binary multiplication children with at least 1 constant."""
        return [c for k in self if isinstance(k, KernelMul) for c in k._bin_consts]

    @property
    def _combiner(self) -> str:
        return "+"

    @property
    def _initial_value(self) -> Union[int, float, Tmat_t]:
        return 0.0


class KernelMul(NaryKernelExpression):
    def _combine_transition_matrices(self, t1: Tmat_t, t2: Tmat_t) -> Tmat_t:
        if issparse(t1):
            return t1.multiply(t2)
        if issparse(t2):
            return t2.multiply(t1)
        return t1 * t2

    def _format(self, formatter: Callable[[KernelExpression], str]) -> str:
        fmt = super()._format(formatter)
        if fmt[0] == "~" or not self._bin_consts:
            return fmt
        if fmt.startswith("("):
            fmt = fmt[1:]
        if fmt.endswith(")"):
            fmt = fmt[:-1]
        return fmt

    @property
    def _bin_consts(self) -> List[KernelExpression]:
        """Return all constants if this expression contains only 2 subexpressions."""
        if len(self) != 2:
            return []

        return [k for k in self if isinstance(k, Constant)]

    @property
    def _split_const(self) -> Tuple[Optional[Constant], Optional[KernelExpression]]:
        """Return a constant and the other expression, iff this expression is of length 2 and contains a constant."""
        if not self._bin_consts:
            return None, None
        k1, k2 = self
        if isinstance(k1, Constant):
            return k1, k2
        return k2, k1

    @property
    def _combiner(self) -> str:
        return "*"

    @property
    def _initial_value(self) -> Union[int, float, Tmat_t]:
        return 1.0

class BidirectionalKernel(BidirectionalMixin, Kernel, ABC):
    """Kernel with either forward or backward direction that can be inverted using :meth:`__invert__`."""

# Commented out IPython magic to ensure Python compatibility.
from typing import Any, Union, Callable, Optional
from typing_extensions import Literal

from enum import auto

from anndata import AnnData

import numpy as np



class ThresholdScheme(ModeEnum):
    SOFT = auto()
    HARD = auto()


@d.dedent
class PseudotimeKernel(ConnectivityMixin, BidirectionalKernel):
    """
    Kernel which computes directed transition probabilities based on a KNN graph and pseudotime.
    The kNN graph contains information about the (undirected) connectivities among cells, reflecting their similarity.
    Pseudotime can be used to either remove edges that point against the direction of increasing pseudotime
    :cite:`setty:19` or to down-weight them :cite:`stassen:21`.
    Parameters
    ----------
#     %(adata)s
#     %(backward)s
    time_key
        Key in :attr:`anndata.AnnData.obs` where the pseudotime is stored.
    kwargs
        Keyword arguments for the parent class.
    """

    def __init__(
        self,
        adata: AnnData,
        time_key: str,
        backward: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            adata,
            backward=backward,
            time_key=time_key,
            **kwargs,
        )

    def _read_from_adata(self, time_key: str, **kwargs: Any) -> None:
        super()._read_from_adata(**kwargs)
        # fmt: off
        self._time_key = time_key
        if time_key not in self.adata.obs:
            raise KeyError(f"Unable to find pseudotime in `adata.obs[{time_key!r}]`.")

        self._pseudotime = np.array(self.adata.obs[time_key]).astype(np.float64, copy=True)
        if np.any(np.isnan(self.pseudotime)):
            raise ValueError("Encountered NaN values in pseudotime.")
        # fmt: on
    
    def compute_projection(
        self,
        basis: str = "umap",
        key_added: Optional[str] = None,
        copy: bool = False,
    ) -> Optional[np.ndarray]:
        """
        Compute a projection of the transition matrix in the embedding.

        Projections can only be calculated for kNN based kernels. The projected matrix
        can be then visualized as::

            scvelo.pl.velocity_embedding(adata, vkey='T_fwd', basis='umap')

        Parameters
        ----------
        basis
            Basis in :attr:`anndata.AnnData.obsm` for which to compute the projection.
        key_added
            If not `None` and ``copy = False``, save the result to :attr:`anndata.AnnData.obsm` ``['{key_added}']``.
            Otherwise, save the result to `'T_fwd_{basis}'` or `T_bwd_{basis}`, depending on the direction.
        copy
            Whether to return the projection or modify :attr:`adata` inplace.

        Returns
        -------
        If ``copy=True``, the projection array of shape `(n_cells, n_components)`.
        Otherwise, it modifies :attr:`anndata.AnnData.obsm` with a key based on ``key_added``.
        """
        # modified from: https://github.com/theislab/scvelo/blob/master/scvelo/tools/velocity_embedding.py
        from scvelo.tools.velocity_embedding import quiver_autoscale

        if self._transition_matrix is None:
            raise RuntimeError(
                "Compute transition matrix first as `.compute_transition_matrix()`."
            )

        for kernel in self.kernels:
            if kernel._conn is None:
                raise AttributeError(
                    f"{kernel!r} is not a kNN based kernel. The embedding projection "
                    "only works for kNN based kernels."
                )

        start = info(f"Projecting transition matrix onto `{basis}`")
        emb = _get_basis(self.adata, basis)
        T_emb = np.empty_like(emb)

        conn = self.kernels[0]._conn

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for row_id, row in enumerate(self.transition_matrix):
                conn_idxs = conn[row_id, :].indices

                dX = emb[conn_idxs] - emb[row_id, None]

                if np.any(np.isnan(dX)):
                    T_emb[row_id, :] = np.nan
                else:
                    probs = row[:, conn_idxs]
                    if issparse(probs):
                        probs = probs.A.squeeze()

                    dX /= np.linalg.norm(dX, axis=1)[:, None]
                    dX = np.nan_to_num(dX)
                    T_emb[row_id, :] = probs.dot(dX) - dX.sum(0) / dX.shape[0]

        T_emb /= 3 * quiver_autoscale(np.nan_to_num(emb), T_emb)

        if copy:
            return T_emb

        key = Key.uns.kernel(self.backward, key=key_added)
        ukey = f"{key}_params"

        embs = self.adata.uns.get(ukey, {}).get("embeddings", [])
        if basis not in embs:
            embs = list(embs) + [basis]
            self.adata.uns[ukey] = self.adata.uns.get(ukey, {})
            self.adata.uns[ukey]["embeddings"] = embs

        key = key + "_" + basis
        info(
            f"Adding `adata.obsm[{key!r}]`\n    Finish",
            time=start,
        )
        self.adata.obsm[key] = T_emb

    @d.dedent
    def compute_transition_matrix(
        self,
        threshold_scheme: Union[
            Literal["soft", "hard"],
            Callable[[float, np.ndarray, np.ndarray], np.ndarray],
        ] = "hard",
        frac_to_keep: float = 0.3,
        b: float = 10.0,
        nu: float = 0.5,
        check_irreducibility: bool = False,
        n_jobs: Optional[int] = None,
        backend: Backend_t = _DEFAULT_BACKEND,
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> "PseudotimeKernel":
        """
        Compute transition matrix based on kNN graph and pseudotemporal ordering.
        Depending on the choice of the ``threshold_scheme``, it is based on ideas by either *Palantir*
        :cite:`setty:19` or *VIA* :cite:`stassen:21`.
        Parameters
        ----------
        threshold_scheme
            Which method to use when biasing the graph. Valid options are:
                - `'hard'` - based on *Palantir* :cite:`setty:19` which removes some edges that point against
                  the direction of increasing pseudotime. To avoid disconnecting the graph, it does not
                  remove all edges that point against the direction of increasing pseudotime, but keeps the ones
                  that point to cells inside a close radius. This radius is chosen according to the local cell density.
                - `'soft'` - based on *VIA* :cite:`stassen:21` which down-weights edges that points against
                  the direction of increasing pseudotime. Essentially, the further "behind"
                  a query cell is in pseudotime with respect
                  to the current reference cell, the more penalized will be its graph-connectivity.
                - :class:`callable` - any function conforming to the signature of
                  :func:`cellrank.kernels.utils.ThresholdSchemeABC.__call__`.
        frac_to_keep
            Fraction of the closest neighbors (according to graph connectivities) are kept, no matter whether they lie
            in the pseudotemporal past or future. This is done to ensure that the graph remains connected.
            Only used when ``threshold_scheme = 'hard'``. Needs to fall within the interval `[0, 1]`.
#         %(soft_scheme_kernel)s
        check_irreducibility
            Optional check for irreducibility of the final transition matrix.
#         %(parallel)s
        kwargs
            Keyword arguments for ``threshold_scheme``.
        Returns
        -------
        Self and updates :attr:`transition_matrix` and :attr:`params`.
        """
        if self.pseudotime is None:
            raise ValueError("Compute pseudotime first.")  # CytoTraceKernel

        start = info("Computing transition matrix based on pseudotime`")
        if isinstance(threshold_scheme, str):
            threshold_scheme = ThresholdScheme(threshold_scheme)
            if threshold_scheme == ThresholdScheme.SOFT:
                scheme = SoftThresholdScheme()
                kwargs["b"] = b
                kwargs["nu"] = nu
            elif threshold_scheme == ThresholdScheme.HARD:
                scheme = HardThresholdScheme()
                kwargs["frac_to_keep"] = frac_to_keep
            else:
                raise NotImplementedError(
                    f"Threshold scheme `{threshold_scheme}` is not yet implemented."
                )
        elif isinstance(threshold_scheme, ThresholdSchemeABC):
            scheme = threshold_scheme
        elif callable(threshold_scheme):
            scheme = CustomThresholdScheme(threshold_scheme)
        else:
            raise TypeError(
                f"Expected `threshold_scheme` to be either a `str` or a `callable`, found `{type(threshold_scheme)}`."
            )

        # fmt: off
        if self._reuse_cache({"dnorm": False, "scheme": str(threshold_scheme), **kwargs}, time=start):
            return self
        # fmt: on

        biased_conn = scheme.bias_knn(
            self.connectivities,
            self.pseudotime,
            n_jobs=n_jobs,
            backend=backend,
            show_progress_bar=show_progress_bar,
            **kwargs,
        )

        # make sure the biased graph is still connected
        if not _connected(biased_conn):
            warning("Biased kNN graph is disconnected")
        if check_irreducibility and not _irreducible(biased_conn):
            warning("Biased kNN graph is not irreducible")

        self.transition_matrix = biased_conn
        info("    Finish", time=start)

        return self

    @property
    def pseudotime(self) -> Optional[np.array]:
        """Pseudotemporal ordering of cells."""
        return self._pseudotime

    def __invert__(self) -> "PseudotimeKernel":
        pk = self._copy_ignore("_transition_matrix")
        pk._pseudotime = np.max(pk.pseudotime) - pk.pseudotime
        pk._backward = not self.backward
        pk._params = {}
        return pk