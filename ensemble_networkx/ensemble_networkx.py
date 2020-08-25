# -*- coding: utf-8 -*-
from __future__ import print_function, division

# ==============================================================================
# Modules
# ==============================================================================
# Built-ins
import os, sys, copy, warnings
from collections import defaultdict, OrderedDict
from itertools import combinations

# PyData
import pandas as pd
import numpy as np
import networkx as nx
import xarray as xr
from scipy import stats
from scipy.special import comb
from scipy.spatial.distance import squareform

# Hive NetworkX
from hive_networkx import Symmetric, signed

# Compositional
from compositional import pairwise_rho, pairwise_phi

# soothsayer_utils
from soothsayer_utils import pv, assert_acceptable_arguments, is_symmetrical, is_graph, write_object, format_memory, format_header, format_path, is_nonstring_iterable, Suppress
try:
    from . import __version__
except ImportError:
    __version__ = "ImportError: attempted relative import with no known parent package"

# ==============================================================================
# Associations
# ==============================================================================
# Biweight midcorrelation
def pairwise_biweight_midcorrelation(X, use_numba=False):
    """
    X: {np.array, pd.DataFrame}

    Code adapted from the following sources:
        * https://stackoverflow.com/questions/61090539/how-can-i-use-broadcasting-with-numpy-to-speed-up-this-correlation-calculation/61219867#61219867
        * https://github.com/olgabot/pandas/blob/e8caf4c09e1a505eb3c88b475bc44d9389956585/pandas/core/nanops.py

    Special thanks to the following people:
        * @norok2 (https://stackoverflow.com/users/5218354/norok2) for optimization (vectorization and numba)
        * @olgabot (https://github.com/olgabot) for NumPy implementation

    Benchmarking:
        * iris_features (4,4)
            * numba: 159 ms ± 2.85 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
            * numpy: 276 µs ± 3.45 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
        * iris_samples: (150,150)
            * numba: 150 ms ± 7.57 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
            * numpy: 686 µs ± 18.8 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

    Future:
        * Handle missing values

    """
    # Data
    result = None
    labels = None
    if isinstance(X, pd.DataFrame):
        labels = X.columns
        X = X.values

    def _base_computation(A):
        n, m = A.shape
        A = A - np.median(A, axis=0, keepdims=True)
        v = 1 - (A / (9 * np.median(np.abs(A), axis=0, keepdims=True))) ** 2
        est = A * v ** 2 * (v > 0)
        norms = np.sqrt(np.sum(est ** 2, axis=0))
        return n, m, est, norms

    # Check if numba is available
    assert_acceptable_arguments(use_numba, {True, False, "infer"})
    if use_numba == "infer":
        if "numba" in sys.modules:
            use_numba = True
        else:
            use_numba = False
        print("Numba is available:", use_numba, file=sys.stderr)

    # Compute using numba
    if use_numba:
        assert "numba" in sys.modules
        from numba import jit

        def _biweight_midcorrelation_numba(A):
            @jit
            def _condensed_to_dense(n, m, est, norms, result):
                for i in range(m):
                    for j in range(i + 1, m):
                        x = 0
                        for k in range(n):
                            x += est[k, i] * est[k, j]
                        result[i, j] = result[j, i] = x / norms[i] / norms[j]
            n, m, est, norms = _base_computation(A)
            result = np.empty((m, m))
            np.fill_diagonal(result, 1.0)
            _condensed_to_dense(n, m, est, norms, result)
            return result

        result = _biweight_midcorrelation_numba(X)
    # Compute using numpy
    else:
        def _biweight_midcorrelation_numpy(A):
            n, m, est, norms = _base_computation(A)
            return np.einsum('mi,mj->ij', est, est) / norms[:, None] / norms[None, :]
        result = _biweight_midcorrelation_numpy(X)

    # Add labels
    if labels is not None:
        result = pd.DataFrame(result, index=labels, columns=labels)

    return result

# =============================
# Ensemble Association Networks
# =============================
class EnsembleAssociationNetwork(object):
    """
    # Load in data
    import soothsayer_utils as syu
    X = syu.get_iris_data(["X"])

    # Create ensemble network
    ens = enx.EnsembleAssociationNetwork(name="Iris", node_type="leaf measurement", edge_type="association", observation_type="specimen")
    ens.fit(X=X, metric="spearman",  n_iter=100, stats_summary=[np.mean,np.var, stats.kurtosis, stats.skew], stats_tests=[stats.normaltest], copy_ensemble=True)
    print(ens)
    # =======================================================
    # EnsembleAssociationNetwork(Name:Iris, Metric: spearman)
    # =======================================================
    #     * Number of nodes (leaf measurement): 4
    #     * Number of edges (association): 6
    #     * Observation type: specimen
    #     ---------------------------------------------------
    #     | Parameters
    #     ---------------------------------------------------
    #     * n_iter: 100
    #     * sampling_size: 92
    #     * random_state: 0
    #     * with_replacement: False
    #     * transformation: None
    #     * memory: 16.156 KB
    #     ---------------------------------------------------
    #     | Data
    #     ---------------------------------------------------
    #     * Features (n=150, m=4, memory=10.859 KB)
    #     * Ensemble (memory=4.812 KB)
    #     * Statistics (['mean', 'var', 'kurtosis', 'skew', 'normaltest|stat', 'normaltest|p_value'], memory=496 B)

    # View ensemble
    print(ens.ensemble_.head())
    # Edges       (sepal_width, sepal_length)  (sepal_length, petal_length)  \
    # Iterations                                                              
    # 0                             -0.113835                      0.880407   
    # 1                             -0.243982                      0.883397   
    # 2                             -0.108511                      0.868627   
    # 3                             -0.151437                      0.879405   
    # 4                             -0.241807                      0.869027  

    # View statistics
    print(ens.stats_.head())
    # Statistics                        mean       var  kurtosis      skew  \
    # Edges                                                                  
    # (sepal_width, sepal_length)  -0.167746  0.002831  0.191176  0.287166   
    # (sepal_length, petal_length)  0.880692  0.000268 -0.107437  0.235619   
    # (petal_width, sepal_length)   0.834140  0.000442 -0.275487 -0.219778   
    # (sepal_width, petal_length)  -0.304403  0.003472 -0.363377  0.059179   
    # (sepal_width, petal_width)   -0.285237  0.003466 -0.606118  0.264103 

    __future__: 
        * Add ability to load in previous data.  However, this is tricky because one needs to validate that the following objects are the same: 
            - X
            - sampling_size
            - n_iter
            - random_state
            - with_replacement
            etc.
    """
    def __init__(
        self, 
        name=None,
        node_type=None,
        edge_type=None,
        observation_type=None,
        assert_symmetry=True,
        assert_draw_size=True,
        assert_nan_safe_functions=True,
        nans_ok=True,
        tol=1e-10,
#         temporary_directory=None,
#         remove_temporary_directory=True,
#         compression="gzip",
#         absolute_path=False,
#         force_overwrite=False,
        ):
        self.name = name
        self.node_type = node_type
        self.edge_type = edge_type
        self.observation_type = observation_type
        self.assert_symmetry = assert_symmetry
        self.assert_draw_size = assert_draw_size
        self.assert_nan_safe_functions = assert_nan_safe_functions
        self.nans_ok = nans_ok
        self.tol = tol
#         if temporary_directory == False:
#             temporary_directory = None
#         if  temporary_directory:
#             # Do tsomething where you can resume from a previous tmp
#             if temporary_directory == True:
#                 temporary_directory = ".EnsembleAssociationNetwork__{}".format(get_unique_identifier())                
#             temporary_directory = format_path(temporary_directory, absolute=absolute_path)
#             os.makedirs(temporary_directory, exist_ok=True)
#         self.temporary_directory = temporary_directory
#         self.remove_temporary_directory = remove_temporary_directory
#         assert_acceptable_arguments(compression, {"gzip", "bz2", None})
#         self.compression = compression
#         self.force_overwrite = force_overwrite

            
        
    def _pandas_association(self, X, metric):
        return X.corr(method=metric)
    
    def fit(
        self,
        X:pd.DataFrame,
        metric="rho",
        n_iter=1000,
        sampling_size=0.6180339887,
        transformation=None,
        random_state=0,
        with_replacement=False,
        function_is_pairwise=True,
        stats_summary=[np.mean, np.median, np.var, stats.kurtosis, stats.skew] ,
        stats_tests=[stats.normaltest],
        copy_X=True,
        copy_ensemble=True,
        a=np.asarray(np.linspace(-1,1,999).tolist() + [np.nan]),

        ):
        
        # Metric
        assert metric is not None
        metric_name = None
        if hasattr(metric, "__call__"):
            if not function_is_pairwise:
                function = metric
                metric_name = function.__name__
                metric = lambda X: self._pandas_association(X=X, metric=function)

        acceptable_metrics = {"rho", "phi", "biweight_midcorrelation", "spearman", "pearson", "kendall"}
        if isinstance(metric, str):
            assert_acceptable_arguments(metric, acceptable_metrics)
            metric_name = metric
            if metric == "rho":
                metric = pairwise_rho
            if metric == "phi":
                metric = pairwise_phi
            if metric == "biweight_midcorrelation":
                metric = pairwise_biweight_midcorrelation
            if metric in {"spearman", "pearson", "kendall"}:
                association = metric
                metric = lambda X: self._pandas_association(X=X, metric=association)

                
        assert hasattr(metric, "__call__"), "`metric` must be either one of the following: [{}], \
        a custom metric that returns an association (set `function_is_pairwise=False`), or a custom \
        metric that returns a 2D square/symmetric pd.DataFrame (set `function_is_pairwise=True`)".format(acceptable_metrics)
        # Transformations
        acceptable_transformations = {"signed", "abs"}
        if transformation:
            if isinstance(transformation, str):
                assert_acceptable_arguments(transformation, acceptable_transformations)
                if transformation == "signed":
                    transformation = signed
                if transformation == "abs":
                    transformation = np.abs
            assert hasattr(transformation, "__call__"), "`transformation` must be either one of the following: [{}] or a function(pd.DataFrame) -> pd.DataFrame".format(acceptable_transformations)
            
       # Check statistics functions
        if self.assert_nan_safe_functions:
            if self.nans_ok:
                number_of_nan = np.isnan(X.values).ravel().sum()
                if number_of_nan > 0:
                    if stats_summary:
                        for func in stats_summary:
                            v = func(a)
                            assert np.isfinite(v), "`stats_summary` function `{}` is cannot handle `nan` ({} missing values)".format(func.__name__, number_of_nan)
                    if stats_tests:
                        for func in stats_tests:
                            v = func(a)[-1]
                            assert np.isfinite(v), "`stats_tests` function `{}` is cannot handle `nan` ({} missing values)".format(func.__name__, number_of_nan)
        # Data
        n, m = X.shape

        # Network
        nodes = pd.Index(X.columns)
        number_of_nodes = len(nodes)
        edges = pd.Index(map(frozenset, combinations(nodes, r=2)), name="Edges")
        number_of_edges = len(edges)


        # Get draws
        draws = list()

        # Use custom draws
        if is_nonstring_iterable(n_iter):
            draw_sizes = list()
            available_observations = set(X.index)
            for draw in n_iter:
                # Check that there are no unique observations in the draw not present in X.index
                query = set(draw) - available_observations
                assert len(query) == 0, "The following observations are not available in `X.index`:\n{}".format(query)
                draws.append(list(draw))
                # Get draw size
                draw_sizes.append(len(draw))
            unique_draw_sizes = set(draw_sizes)
            number_unique_draw_sizes = len(unique_draw_sizes)
            if self.assert_draw_size:
                assert number_unique_draw_sizes == 1, "With `assert_draw_size=True` all draw sizes must be the same length"
                
            # Update
            if number_unique_draw_sizes == 1:
                sampling_size = list(unique_draw_sizes)[0]
            else:
                sampling_size = draw_sizes

            n_iter = len(draws)
            random_state = np.nan
            with_replacement = np.nan

        # Do not use custom draws (this is default)
        else:
            assert 0 < sampling_size < n
            if 0 < sampling_size < 1:
                sampling_size = int(sampling_size*n)

            # Iterations
            number_of_unique_draws_possible = comb(n, sampling_size, exact=True, repetition=with_replacement)
            assert n_iter <= number_of_unique_draws_possible, "`n_iter` exceeds the number of possible draws (total_possible={})".format(number_of_unique_draws_possible)

            if random_state is not None:
                assert isinstance(random_state, int), "`random_state` must either be `None` or of `int` type"
            for j in range(n_iter):
                # Get draw of samples
                if random_state is None:
                    rs = None
                else:
                    rs = j + random_state
                index = np.random.RandomState(rs).choice(X.index, size=sampling_size, replace=with_replacement) 
                draws.append(index.tolist())
                
        # Stats
        if (stats_tests is None) or (stats_tests is False):
            stats_tests = []
        if hasattr(stats_tests, "__call__"):
            stats_tests = [stats_tests]
        stats_tests = list(stats_tests)
        if (stats_summary is None) or (stats_summary is False):
            stats_summary = []
        if hasattr(stats_summary, "__call__"):
            stats_summary = [stats_summary]
        stats_summary = list(stats_summary)
            
        for func in (stats_tests + stats_summary):
            assert hasattr(func, "__name__")
        

        # Associations
        ensemble = np.empty((n_iter, number_of_edges))
        ensemble[:] = np.nan
        for i, index in pv(enumerate(draws), description="Computing associations ({})".format(self.name), total=n_iter, unit=" draws"):
            # Compute associations with current draw
            df_associations = metric(X.loc[index])

            if self.assert_symmetry:
                assert is_symmetrical(df_associations, tol=self.tol)
            weights = squareform(df_associations.values, checks=False) #dense_to_condensed(X=df_associations, assert_symmetry=self.assert_symmetry, tol=self.tol)
            ensemble[i] = weights
        
        ensemble = pd.DataFrame(ensemble, columns=edges)
        ensemble.columns.name = "Edges"
        ensemble.index.name = "Iterations" #"n_iter={}".format(n_iter)

        if transformation is not None:
            ensemble = transformation(ensemble)

        # Parameters
        self.memory_ = 0
        if copy_X: 
            self.X_ = X.copy()
            self.X_memory_ = X.memory_usage().sum()
            self.memory_ += self.X_memory_ 
        self.n_iter = n_iter
        self.sampling_size_ = sampling_size
        self.transformation_ = transformation
        self.function_is_pairwise = function_is_pairwise
        self.random_state = random_state
        self.metric_ = metric
        self.metric_name = metric_name
        self.with_replacement = with_replacement
        # Network
        self.n_ = n
        self.m_ = m
        self.nodes_ = nodes
        self.number_of_nodes_ = number_of_nodes
        self.edges_ = edges
        self.number_of_edges_ = number_of_edges
        self.draws_ = draws # self.draws_ = OrderedDict(zip(range(self.random_state, self.random_state + self.n_iter), draws))

        if copy_ensemble:
            self.ensemble_ = ensemble
            self.ensemble_memory_ = ensemble.memory_usage().sum()
            self.memory_ += self.ensemble_memory_ 
            
        # Statistics
        number_of_statistic_fields = 0
        if stats_summary is not None:
            number_of_statistic_fields += len(stats_summary)
        if stats_tests is not None:
            number_of_statistic_fields += 2*len(stats_tests)

        self.stats_ = np.empty((number_of_edges, number_of_statistic_fields)) #defaultdict(dict) # ensemble.describe(percentiles=percentiles).to_dict()
        self.stats_[:] = np.nan 

        k = 0
        values = ensemble.values
        stat_fields = list()
        if stats_summary:
            for func in pv(stats_summary, description="Computing summary statistics ({})".format(self.name), total=len(stats_summary), unit=" stats"):  
                stat_name = func.__name__
                # self.stats_[stat_name] = func(u, axis=0).to_dict()
                self.stats_[:,k] = func(values, axis=0)
                stat_fields.append(stat_name)
                k += 1
        if stats_tests:
            for func in pv(stats_tests, description="Computing statistical tests ({})".format(self.name), total=len(stats_tests), unit=" tests"):
                stat_name = func.__name__
                stat_fields.append("{}|stat".format(stat_name))
                stat_fields.append("{}|p_value".format(stat_name))
                for j in range(number_of_edges):
                    v = values[:,j]
                    stat, p = func(v)
                    self.stats_[j,[k, k+1]] = [stat,p]
                    # self.stats_[:,k+1] = p
                k += 2
                    
        self.stats_ = pd.DataFrame(self.stats_, index=edges, columns=stat_fields)
        self.stats_.index.name = "Edges"
        self.stats_.columns.name = "Statistics"
        
        self.stats_memory_ = self.stats_.memory_usage().sum()
        self.memory_ += self.stats_memory_ 
        return self
    
    # I/O
    # ===
    def to_file(self, path, compression='infer', **kwargs):
        write_object(self, path=path, compression=compression, **kwargs)
        
    # Convert
    # =======
    def to_condensed(self, weight="mean", into=Symmetric):
        if not hasattr(self, "stats_"):
            raise Exception("Please fit model")
        assert weight in self.stats_
        assert into in {Symmetric, pd.Series}
        sym_network = Symmetric(
            data=self.stats_[weight], 
            name=self.name, 
            node_type=self.node_type, 
            edge_type=self.edge_type, 
            func_metric=self.metric_, 
            association="network", 
            assert_symmetry=self.assert_symmetry, 
            nans_ok=self.nans_ok, tol=self.tol,
        )
        if into == Symmetric:
            return sym_network
        if into == pd.Series:
            return sym_network.weights
        
    def to_dense(self, weight="mean", fill_diagonal=1):
        df_dense = self.to_condensed(weight=weight).to_dense(index=self.nodes_)
        if fill_diagonal is not None:
            np.fill_diagonal(df_dense.values, fill_diagonal)
        return df_dense

    def to_networkx(self, into=None, **attrs):
        if into is None:
            into = nx.Graph
        if not hasattr(self, "stats_"):
            raise Exception("Please fit model")

        metadata = { "node_type":self.node_type, "edge_type":self.edge_type, "observation_type":self.observation_type, "metric":self.metric_}
        metadata.update(attrs)
        graph = into(name=self.name, **metadata)
        for (node_A, node_B), statistics in pv(self.stats_.iterrows(), description="Building NetworkX graph from statistics", total=self.number_of_edges_, unit=" edges"):
            graph.add_edge(node_A, node_B, **statistics)
        return graph 

    def copy(self):
        return copy.deepcopy(self)
    
    # Built-in
    # ========
    def __repr__(self):
        pad = 4
        fitted = hasattr(self, "stats_")
        if fitted:
            header = format_header("{}(Name:{}, Metric: {})".format(type(self).__name__, self.name, self.metric_name),line_character="=")
            n = len(header.split("\n")[0])
            fields = [
                header,
                pad*" " + "* Number of nodes ({}): {}".format(self.node_type, self.number_of_nodes_),
                pad*" " + "* Number of edges ({}): {}".format(self.edge_type, self.number_of_edges_),
                pad*" " + "* Observation type: {}".format(self.observation_type),
                *map(lambda line:pad*" " + line, format_header("| Parameters", "-", n=n-pad).split("\n")),
                pad*" " + "* n_iter: {}".format(self.n_iter),
                pad*" " + "* sampling_size: {}".format(self.sampling_size_),
                pad*" " + "* random_state: {}".format(self.random_state),
                pad*" " + "* with_replacement: {}".format(self.with_replacement),
                pad*" " + "* transformation: {}".format(self.transformation_),
                pad*" " + "* memory: {}".format(format_memory(self.memory_)),
                *map(lambda line:pad*" " + line, format_header("| Data", "-", n=n-pad).split("\n")),

                ]
            if hasattr(self, "X_"):
                fields.append(pad*" " + "* Features (n={}, m={}, memory={})".format(self.n_, self.m_, format_memory(self.X_memory_))),
            else:
                fields.append(pad*" " + "* Features (n={}, m={})".format(self.n_, self.m_)),
            if hasattr(self, "ensemble_"):
                fields.append(pad*" " + "* Ensemble (memory={})".format( format_memory(self.ensemble_memory_))),
            fields.append(pad*" " + "* Statistics ({}, memory={})".format(self.stats_.columns.tolist(), format_memory(self.stats_memory_))),
            return "\n".join(fields)
        else:
            header = format_header("{}(Name:{})".format(type(self).__name__, self.name),line_character="=")
            n = len(header.split("\n")[0])
            fields = [
                header,
                pad*" " + "* Number of nodes ({}): {}".format(self.node_type, 0),
                pad*" " + "* Number of edges ({}): {}".format(self.edge_type, 0),
                pad*" " + "* Observation type: {}".format(self.observation_type),
                ]
            return "\n".join(fields)



# Sample-specific Perturbation Networks
class SampleSpecificPerturbationNetwork(object):
    """
    # Load in data
    import soothsayer_utils as syu
    X, y, colors = syu.get_iris_data(["X","y", "colors"])
    reference = "setosa"
    
    # Create ensemble network
    sspn_rho = enx.SampleSpecificPerturbationNetwork(name="Iris", node_type="leaf measurement", edge_type="association", observation_type="specimen")
    sspn_rho.fit(X=X, y=y, metric="rho", reference="setosa", n_iter=100, stats_summary=[np.mean,np.var], copy_ensemble=True)

    print(sspn_rho)
    # ============================================================================
    # SampleSpecificPerturbationNetwork(Name:Iris, Reference: setosa, Metric: rho)
    # ============================================================================
    #     * Number of nodes (leaf measurement): 4
    #     * Number of edges (association): 6
    #     * Observation type: specimen
    #     ------------------------------------------------------------------------
    #     | Parameters
    #     ------------------------------------------------------------------------
    #     * n_iter: 100
    #     * sampling_size: 30
    #     * random_state: 0
    #     * with_replacement: False
    #     * transformation: None
    #     * memory: 518.875 KB
    #     ------------------------------------------------------------------------
    #     | Data
    #     ------------------------------------------------------------------------
    #     * Features (n=150, m=4, memory=10.859 KB)
    #     ------------------------------------------------------------------------
    #     | Intermediate
    #     ------------------------------------------------------------------------
    #     * Reference Ensemble (memory=208 B)
    #     * Sample-specific Ensembles (memory=20.312 KB)
    #     ------------------------------------------------------------------------
    #     | Terminal
    #     ------------------------------------------------------------------------
    #     * Ensemble (memory=468.750 KB)
    #     * Statistics (['mean', 'var', 'normaltest|stat', 'normaltest|p_value'], memory=18.750 KB)
    # Coordinates:
    #   * Samples     (Samples) object 'iris_50' 'iris_51' ... 'iris_148' 'iris_149'
    #   * Iterations  (Iterations) int64 0 1 2 3 4 5 6 7 8 ... 92 93 94 95 96 97 98 99
    #   * Edges       (Edges) object frozenset({'sepal_width', 'sepal_length'}) ... frozenset({'petal_width', 'petal_length'})
    # Coordinates:
    #   * Samples     (Samples) object 'iris_50' 'iris_51' ... 'iris_148' 'iris_149'
    #   * Edges       (Edges) object frozenset({'sepal_width', 'sepal_length'}) ... frozenset({'petal_width', 'petal_length'})
    #   * Statistics  (Statistics) <U18 'mean' 'var' ... 'normaltest|p_value'

    # View ensemble
    print(*repr(sspn_rho.ensemble_).split("\n")[-4:], sep="\n")
    # Coordinates:
    #   * Samples     (Samples) object 'iris_50' 'iris_51' ... 'iris_148' 'iris_149'
    #   * Iterations  (Iterations) int64 0 1 2 3 4 5 6 7 8 ... 92 93 94 95 96 97 98 99
    #   * Edges       (Edges) object frozenset({'sepal_width', 'sepal_length'}) ... frozenset({'petal_width', 'petal_length'})

    # View statistics
    print(*repr(sspn_rho.stats_).split("\n")[-4:], sep="\n")
    # Coordinates:
    #   * Samples     (Samples) object 'iris_50' 'iris_51' ... 'iris_148' 'iris_149'
    #   * Edges       (Edges) object frozenset({'sepal_width', 'sepal_length'}) ... frozenset({'petal_width', 'petal_length'})
    #   * Statistics  (Statistics) <U18 'mean' 'var' ... 'normaltest|p_value'

    # View SSPN for a particular sample
    graph = sspn_rho.to_networkx("iris_50")
    list(graph.edges(data=True))[0]
    # ('sepal_width',
    #  'sepal_length',
    #  {'mean': 0.04004398613232575,
    #   'var': 0.0011399047127054046,
    #   'normaltest|stat': 6.063925790957182,
    #   'normaltest|p_value': 0.04822089259665142})
    """
    def __init__(
        self, 
        name=None,
        node_type=None,
        edge_type=None,
        observation_type=None,
        reference_type=None,
        assert_symmetry=True,
        assert_nan_safe_functions=True,
        nans_ok=True,
        tol=1e-10,
        ):
        self.name = name
        self.node_type = node_type
        self.edge_type = edge_type
        self.observation_type = observation_type
        self.reference_type = reference_type
        self.assert_symmetry = assert_symmetry
        self.assert_nan_safe_functions = assert_nan_safe_functions
        self.nans_ok = nans_ok
        self.tol = tol
        
    def fit(
        self,
        X:pd.DataFrame,
        y:pd.Series,
        reference,
        metric="rho",
        n_iter=1000,
        sampling_size=0.6180339887,
        transformation=None,
        random_state=0,
        with_replacement=False,
        function_is_pairwise=True,
        stats_summary=[np.mean, np.var, stats.kurtosis, stats.skew], # Need to adjust for NaN robust
        stats_tests=[stats.normaltest],
        stats_summary_initial=None, 
        stats_tests_initial=None,
        copy_X=True,
        copy_y=True,
        copy_ensemble=False, # This will get very big very quickly
        copy_ensemble_reference=False,
        copy_ensemble_samplespecific=False,  # This will get very big very quickly
        # a=np.asarray(np.linspace(-1,1,999).tolist() + [np.nan]),
        ):
        # Assert there is some summary stats
        assert stats_summary is not None
        assert stats_summary is not False 
        if hasattr(stats_summary, "__call__"):
            stats_summary = [stats_summary]
        if hasattr(stats_tests, "__call__"):
            stats_tests = [stats_tests]

        # Memory
        self.memory_ = 0
        
        assert reference in y.unique(), "`reference({}, type:{}) not in `y`".format(reference, type(reference))
        assert set(X.index) == set(y.index), "`X.index` must have same keys as `y.index`"
        y = y[X.index]
 
        # Ensemble Reference 
        index_reference = sorted(y[lambda i: i == reference].index)
        ensemble_reference = EnsembleAssociationNetwork(
                name=reference, 
                node_type=self.node_type, 
                edge_type=self.edge_type, 
                observation_type=self.observation_type, 
                assert_symmetry=self.assert_symmetry, 
                assert_nan_safe_functions=self.assert_nan_safe_functions,
                nans_ok=self.nans_ok,
                tol=self.tol,
        )
        # Fit
        ensemble_reference.fit(
                X=X.loc[index_reference],
                metric=metric,
                n_iter=n_iter,
                sampling_size=sampling_size,
                transformation=transformation,
                random_state=random_state,
                with_replacement=with_replacement,
                function_is_pairwise=function_is_pairwise,
                stats_summary=stats_summary_initial,
                stats_tests=stats_tests_initial,
                copy_X=False,
                copy_ensemble=True,
        )

        # Weights
        values_reference = ensemble_reference.ensemble_.values
        if not copy_ensemble_reference:
            ensemble_reference.memory_ -= ensemble_reference.ensemble_memory_
            delattr(ensemble_reference, "ensemble_")

        # Data
        n, m = X.shape

        # Network
        nodes = ensemble_reference.nodes_
        number_of_nodes = ensemble_reference.number_of_nodes_
        edges = ensemble_reference.edges_
        number_of_edges = ensemble_reference.number_of_edges_
        draws_reference = ensemble_reference.draws_
        n_iter = ensemble_reference.n_iter

        # Query samples
        index_samplespecific = y[lambda i: i != reference].index #sorted(set(X.index) - set(index_reference))
         
        if copy_ensemble:
            self.ensemble_ = np.empty((len(index_samplespecific),  n_iter, number_of_edges))
            self.ensemble_[:] = np.nan
            
        # Statistics
        number_of_statistic_fields = 0

        stat_fields = list()
        if stats_summary:# is not None:
            for func in stats_summary:  
                stat_name = func.__name__
                stat_fields.append(stat_name)
        if stats_tests:# is not None:
            for func in stats_tests:
                stat_name = func.__name__
                stat_fields.append("{}|stat".format(stat_name))
                stat_fields.append("{}|p_value".format(stat_name))

        self.stats_ = np.empty((len(index_samplespecific), number_of_edges, len(stat_fields)))
        self.stats_[:] = np.nan 

        ensembles_samplespecific = OrderedDict()
        for i, query_sample in pv(enumerate(index_samplespecific), description="Computing sample-specific perturbation networks", unit=" samples"):
            # Get reference draws and add in query sample
            draws = copy.deepcopy(draws_reference)
            for draw in draws:
                draw.append(query_sample)
            # Get index
            index_query = index_reference + [query_sample]
            # Create sample-specific network for each query
            with Suppress():
                ensemble_query = EnsembleAssociationNetwork(
                        name=query_sample, 
                        node_type=self.node_type, 
                        edge_type=self.edge_type, 
                        observation_type=self.observation_type, 
                        assert_symmetry=self.assert_symmetry, 
                        assert_nan_safe_functions=self.assert_nan_safe_functions, 
                        nans_ok=self.nans_ok,
                        tol=self.tol,
                )
                # Fit
                ensemble_query.fit(
                        X=X.loc[index_query],
                        metric=metric,
                        n_iter=draws,
                        sampling_size=sampling_size,
                        transformation=transformation,
                        random_state=random_state,
                        with_replacement=with_replacement,
                        function_is_pairwise=function_is_pairwise,
                        stats_summary=stats_summary_initial,
                        stats_tests=stats_tests_initial,
                        copy_X=False,
                        copy_ensemble=True,
                )


            # Weights
            values_query = ensemble_query.ensemble_.values

            # Store
            if not copy_ensemble_samplespecific:
                ensemble_query.memory_ -= ensemble_query.ensemble_memory_
                delattr(ensemble_query, "ensemble_")
            ensembles_samplespecific[query_sample] = ensemble_query

            # Perturbation
            values_perturbation = values_query - values_reference
            if copy_ensemble:
                self.ensemble_[i] = values_perturbation

            # Calculating statistics
            k = 0
            if stats_summary is not None:
                for func in stats_summary:  
                    self.stats_[i,:,k] = func(values_perturbation, axis=0)
                    k += 1
                    
            if stats_tests:# is not None:
                for func in stats_tests:
                    for j in range(number_of_edges):
                        v = values_perturbation[:,j]
                        stat, p = func(v)
                        self.stats_[i, j,[k, k+1]] = [stat,p]
                    k += 2

        # Statistics after calculating difference
        self.stats_ = xr.DataArray(
                data=self.stats_, 
                name=self.name,
                dims=["Samples", "Edges", "Statistics"], 
                coords={
                    "Samples":index_samplespecific, 
                    "Edges":edges, 
                    "Statistics":stat_fields,
                },
        )
        self.stats_memory_ = self.stats_.nbytes
        self.memory_ += self.stats_memory_ 

        # Data
        if copy_X:
            self.X_ = X.copy()
            self.X_memory_ = X.memory_usage().sum()
            self.memory_ += self.X_memory_ 

        if copy_y:
            self.y_ = y.copy()

        # Reference ensemble
        self.ensemble_reference_ = ensemble_reference
        self.ensemble_reference_memory_ = ensemble_reference.memory_
        self.memory_ += self.ensemble_reference_memory_

        # Sample-specific ensembles
        self.ensembles_samplespecific_ = ensembles_samplespecific
        self.ensembles_samplespecific_memory_ = pd.Series(OrderedDict(map(lambda item: (item[0],item[1].memory_), ensembles_samplespecific.items())), name="Memory [Bytes")
        self.memory_ += self.ensembles_samplespecific_memory_.sum()

        # Ensemble
        if hasattr(self, "ensemble_"):
            self.ensemble_ = xr.DataArray(
                    data=self.ensemble_, 
                    name=self.name,
                    dims=["Samples", "Iterations", "Edges"], 
                    coords={
                        "Samples":index_samplespecific, 
                        "Iterations":range(n_iter),
                        "Edges":edges, 
                    },
            )
            self.ensemble_memory_ = self.ensemble_.nbytes
            self.memory_ += self.ensemble_memory_
        
        # Network
        self.n_ = n
        self.m_ = m
        self.nodes_ = nodes
        self.number_of_nodes_ = number_of_nodes
        self.edges_ = edges
        self.number_of_edges_ = number_of_edges
        self.draws_reference_ = draws_reference
        
        # Get metric
        self.metric_ = ensemble_reference.metric_
        self.metric_name = ensemble_reference.metric_name

        # Store parameters and data
        self.reference_ = reference
        self.n_iter = n_iter
        self.sampling_size_ = ensemble_reference.sampling_size_
        self.transformation_ = ensemble_reference.transformation_
        self.random_state = random_state
        self.with_replacement = ensemble_reference.with_replacement

        self.index_reference_ = pd.Index(index_reference, name="Reference={}".format(reference))
        self.index_samplespecific_ = pd.Index(index_samplespecific, name="Sample-specific")

        return self

    # I/O
    # ===
    def to_file(self, path, compression='infer', **kwargs):
        write_object(self, path=path, compression=compression, **kwargs)
        
    # Convert
    # =======
    def to_condensed(self, sample, weight="mean", into=Symmetric):
        if not hasattr(self, "stats_"):
            raise Exception("Please fit model")
        assert weight in self.stats_.coords["Statistics"]
        assert into in {Symmetric, pd.Series}
        sym_network = Symmetric(
            data=self.stats_.sel(Samples=sample, Statistics=weight).to_pandas(), 
            name=self.name, 
            node_type=self.node_type, 
            edge_type=self.edge_type, 
            func_metric=self.metric_, 
            association="network", 
            assert_symmetry=self.assert_symmetry, 
            nans_ok=self.nans_ok, 
            tol=self.tol,
        )
        if into == Symmetric:
            return sym_network
        if into == pd.Series:
            return sym_network.weights
        
    def to_dense(self, sample, weight="mean", fill_diagonal=1):
        df_dense = self.to_condensed(sample=sample, weight=weight).to_dense(index=self.nodes_)
        if fill_diagonal is not None:
            np.fill_diagonal(df_dense.values, fill_diagonal)
        return df_dense

    def to_networkx(self, sample, into=None, **attrs):
        if into is None:
            into = nx.Graph
        if not hasattr(self, "stats_"):
            raise Exception("Please fit model")

        metadata = { "node_type":self.node_type, "edge_type":self.edge_type, "observation_type":self.observation_type, "metric":self.metric_name}
        metadata.update(attrs)
        graph = into(name=sample, **metadata)
        for (node_A, node_B), statistics in pv(self.stats_.sel(Samples=sample).to_pandas().iterrows(), description="Building NetworkX graph from statistics", total=self.number_of_edges_, unit=" edges"):
            graph.add_edge(node_A, node_B, **statistics)
        return graph 

    def copy(self):
        return copy.deepcopy(self)

        
    # Built-in
    # ========
    def __repr__(self):
        pad = 4
        fitted = hasattr(self, "stats_") # Do not keep `fit_`
        if fitted:
            header = format_header("{}(Name:{}, Reference: {}, Metric: {})".format(type(self).__name__, self.name, self.reference_, self.metric_name),line_character="=")
            n = len(header.split("\n")[0])
            fields = [
                header,
                pad*" " + "* Number of nodes ({}): {}".format(self.node_type, self.number_of_nodes_),
                pad*" " + "* Number of edges ({}): {}".format(self.edge_type, self.number_of_edges_),
                pad*" " + "* Observation type: {}".format(self.observation_type),
                *map(lambda line:pad*" " + line, format_header("| Parameters", "-", n=n-pad).split("\n")),
                pad*" " + "* n_iter: {}".format(self.n_iter),
                pad*" " + "* sampling_size: {}".format(self.sampling_size_),
                pad*" " + "* random_state: {}".format(self.random_state),
                pad*" " + "* with_replacement: {}".format(self.with_replacement),
                pad*" " + "* transformation: {}".format(self.transformation_),
                pad*" " + "* memory: {}".format(format_memory(self.memory_)),
                *map(lambda line:pad*" " + line, format_header("| Data", "-", n=n-pad).split("\n")),
                ]
            # Data
            if hasattr(self, "X_"):
                fields.append(pad*" " + "* Features (n={}, m={}, memory={})".format(self.n_, self.m_, format_memory(self.X_memory_)))
            else:
                fields.append(pad*" " + "* Features (n={}, m={})".format(self.n_, self.m_))
            
            # Intermediate
            fields += list(map(lambda line:pad*" " + line, format_header("| Intermediate", "-", n=n-pad).split("\n")))

            fields.append(pad*" " + "* Reference Ensemble (memory={})".format(format_memory(self.ensemble_reference_memory_)))
            fields.append(pad*" " + "* Sample-specific Ensembles (memory={})".format(format_memory(self.ensembles_samplespecific_memory_.sum())))

            # Terminal
            fields += list(map(lambda line:pad*" " + line, format_header("| Terminal", "-", n=n-pad).split("\n")))

            if hasattr(self, "ensemble_"):
                fields.append(pad*" " + "* Ensemble (memory={})".format( format_memory(self.ensemble_memory_)))
            
            fields.append(pad*" " + "* Statistics ({}, memory={})".format(self.stats_.coords["Statistics"].values.tolist(), format_memory(self.stats_memory_)))
            return "\n".join(fields)
        else:
            header = format_header("{}(Name:{})".format(type(self).__name__, self.name),line_character="=")
            n = len(header.split("\n")[0])
            fields = [
                header,
                pad*" " + "* Number of nodes ({}): {}".format(self.node_type, 0),
                pad*" " + "* Number of edges ({}): {}".format(self.edge_type, 0),
                pad*" " + "* Observation type: {}".format(self.observation_type),
                ]
            return "\n".join(fields)

# Differential Ensemble Association Network
class DifferentialEnsembleAssociationNetwork(object):
    def __init__(
        self, 
        name=None,
        node_type=None,
        edge_type=None,
        observation_type=None,
        reference_type=None,
        treatment_type=None,
        assert_symmetry=True,
        assert_nan_safe_functions=True,
        nans_ok=True,
        tol=1e-10,
        ):
        self.name = name
        self.node_type = node_type
        self.edge_type = edge_type
        self.observation_type = observation_type
        self.reference_type = reference_type
        self.treatment_type = treatment_type
        self.assert_symmetry = assert_symmetry
        self.assert_nan_safe_functions = assert_nan_safe_functions
        self.nans_ok = nans_ok
        self.tol = tol
        
    def fit(
        self,
        X:pd.DataFrame,
        y:pd.Series,
        reference,
        treatment,
        metric="rho",
        n_iter=1000,
        sampling_size=0.6180339887,
        transformation=None,
        random_state=0,
        with_replacement=False,
        function_is_pairwise=True,
        stats_comparative = [stats.wasserstein_distance],
        stats_tests_comparative = None,
        stats_summary_initial=[np.mean, np.var, stats.kurtosis, stats.skew],
        stats_tests_initial=[stats.normaltest],
        stats_differential=[np.mean],
        copy_X=True,
        copy_y=True,
        copy_ensemble_reference=True, # This will get very big very quickly
        copy_ensemble_treatment=True, # This will get very big very quickly
#         ensemble_reference=None, OPTION TO INCLUDE A PREVIOUS FITTED ENSEMBLE NETWORK FOR EITHER REFERENCE OR TREATMENT
#         ensemble_treatment=None, THIS WILL BE USEFUL FOR RUNNING MULTIPLE DIN AGAINST SAME REFERENCE NETWORK

        ):
        if stats_comparative is None:
            stats_comparative = []
        if stats_tests_comparative is None:
            stats_tests_comparative = []
        assert stats_summary_initial is not None, "`stats_summary_initial` cannot be None.  Recommended using either np.mean or np.median"
        if stats_differential is None:
            stats_differential = True
        if stats_differential is True:
            stats_differential = stats_summary_initial
        if hasattr(stats_differential, "__call__"):
            stats_differential = [stats_differential] # Make it string compatible here.  It's tricky because need to check if its in `stats_summary_initial`
        assert set(stats_differential) <= set(stats_summary_initial), "Please include at least one stat from `stats_summary_initial` such as `np.mean` or `np.median`"

        # Memory
        self.memory_ = 0
        
        assert reference in y.unique(), "`reference({}, type:{}) not in `y`".format(reference, type(reference))
        assert treatment in y.unique(), "`treatment({}, type:{}) not in `y`".format(treatment, type(treatment))

        assert set(X.index) == set(y.index), "`X.index` must have same keys as `y.index`"
        y = y[X.index]
 
        # Ensemble Reference 
        index_reference = sorted(y[lambda i: i == reference].index)
        ensemble_reference = EnsembleAssociationNetwork(
                name=reference, 
                node_type=self.node_type, 
                edge_type=self.edge_type, 
                observation_type=self.observation_type, 
                assert_symmetry=self.assert_symmetry, 
                assert_nan_safe_functions=self.assert_nan_safe_functions,
                nans_ok=self.nans_ok,
                tol=self.tol,
        )
        # Fit
        ensemble_reference.fit(
                X=X.loc[index_reference],
                metric=metric,
                n_iter=n_iter,
                sampling_size=sampling_size,
                transformation=transformation,
                random_state=random_state,
                with_replacement=with_replacement,
                function_is_pairwise=function_is_pairwise,
                stats_summary=stats_summary_initial,
                stats_tests=stats_tests_initial,
                copy_X=False,
                copy_ensemble=True,
        )

        # Treatment samples
        index_treatment = y[lambda i: i != treatment].index #sorted(set(X.index) - set(index_reference))
        ensemble_treatment = EnsembleAssociationNetwork(
                name=treatment, 
                node_type=self.node_type, 
                edge_type=self.edge_type, 
                observation_type=self.observation_type, 
                assert_symmetry=self.assert_symmetry, 
                assert_nan_safe_functions=self.assert_nan_safe_functions,
                nans_ok=self.nans_ok,
                tol=self.tol,
        )
        # Fit
        ensemble_treatment.fit(
                X=X.loc[index_treatment],
                metric=metric,
                n_iter=n_iter,
                sampling_size=sampling_size,
                transformation=transformation,
                random_state=random_state,
                with_replacement=with_replacement,
                function_is_pairwise=function_is_pairwise,
                stats_summary=stats_summary_initial,
                stats_tests=stats_tests_initial,
                copy_X=False,
                copy_ensemble=True,
        )
        assert np.all(ensemble_reference.ensemble_.columns == ensemble_treatment.ensemble_.columns), "Edges must match between `ensemble_reference_` and `ensemble_treatment_`"
        # Data
        n, m = X.shape

        # Network
        nodes = ensemble_reference.nodes_
        number_of_nodes = ensemble_reference.number_of_nodes_
        edges = ensemble_reference.edges_
        number_of_edges = ensemble_reference.number_of_edges_

        # Statistics
        number_of_statistic_fields = 0

        stat_fields = list()
        if stats_comparative:# is not None:
            for func in stats_comparative:  
                stat_name = func.__name__
                stat_fields.append(stat_name)
        if stats_tests_comparative:# is not None:
            for func in stats_tests_comparative:
                stat_name = func.__name__
                stat_fields.append("{}|stat".format(stat_name))
                stat_fields.append("{}|p_value".format(stat_name))

        self.stats_comparative_ = np.empty((number_of_edges, len(stat_fields)))
        self.stats_comparative_[:] = np.nan 



        # Comparative statistics
        k = 0
        if stats_comparative is not None:
            for func in stats_comparative:  
                for i in range(number_of_edges):
                    u = ensemble_reference.ensemble_.values[:,i]
                    v = ensemble_treatment.ensemble_.values[:,i]
                    self.stats_comparative_[i,k] = func(u, v)
                k += 1

            if stats_tests_comparative:# is not None:
                for func in stats_tests_comparative:
                    for i in range(number_of_edges):
                        u = ensemble_reference.ensemble_.values[:,i]
                        v = ensemble_treatment.ensemble_.values[:,i]
                        stat, p = func(u, v)
                        self.stats_comparative_[i, [k, k+1]] = [stat,p]
                    k += 2
                    
            
       # Comparative statistics
        self.stats_comparative_ = pd.DataFrame(
                data=self.stats_comparative_, 
                index=edges,
                columns=stat_fields,
        )

        self.stats_comparative_memory_ = self.stats_comparative_.memory_usage().sum()
        self.memory_ += self.stats_comparative_memory_ 

        # Differential statistics
        self.ensemble_ = list()
        for func in stats_differential:
            func_name = func
            if hasattr(func, "__call__"):
                func_name = func.__name__
            distribution_reference = ensemble_reference.stats_[func_name]
            distribution_treatment = ensemble_treatment.stats_[func_name]
            differential = pd.Series(distribution_treatment - distribution_reference, name=func_name)
            self.ensemble_.append(differential)
        self.ensemble_ = pd.DataFrame(self.ensemble_).T
        self.ensemble_memory_ = self.ensemble_.memory_usage().sum()
        self.memory_ += self.ensemble_memory_
        
        # Remove ensemble_ if relevant
        if not copy_ensemble_reference:
            ensemble_reference.memory_ -= ensemble_reference.ensemble_memory_
            delattr(ensemble_reference, "ensemble_")
        if not copy_ensemble_treatment:
            ensemble_treatment.memory_ -= ensemble_treatment.ensemble_memory_
            delattr(ensemble_treatment, "ensemble_")
            
        # Data
        if copy_X:
            self.X_ = X.copy()
            self.X_memory_ = X.memory_usage().sum()
            self.memory_ += self.X_memory_ 

        if copy_y:
            self.y_ = y.copy()

        # Reference ensemble
        self.ensemble_reference_ = ensemble_reference
        self.ensemble_reference_memory_ = ensemble_reference.memory_
        self.memory_ += self.ensemble_reference_memory_
        
        # Treatment ensemble
        self.ensemble_treatment_ = ensemble_treatment
        self.ensemble_treatment_memory_ = ensemble_treatment.memory_
        self.memory_ += self.ensemble_treatment_memory_

        
        # Network
        n,m = X.shape
        self.n_ = n
        self.m_ = m
        self.nodes_ = nodes
        self.number_of_nodes_ = number_of_nodes
        self.edges_ = edges
        self.number_of_edges_ = number_of_edges
        
        # Get metric
        self.metric_ = ensemble_reference.metric_
        self.metric_name = ensemble_reference.metric_name

        # Store parameters and data
        self.reference_ = reference
        self.treatment_ = treatment

        self.n_iter = n_iter
        self.sampling_size_ = ensemble_reference.sampling_size_
        self.transformation_ = ensemble_reference.transformation_
        self.random_state = random_state
        self.with_replacement = ensemble_reference.with_replacement

        self.index_reference_ = pd.Index(index_reference, name="Reference={}".format(reference))
        self.index_treatment_ = pd.Index(index_treatment, name="Treatment={}".format(treatment))

        return self
    
    
        # I/O
    # ===
    def to_file(self, path, compression='infer', **kwargs):
        write_object(self, path=path, compression=compression, **kwargs)
        
    # Convert
    # =======
    def to_condensed(self, weight="mean", into=Symmetric):
        if not hasattr(self, "ensemble_"):
            raise Exception("Please fit model")
        assert weight in self.ensemble_.columns
        assert into in {Symmetric, pd.Series}
        sym_network = Symmetric(
            data=self.ensemble_[weight], 
            name=self.name, 
            node_type=self.node_type, 
            edge_type=self.edge_type, 
            func_metric=self.metric_, 
            association="network", 
            assert_symmetry=self.assert_symmetry, 
            nans_ok=self.nans_ok, 
            tol=self.tol,
        )
        if into == Symmetric:
            return sym_network
        if into == pd.Series:
            return sym_network.weights
        
    def to_dense(self, weight="mean", fill_diagonal=1):
        df_dense = self.to_condensed(weight=weight).to_dense(index=self.nodes_)
        if fill_diagonal is not None:
            np.fill_diagonal(df_dense.values, fill_diagonal)
        return df_dense

    def to_networkx(self, into=None, **attrs):
        if into is None:
            into = nx.Graph
        if not hasattr(self, "ensemble_"):
            raise Exception("Please fit model")

        metadata = { "node_type":self.node_type, "edge_type":self.edge_type, "observation_type":self.observation_type, "metric":self.metric_name}
        metadata.update(attrs)
        graph = into(name=self.name, **metadata)
        for (node_A, node_B), statistics in pv(self.ensemble_.iterrows(), description="Building NetworkX graph from statistics", total=self.number_of_edges_, unit=" edges"):
            graph.add_edge(node_A, node_B, **statistics)
        return graph 

    def copy(self):
        return copy.deepcopy(self)
    
    # Built-in
    # ========
    def __repr__(self):
        pad = 4
        fitted = hasattr(self, "ensemble_") # Do not keep `fit_`
        if fitted:
            header = format_header("{}(Name:{}, Reference: {}, Treatment: {}, Metric: {})".format(type(self).__name__, self.name, self.reference_, self.treatment_, self.metric_name),line_character="=")
            n = len(header.split("\n")[0])
            fields = [
                header,
                pad*" " + "* Number of nodes ({}): {}".format(self.node_type, self.number_of_nodes_),
                pad*" " + "* Number of edges ({}): {}".format(self.edge_type, self.number_of_edges_),
                pad*" " + "* Observation type: {}".format(self.observation_type),
                *map(lambda line:pad*" " + line, format_header("| Parameters", "-", n=n-pad).split("\n")),
                pad*" " + "* n_iter: {}".format(self.n_iter),
                pad*" " + "* sampling_size: {}".format(self.sampling_size_),
                pad*" " + "* random_state: {}".format(self.random_state),
                pad*" " + "* with_replacement: {}".format(self.with_replacement),
                pad*" " + "* transformation: {}".format(self.transformation_),
                pad*" " + "* memory: {}".format(format_memory(self.memory_)),
                *map(lambda line:pad*" " + line, format_header("| Data", "-", n=n-pad).split("\n")),
                ]
            # Data
            if hasattr(self, "X_"):
                fields.append(pad*" " + "* Features (n={}, m={}, memory={})".format(self.n_, self.m_, format_memory(self.X_memory_)))
            else:
                fields.append(pad*" " + "* Features (n={}, m={})".format(self.n_, self.m_))
            
            # Intermediate
            fields += list(map(lambda line:pad*" " + line, format_header("| Intermediate", "-", n=n-pad).split("\n")))

            fields.append(pad*" " + "* Reference Ensemble (memory={})".format(format_memory(self.ensemble_reference_memory_)))
            fields.append(pad*" " + "* Treatment Ensemble (memory={})".format(format_memory(self.ensemble_treatment_memory_)))

            # Terminal
            fields += list(map(lambda line:pad*" " + line, format_header("| Terminal", "-", n=n-pad).split("\n")))

            fields.append(pad*" " + "* Initial Statistics ({})".format(self.ensemble_reference_.stats_.columns.tolist()))
            fields.append(pad*" " + "* Comparative Statistics ({}, memory={})".format(self.stats_comparative_.columns.tolist(), format_memory(self.stats_comparative_memory_)))
            fields.append(pad*" " + "* Differential Statistics ({}, memory={})".format(self.ensemble_.columns.tolist(), format_memory(self.ensemble_memory_)))

            return "\n".join(fields)
        else:
            header = format_header("{}(Name:{})".format(type(self).__name__, self.name),line_character="=")
            n = len(header.split("\n")[0])
            fields = [
                header,
                pad*" " + "* Number of nodes ({}): {}".format(self.node_type, 0),
                pad*" " + "* Number of edges ({}): {}".format(self.edge_type, 0),
                pad*" " + "* Observation type: {}".format(self.observation_type),
                ]
            return "\n".join(fields)