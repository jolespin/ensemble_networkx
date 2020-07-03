# -*- coding: utf-8 -*-
from __future__ import print_function, division

# ==============================================================================
# Modules
# ==============================================================================
# Built-ins
import os, sys, copy, warnings
from collections import defaultdict
from itertools import combinations

# PyData
import pandas as pd
import numpy as np
import networkx as nx
from scipy import stats
from scipy.special import comb
from scipy.spatial.distance import squareform

# Hive NetworkX
from hive_networkx import Symmetric, signed

# Compositional
from compositional import pairwise_rho, pairwise_phi

# soothsayer_utils
from soothsayer_utils import pv, assert_acceptable_arguments, is_symmetrical, is_graph, write_object, format_memory, format_header, format_path
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
    X = sy.io.get_iris_data(["X"])
    # Create ensemble network
    ens = EnsembleAssociationNetwork(name="Iris", node_type="leaf measurement", edge_type="rho", observation_type="specimen")
    ens.fit(X, n_iter=100)
    # ===========================================================
    # EnsembleAssociationNetwork(Name:Iris, method: pairwise_rho)
    # ===========================================================
    #     * Number of nodes (leaf measurement): 4
    #     * Number of edges (rho): 6
    #     * Observation type: specimen
    #     -------------------------------------------------------
    #     | Parameters
    #     -------------------------------------------------------
    #     * n_iter: 100
    #     * sample_size: 92
    #     * random_state: 0
    #     * memory: 11.047 KB
    #     -------------------------------------------------------
    #     | Data
    #     -------------------------------------------------------
    #     * Features (n=150, m=4, memory=5.859 KB)
    #     * Ensemble (memory=4.812 KB)
    #     * Statistics (['mean', 'median', 'var', 'kurtosis', 'skew', 'normaltest|stat', 'normaltest|p_value'], memory=384 B)

    # View summary stats
    ens.stats_
    # Statistics	mean	median	var	kurtosis	skew	normaltest|stat	normaltest|p_value
    # Edges							
    # (sepal_length, sepal_width)	0.855421	0.854771	0.000066	0.363698	0.151548	1.472627	0.478876
    # (sepal_length, petal_length)	-0.801119	-0.799696	0.000445	-0.039800	-0.169622	0.590829	0.744223
    # (sepal_length, petal_width)	-0.803683	-0.804366	0.000045	-0.814222	-0.103616	6.422311	0.040310
    # (sepal_width, petal_length)	-0.672675	-0.671192	0.000423	0.065722	-0.447298	3.719766	0.155691
    # (petal_width, sepal_width)	-0.973008	-0.972823	0.000014	0.386471	-0.204413	1.898091	0.387110
    # (petal_width, petal_length)	0.487433	0.484487	0.000692	0.124931	0.517601	4.929038	0.085050

    # View ensemble network
    ens.ensemble_
    # Edges	(sepal_length, sepal_width)	(sepal_length, petal_length)	(sepal_length, petal_width)	(sepal_width, petal_length)	(petal_width, sepal_width)	(petal_width, petal_length)
    # n_iter=100						
    # 0	0.849206	-0.793988	-0.799695	-0.672147	-0.971270	0.481301
    # 1	0.855512	-0.828456	-0.805577	-0.687045	-0.978446	0.511237
    # 2	0.868281	-0.778928	-0.804862	-0.654763	-0.968321	0.459651
    # 3	0.859762	-0.812899	-0.808884	-0.682954	-0.974394	0.499853
    # 4	0.858101	-0.786517	-0.803897	-0.667418	-0.969415	0.473680
    # ...	...	...	...	...	...	...
    # 95	0.856910	-0.815229	-0.816463	-0.701705	-0.973710	0.519174
    # 96	0.853866	-0.796767	-0.799694	-0.669505	-0.972546	0.480086
    # 97	0.858599	-0.749676	-0.799779	-0.645614	-0.963164	0.441183
    # 98	0.856120	-0.845143	-0.811837	-0.698145	-0.981585	0.532996
    # 99	0.860316	-0.782562	-0.797666	-0.651384	-0.970254	0.458598
    __future__: 
        * Add ability to load in previous data.  However, this is tricky because one needs to validate that the following objects are the same: 
            - X
            - sample_size
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

            
        
    def _pandas_correlation(self, X, method):
        return X.corr(method=method)
    
    def fit(
        self,
        X:pd.DataFrame,
        method="rho",
        n_iter=1000,
        sample_size=0.6180339887,
        transformation=None,
        random_state=0,
        with_replacement=False,
        function_is_pairwise=True,
        stats_summary=[np.mean, np.median, np.var, stats.kurtosis, stats.skew] ,
        stats_tests=[stats.normaltest],
        copy_X=True,
        copy_ensemble=True,
        ):
        
        # Metric
        assert method is not None
        acceptable_methods = {"rho", "phi", "biweight_midcorrelation", "spearman", "pearson", "kendall"}
        if isinstance(method, str):
            assert_acceptable_arguments(method, acceptable_methods)
            if method == "rho":
                method = pairwise_rho
            if method == "phi":
                method = pairwise_phi
            if method == "biweight_midcorrelation":
                method = pairwise_biweight_midcorrelation
            if method in {"spearman", "pearson", "kendall"}:
                method = lambda X: self._pandas_correlation(X, method)
            
        if hasattr(method, "__call__"):
            if not function_is_pairwise:
                method = lambda X: self._pandas_correlation(X, method)
                
        assert hasattr(method, "__call__"), "`method` must be either one of the following: [{}], \
        a custom metric that returns an association (set `function_is_pairwise=False`), or a custom \
        metric that returns a 2D square/symmetric pd.DataFrame (set `function_is_pairwise=True`)".format(acceptable_methods)
        
        # Transformations
        acceptable_transformations = {"signed", "abs"}
        if transformation:
            if isinstance(transformation, str):
                assert_acceptable_arguments(transformation, acceptable_transformations)
                if method == "signed":
                    method = signed
                if method == "abs":
                    method = np.abs
            assert hasattr(method, "__call__"), "`method` must be either one of the following: [{}] or a function(pd.DataFrame) -> pd.DataFrame".format(acceptable_transformations)
            
        # Sample size
        n, m = X.shape
        assert 0 < sample_size < n
        if 0 < sample_size < 1:
            sample_size = int(sample_size*n)
            
        # Iterations
        number_of_unique_draws_possible = comb(n, sample_size, exact=True, repetition=with_replacement)
        assert n_iter <= number_of_unique_draws_possible, "`n_iter` exceeds the number of possible draws (total_possible={})".format(number_of_unique_draws_possible)
                
        # Parameters
        self.memory_ = 0
        if copy_X: 
            self.X_ = X.copy()
            self.X_memory_ = X.memory_usage().sum()
            self.memory_ += self.X_memory_ 
        self.n_iter = n_iter
        self.sample_size = sample_size
        self.transformation = transformation
        self.function_is_pairwise = function_is_pairwise
        self.random_state = random_state
        self.method = method
        
        # Network
        self.n_ = n
        self.m_ = m
        self.nodes_ = pd.Index(X.columns)
        self.number_of_nodes_ = len(self.nodes_)
        self.edges_ = pd.Index(map(frozenset, combinations(self.nodes_, r=2)), name="Edges")
        self.number_of_edges_ = len(self.edges_)
        
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
        
        
#         # Save data #! This is a template to use in the future for loading in existing data
#         ext = {None:"", "gzip":".gz", "bz2":".bz2"}[self.compression]
        
#         if self.temporary_directory is not None:
#             write_parameters = False
            
#             if not self.force_overwrite:
#                 while write_parameters == False:
#                     # Random state
#                     path = os.path.join(self.temporary_directory, "random_state.pkl{}".format(ext))
#                     if os.path.exists(path):
#                         if read_object(path) != self.random_state:
#                                 write_parameters = True

#         if self.temporary_directory is not None:

        # Associations
        ensemble = np.empty((self.n_iter, self.number_of_edges_))
        ensemble[:] = np.nan
        i = 0
        for j in pv(range(self.n_iter), description="Computing associations", total=self.n_iter, unit=" draws"):
            # Get draw of samples
            if self.random_state is None:
                rs = None
            else:
                rs = j + self.random_state
            index = np.random.RandomState(rs).choice(X.index,size=self.sample_size, replace=with_replacement) 
            # Compute associations with current draw
            df_associations = method(X.loc[index])
            if self.transformation is not None:
                df_associations = self.transformation(df_associations)
                
            if self.assert_symmetry:
                assert is_symmetrical(df_associations, tol=self.tol)
            weights = squareform(df_associations.values, checks=False) #dense_to_condensed(X=df_associations, assert_symmetry=self.assert_symmetry, tol=self.tol)
            ensemble[i] = weights
#             if self.temporary_directory:
#                 path = os.path.join(self.temporary_directory, "network_{}.npy".format(j))
#                 np.save(path, weights)
            i += 1
            
                
        ensemble = pd.DataFrame(ensemble, columns=self.edges_)
        ensemble.columns.name = "Edges"
        ensemble.index.name = "n_iter={}".format(self.n_iter)
        
        if copy_ensemble:
            self.ensemble_ = ensemble
            self.ensemble_memory_ = ensemble.memory_usage().sum()
            self.memory_ += self.ensemble_memory_ 
            
        # Statistics
        self.stats_ = defaultdict(dict) # ensemble.describe(percentiles=percentiles).to_dict()
        for edge, u  in pv(ensemble.T.iterrows(), description="Computing summary statistics", total=self.number_of_edges_, unit=" edges"):
            for func in stats_summary:
                name = func.__name__
                self.stats_[edge][name] = func(u)
                
            for func in stats_tests:
                name = func.__name__
                stat, p = func(u)
                self.stats_[edge]["{}|stat".format(name)] = stat
                self.stats_[edge]["{}|p_value".format(name)] = p
        self.stats_ = pd.DataFrame(self.stats_).T
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
            func_metric=self.method, 
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

        metadata = { "node_type":self.node_type, "edge_type":self.edge_type, "observation_type":self.observation_type, "method":self.method}
        metadata.update(attrs)
        graph = into(name=self.name, **metadata)
        for (node_A, node_B), statistics in pv(self.stats_.iterrows(), description="Building NetworkX graph from statistics", total=self.number_of_edges_, unit=" edges"):
            graph.add_edge(node_A, node_B, **statistics)
        return graph        
    
    # Built-in
    # ========
    def __repr__(self):
        pad = 4
        fitted = hasattr(self, "stats_")
        if fitted:
            header = format_header("EnsembleAssociationNetwork(Name:{}, method: {})".format(self.name, self.method.__name__),line_character="=")
            n = len(header.split("\n")[0])
            fields = [
                header,
                pad*" " + "* Number of nodes ({}): {}".format(self.node_type, self.number_of_nodes_),
                pad*" " + "* Number of edges ({}): {}".format(self.edge_type, self.number_of_edges_),
                pad*" " + "* Observation type: {}".format(self.observation_type),
                *map(lambda line:pad*" " + line, format_header("| Parameters", "-", n=n-pad).split("\n")),
                pad*" " + "* n_iter: {}".format(self.n_iter),
                pad*" " + "* sample_size: {}".format(self.sample_size),
                pad*" " + "* random_state: {}".format(self.random_state),
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
            header = format_header("EnsembleAssociationNetwork(Name:{})".format(self.name),line_character="=")
            n = len(header.split("\n")[0])
            fields = [
                header,
                pad*" " + "* Number of nodes ({}): {}".format(self.node_type, 0),
                pad*" " + "* Number of edges ({}): {}".format(self.edge_type, 0),
                pad*" " + "* Observation type: {}".format(self.observation_type),
                ]
            return "\n".join(fields)

