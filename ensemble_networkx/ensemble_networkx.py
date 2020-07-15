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
    X = sy.io.get_iris_data(["X"])
    # Create ensemble network
    ens = EnsembleAssociationNetwork(name="Iris", node_type="leaf measurement", edge_type="rho", observation_type="specimen")
    ens.fit(X, n_iter=100)
    # ===========================================================
    # EnsembleAssociationNetwork(Name:Iris, Metric: pairwise_rho)
    # ===========================================================
    #     * Number of nodes (leaf measurement): 4
    #     * Number of edges (rho): 6
    #     * Observation type: specimen
    #     -------------------------------------------------------
    #     | Parameters
    #     -------------------------------------------------------
    #     * n_iter: 100
    #     * sampling_size: 92
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
    # Get data
    import soothsayer_utils as syu
    X, y, colors = syu.get_iris_data(["X","y", "colors"])
    reference = "setosa"
    # Create SSPN
    sspn = SampleSpecificPerturbationNetwork(name="Iris")
    sspn.fit(X,y,reference, n_iter=1000)
    # Computing associations (setosa): 100%|██████████| 1000/1000 [00:00<00:00, 1366.95 draws/s]
    # Computing sample-specific perturbation networks: 100 samples [01:08,  1.46 samples/s]

    sspn
    # =====================================================================================
    # SampleSpecificPerturbationNetwork(Name:Iris, Reference: setosa, Metric: pairwise_rho)
    # =====================================================================================
    #     * Number of nodes (None): 4
    #     * Number of edges (None): 6
    #     * Observation type: None
    #     ---------------------------------------------------------------------------------
    #     | Parameters
    #     ---------------------------------------------------------------------------------
    #     * n_iter: 1000
    #     * sampling_size: 30
    #     * random_state: 0
    #     * memory: 43.672 KB
    #     ---------------------------------------------------------------------------------
    #     | Data
    #     ---------------------------------------------------------------------------------
    #     * Features (n=150, m=4, memory=10.859 KB)
    #     * Statistics (['mean', 'median', 'var', 'kurtosis', 'skew', 'normaltest|stat', 'normaltest|p_value'], memory=32.812 KB)
    
    # Get SSPN
    sspn.stats_.sel(Statistics="mean").to_pandas()
    # Edges	(sepal_width, sepal_length)	(sepal_length, petal_length)	(petal_width, sepal_length)	(sepal_width, petal_length)	(petal_width, sepal_width)	(petal_width, petal_length)
    # Samples						
    # iris_100	0.127281	-0.646881	-0.119744	-0.565109	-0.185594	0.457830
    # iris_101	0.090062	-0.622146	-0.095830	-0.582579	-0.189793	0.447629
    # iris_102	0.053159	-0.596144	-0.076951	-0.592005	-0.194104	0.443015
    # iris_103	0.080395	-0.642868	-0.093303	-0.618016	-0.183654	0.448550
    # iris_104	0.090275	-0.630976	-0.096987	-0.588645	-0.193117	0.455613
    # ...	...	...	...	...	...	...
    # iris_95	0.087047	-0.510475	-0.079858	-0.495492	-0.139211	0.336770
    # iris_96	0.083525	-0.506814	-0.078997	-0.495323	-0.149577	0.349434
    # iris_97	0.050871	-0.469799	-0.061969	-0.497563	-0.153025	0.340282
    # iris_98	0.056155	-0.324093	-0.049098	-0.346397	-0.141890	0.274081
    # iris_99	0.073023	-0.489976	-0.072948	-0.490904	-0.153607	0.348242

    sspn.stats_
    # xarray.DataArray'Iris'Samples: 100Edges: 6Statistics: 7
    # 0.1273 0.1238 0.00184 0.07746 0.4722 ... 0.2709 0.6676 65.84 5.044e-15
    # Coordinates:
    # Samples
    # (Samples)
    # <U8
    # 'iris_100' 'iris_101' ... 'iris_99'
    # Edges
    # (Edges)
    # object
    # frozenset({'sepal_width', 'sepal_length'}) ... frozenset({'petal_width', 'petal_length'})
    # Statistics
    # (Statistics)
    # <U18
    # 'mean' ... 'normaltest|p_value'
    # Attributes: (0)
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