# -*- coding: utf-8 -*-
from __future__ import print_function, division

# ==============================================================================
# Modules
# ==============================================================================
# Built-ins
import os, sys, time, datetime, copy, warnings
from typing import Dict, Union, Any
from collections import defaultdict, OrderedDict
from collections.abc import Mapping, Hashable
from itertools import combinations, product

# Packages
import pandas as pd
import numpy as np
import networkx as nx
import igraph as ig
import xarray as xr
from scipy import stats
from scipy.special import comb
from scipy.spatial.distance import squareform, pdist

# Compositional
from compositional import pairwise_rho, pairwise_phi

# soothsayer_utils
from soothsayer_utils import pv, flatten, assert_acceptable_arguments, is_symmetrical, is_graph, write_object, format_memory, format_header, format_path, is_nonstring_iterable, Suppress, dict_build, dict_filter, is_dict, is_dict_like, is_color, is_number, check_packages, is_query_class

try:
    from . import __version__
except ImportError:
    __version__ = "ImportError: attempted relative import with no known parent package"


# ===================
# Converting Networks
# ===================  
# Remove self-loops
def remove_self_edges(graph):
    self_edges = list(nx.selfloop_edges(graph))
    return graph.remove_edges_from(self_edges)

# pd.DataFrame 2D to pd.Series
def dense_to_condensed(X, name=None, assert_symmetry=True, tol=None, nans_ok=True):
    if assert_symmetry:
        assert is_symmetrical(X, tol=tol, nans_ok=nans_ok), "`X` is not symmetric with tol=`{}` or try with `nans_ok=True`".format(tol)
    labels = X.index
    index = pd.Index(list(map(frozenset, combinations(labels, 2))), name=name)
    data = squareform(X, checks=False)
    return pd.Series(data, index=index, name=name)

# pd.Series to pd.DataFrame 2D
def condensed_to_dense(y:pd.Series, fill_diagonal=None, index=None):

    # Need to optimize this
    data = defaultdict(dict)
    for edge, w in y.iteritems():
        number_of_unique_nodes = len(edge)
        if len(edge) == 2:
            node_a, node_b = tuple(edge)
        else:
            node_a = node_b = list(edge)[0]
            
        data[node_a][node_b] = data[node_b][node_a] = w
        
    if fill_diagonal is not None:
        if is_dict_like(fill_diagonal):
            for node in data:
                data[node][node] = fill_diagonal[node]
        else:
            for node in data:
                data[node][node] = fill_diagonal
            
    df_dense = pd.DataFrame(data)
    if index is None:
        index = sorted(df_dense.index)
    return df_dense.loc[index,index]

# Get symmetric category
def get_symmetric_category(obj):
    """
    Future: Add support for Hive objects
    """
    if type(obj) is type:
        string_representation = str(obj)
    else:
        string_representation = str(type(obj))
    class_type = string_representation.split("'")[1]
    fields = class_type.split(".")
    module = fields[0]
    category = None
    if fields[-1] == "Symmetric":
        category = "Symmetric"
    if module == "pandas":
        if fields[-1] == "Series":
            category = ("pandas", "Series")
        if fields[-1] == "DataFrame":
            category = ("pandas", "DataFrame")
    if module in  {"networkx", "igraph"}:
        category = module
    assert category is not None, "`data` must be either a pandas[Series, DataFrame], ensemble_networkx[Symmetric], networkx[Graph, DiGraph, OrdereredGraph, DiOrderedGraph], igraph[Graph]"
    return category
    
# Convert Networks
def convert_network(
    # I/O
    data, 
    into, 
    
    # Subgraphs
    node_subgraph:list=None, 
    edge_subgraph:list=None, 
        
    # Symmetry
    remove_self_interactions:bool=True,
    assert_symmetry=True, 
    tol=None, 

    # Missing values
    remove_missing_values:bool=True,
    fill_missing_values_with=np.nan,
    fill_diagonal=np.nan, 
    
    # Attributes
    # propogate_input_attributes=False,
    **attrs,
    ):
    """
    Convert to and from the following network structures:
        * pd.DataFrame (must be symmetrical)
        * pd.Series (index must be frozenset of {node_a, node_b})
        * Symmetric
        * nx.[Di|Ordered]Graph
        * ig.Graph
    
    Future: 
    Add support for existing attributes to propogate to next level. Currently, Symmetric isn't supported so I removed it entirely.
    Add support for Hive objects
    """
    input_category = get_symmetric_category(data)
    output_category = get_symmetric_category(into)

    assert output_category in {("pandas", "Series"), ("pandas", "DataFrame"), "Symmetric", "networkx", "igraph"}, "`data` must be either a pandas[Series, DataFrame], ensemble_networkx[Symmetric], networkx[Graph, DiGraph, OrdereredGraph, DiOrderedGraph], igraph[Graph]"
    assert isinstance(into, type), "`into` must be an instantiated object: a pandas[Series, DataFrame], ensemble_networkx[Symmetric], networkx[Graph, DiGraph, OrdereredGraph, DiOrderedGraph], igraph[Graph]"
    assert into not in {nx.MultiGraph, nx.MultiDiGraph},  "`into` cannot be a `Multi[Di]Graph`"
    
    
    assert not (node_subgraph is not None) & (edge_subgraph is not None), "Cannot create subgraph from `node_subgraph` and `edge_subgraph`"
    
    if all([
        input_category == output_category,
        node_subgraph is None, 
        edge_subgraph is None, 
        remove_self_interactions == False,
        ]):
        return data.copy()
    
    _attrs = dict()
    
    # Convert to pd.Series
    if input_category == "igraph":
        # iGraph -> NetworkX
        if all([
            output_category == "network",
            node_subgraph is None, 
            edge_subgraph is None, 
            remove_self_interactions == False,
            ]):
            return data.to_networkx(into)
        else:
            if data.is_directed():
                warnings.warn("Currently conversions cannot handle directed graphs and will force into undirected configuration")
            weights = data.es["weight"]
            nodes = np.asarray(data.vs["name"])
            edges = list(map(lambda edge_indicies: frozenset(nodes[list(edge_indicies)]), data.get_edgelist()))
            data = pd.Series(weights, index=edges)
            input_category == ("pandas", "Series")
        
    if input_category == "networkx":
        # Update attrs
        # _attrs.update(graph.data)
        # NetworkX -> iGraph
        if all([
            output_category == "igraph",
            node_subgraph is None, 
            edge_subgraph is None, 
            remove_self_interactions == False,
            ]):
            return ig.Graph.from_networkx(data)
        else:
            # Weights
            edge_weights = dict()
            for edge_data in data.edges(data=True):
                edge = frozenset(edge_data[:-1])
                weight = edge_data[-1]["weight"]
                edge_weights[edge] = weight
            data = pd.Series(edge_weights)
            input_category == ("pandas", "Series")
            
    if input_category == ("pandas", "DataFrame"):
        assert len(data.shape) == 2, "`data.shape` must be square (2 dimensions)"
        assert np.all(data.index == data.columns), "`data.index` must equal `data.columns`"
        # if assert_symmetry:
        #     assert is_symmetrical(data, tol=tol, nans_ok=True), "`data` is not symmetric with tol=`{}` or try with `nans_ok=True`".format(tol)
        # data = data.stack()
        # data.index = data.index.map(frozenset)
        
        data = dense_to_condensed(data, assert_symmetry=assert_symmetry, tol=tol, nans_ok=True)
        input_category == ("pandas", "Series")

    if input_category == "Symmetric":
        data = data.weights
        input_category == ("pandas", "Series")
        
    
    if input_category == ("pandas", "Series"):
        if data.name is not None:
            if "name" in _attrs:
                if _attrs["name"] is None:
                    _attrs["name"] = data.name
            else:
                _attrs["name"] = data.name

    # Overwrite existing attrs with provided attrs
    for k, v in attrs.items():
        if v is not None:
            _attrs[k] = v
    
    # Remove duplicates
    data = data[~data.index.duplicated()]
    
    # Subgraphs
    if node_subgraph is not None:
        nodes = frozenset.union(*data.index)
        node_subgraph = frozenset(node_subgraph)
        assert node_subgraph <= nodes, "`node_subgraph` must be a subset of the nodes in `data`"
        edge_subgraph = sorted(set(data.index[data.index.map(lambda edge: edge <= node_subgraph)]))
        
    if edge_subgraph is not None:
        assert set(edge_subgraph) <= set(data.index), "`edge_subgraph` must be a subset of the edge set in `data`"
        data = data[edge_subgraph]
        
    if remove_self_interactions:
        data = data[data.index.map(lambda edge: len(edge) > 1)]
        
    # Missing values
    if remove_missing_values:
        data = data.dropna()
    else:
        if data.isnull().sum() > 0:
            data = data.fillna(fill_missing_values_with)
            
    # Output
    if output_category == ("pandas", "Series"):
        data.name = _attrs.get("name", None)
        return data
    
    if output_category == ("pandas", "DataFrame"):
        return condensed_to_dense(data, fill_diagonal=fill_diagonal)
    
    if output_category == "networkx":
        graph = into(**_attrs)
        for edge, w in data.items():
            if len(edge) == 1:
                node_a = node_b = list(edge)[0]
            else:
                node_a, node_b = list(edge)
            graph.add_edge(node_a, node_b, weight=w)
        return graph
    
    if output_category == "igraph":
        graph = into() 
        # nodes = sorted(flatten(data.index.map(list), unique=True))
        nodes = sorted(frozenset.union(*data.index))
        graph.add_vertices(nodes)
        for edge, w in data.items():
            if len(edge) == 1:
                node_a = node_b = list(edge)[0]
            else:
                node_a, node_b = list(edge)
            graph.add_edge(node_a, node_b, weight=w)
        for k, v in _attrs.items():
            graph[k] = v
        return graph
    
    if output_category == "Symmetric":
        return Symmetric(data=data, **_attrs)
    
# ===================
# Network Statistics
# ===================
# Connectivity
def connectivity(data, groups:pd.Series=None, remove_self_interactions=True, tol=1e-10):
    """
    Calculate connectivity from pd.DataFrame (must be symmetric), Symmetric, Hive, or NetworkX graph
    
    groups must be dict-like: {node:group}
    """
    # This is a hack to allow Hives from hive_networkx
    if is_query_class(data, "Hive"):
        data = data.weights
    # assert isinstance(data, (pd.DataFrame, Symmetric, nx.Graph, nx.DiGraph, nx.OrderedGraph, nx.OrderedDiGraph)), "Must be either a symmetric pd.DataFrame, Symmetric, nx.Graph, or hx.Hive object"
#     if is_graph(data):
#         weights = dict()
#         for edge_data in data.edges(data=True):
#             edge = frozenset(edge_data[:-1])
#             weight = edge_data[-1]["weight"]
#             weights[edge] = weight
#         weights = pd.Series(weights, name="Weights")#.sort_index()
#         data = Symmetric(weights)
#     if isinstance(data, Symmetric):
#         df_dense = condensed_to_dense(data.weights)

#     if isinstance(data, pd.DataFrame):
#         assert is_symmetrical(data, tol=tol)
#         df_dense = data

    if not isinstance(data, pd.DataFrame):
        df_dense = convert_network(data=data, into=pd.DataFrame, remove_missing_values=False, remove_self_interactions=False, tol=tol)
    else:
        assert is_symmetrical(X=data, tol=tol, nans_ok=True)
        df_dense = data.copy()
        
    if remove_self_interactions:
        np.fill_diagonal(df_dense.values, 0)

    #kTotal
    k_total = df_dense.sum(axis=1)
    
    if groups is None:
        return k_total
    else:
        groups = pd.Series(groups)
        data_connectivity = OrderedDict()
        
        data_connectivity["kTotal"] = k_total
        
        #kWithin
        k_within = list()
        for group in groups.unique():
            index_nodes = pd.Index(sorted(set(groups[lambda x: x == group].index) & set(df_dense.index)))
            k_group = df_dense.loc[index_nodes,index_nodes].sum(axis=1)
            k_within.append(k_group)
        data_connectivity["kWithin"] = pd.concat(k_within)
        
        #kOut
        data_connectivity["kOut"] = data_connectivity["kTotal"] - data_connectivity["kWithin"]

        #kDiff
        data_connectivity["kDiff"] = data_connectivity["kWithin"] - data_connectivity["kOut"]

        return pd.DataFrame(data_connectivity)

def density(k:pd.Series):
    """
    Density = sum(khelp)/(nGenes * (nGenes - 1))
    https://github.com/cran/WGCNA/blob/15de0a1fe2b214f7047b887e6f8ccbb1c681e39e/R/Functions.R#L1963
    """
    k_total = k.sum()
    number_of_nodes = k.size
    return k_total/(number_of_nodes * (number_of_nodes - 1))

def centralization(k:pd.Series):
    """
    Centralization = nGenes*(max(khelp)-mean(khelp))/((nGenes-1)*(nGenes-2))
    https://github.com/cran/WGCNA/blob/15de0a1fe2b214f7047b887e6f8ccbb1c681e39e/R/Functions.R#L1965
    """
    k_max = k.max()
    k_mean = k.mean()
    number_of_nodes = k.size
    return number_of_nodes * (k_max - k_mean)/((number_of_nodes - 1) * (number_of_nodes - 2))

def heterogeneity(k:pd.Series):
    """
    Heterogeneity = sqrt(nGenes * sum(khelp^2)/sum(khelp)^2 - 1)
    https://github.com/cran/WGCNA/blob/15de0a1fe2b214f7047b887e6f8ccbb1c681e39e/R/Functions.R#L1967
    """
    number_of_nodes = k.size
    return np.sqrt(number_of_nodes * np.sum(k**2)/np.sum(k)**2 - 1)

# Topological overlap
def topological_overlap_measure(
    data, 
    into=None, 
    node_type=None, 
    edge_type="topological_overlap_measure", 
    association_type="network", 
    assert_symmetry=True, 
    tol=1e-10,
    fill_diagonal=np.nan,
    ):
    """
    Compute the topological overlap for a weighted adjacency matrix
    
    `data` and `into` can be the following network structures/objects:
        * pd.DataFrame (must be symmetrical)
        * Symmetric
        * nx.[Di|Ordered]Graph
    ====================================================
    Benchmark 5000 nodes (iris w/ 4996 noise variables):
    ====================================================
    TOM via rpy2 -> R -> WGCNA: 24 s ± 471 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    TOM via this function: 7.36 s ± 212 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

    =================
    Acknowledgements:
    =================
    Original source:
        * Peter Langfelder and Steve Horvath
        https://www.rdocumentation.org/packages/WGCNA/versions/1.67/topics/TOMsimilarity
        https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-9-559

    Implementation adapted from the following sources:
        * Credits to @scleronomic
        https://stackoverflow.com/questions/56574729/how-to-compute-the-topological-overlap-measure-tom-for-a-weighted-adjacency-ma/56670900#56670900
        * Credits to @benmaier
        https://github.com/benmaier/GTOM/issues/3
    """
    # Compute topological overlap
    def _compute_tom(A):
        # Prepare adjacency
        np.fill_diagonal(A, 0)
        # Prepare TOM
        A_tom = np.zeros_like(A)
        # Compute TOM
        L = np.matmul(A,A)
        ki = A.sum(axis=1)
        kj = A.sum(axis=0)
        MINK = np.array([ np.minimum(ki_,kj) for ki_ in ki ])
        A_tom = (L+A) / (MINK + 1 - A)
        np.fill_diagonal(A_tom,1)
        return A_tom

    # Check input type
    if into is None:
        into = type(data)
        
    node_labels = None
    if not isinstance(data, np.ndarray):
        if not isinstance(data, pd.DataFrame):
            data = convert_network(data, into=pd.DataFrame)
        assert np.all(data.index == data.columns), "`data` index and columns must have identical ordering"
        np.fill_diagonal(data.values,0) #! redundant
        node_labels = data.index

    # Check input type
    if assert_symmetry:
        assert is_symmetrical(data, tol=tol), "`data` is not symmetric"
    assert np.all(data >= 0), "`data` weights must ≥ 0"

    # Compute TOM
    A_tom = _compute_tom(np.asarray(data))
    if assert_symmetry:
        A_tom = (A_tom + A_tom.T)/2

    # Unlabeled adjacency
    if node_labels is None:
        return A_tom

    # Labeled adjacency
    else:
        df_tom = pd.DataFrame(A_tom, index=node_labels, columns=node_labels)
        df_tom.index.name = df_tom.columns.name = node_type
        return convert_network(df_tom, into=into, fill_diagonal=fill_diagonal, assert_symmetry=assert_symmetry, tol=tol, association_type="network", node_type=node_type, edge_type=edge_type)




# =======================================================
# Community Detection
# =======================================================
# Graph community detection
def community_detection(graph, n_iter:int=100, weight:str="weight", random_state:int=0, algorithm="louvain", algo_kws=dict()):
    assert isinstance(n_iter, int)
    assert isinstance(random_state, int)
    assert isinstance(algorithm, str)
    assert_acceptable_arguments(algorithm, {"louvain", "leiden"})
    
    # Louvain    
    if algorithm == "louvain":
        try:
            from community import best_partition
        except ModuleNotFoundError:
            Exception("Please install `python-louvain` to use {} algorithm".format(algorithm))
    
        # Keywords
        _algo_kws = {}
        _algo_kws.update(algo_kws)
        
        graph = convert_network(graph, nx.Graph) 
        
        def partition_function(graph, weight, random_state, algo_kws):
            return best_partition(graph, weight=weight, random_state=random_state, **algo_kws)
        
    # Leiden
    if algorithm == "leiden":
        try:
            from leidenalg import find_partition, ModularityVertexPartition
        except ModuleNotFoundError:
            Exception("Please install `leidenalg` to use {} algorithm".format(algorithm))

        # Convert NetworkX to iGraph
        # graph = ig.Graph.from_networkx(graph)
        graph = convert_network(graph, ig.Graph) 
        nodes_list = np.asarray(graph.vs["name"])
        
        # Keywords
        _algo_kws = {"partition_type":ModularityVertexPartition, "n_iterations":-1}
        _algo_kws.update(algo_kws)
        
        def partition_function(graph, weight, random_state, algo_kws, nodes_list=nodes_list):
            node_to_partition = dict()
            for partition, nodes in enumerate(find_partition(graph, weights=weight, seed=random_state, **algo_kws)):
                mapping = dict(zip(nodes_list[nodes], [partition]*len(nodes)))
                node_to_partition.update(mapping)
            return node_to_partition
    
    # Get partitions
    partitions = dict()
    for rs in pv(range(random_state, n_iter + random_state), "Detecting communities via `{}` algorithm".format(algorithm)):
        partitions[rs] = partition_function(graph=graph, weight=weight, random_state=rs, algo_kws=_algo_kws)
        
    # Create DataFrame
    df = pd.DataFrame(partitions)
    df.index.name = "Node"
    df.columns.name = "Partition"
    return df

# Cluster homogeneity matrix
def cluster_homogeneity(df:pd.DataFrame, edge_type="Edge", iteration_type="Iteration"):
    """
    # Create Graph
    from soothsayer_utils import get_iris_data
    df_adj = get_iris_data(["X"]).iloc[:5].T.corr() + np.random.RandomState(0).normal(size=(5,5))
    graph = nx.from_pandas_adjacency(df_adj)
    graph.nodes()
    # NodeView(('sepal_length', 'sepal_width', 'petal_length', 'petal_width'))

    # Community detection (network clustering)
    df_louvain = community_detection(graph, n_iter=10, algorithm="louvain")
    df_louvain
    # Partition	0	1	2	3	4	5	6	7	8	9
    # Node										
    # iris_0	0	0	0	0	0	0	0	0	0	0
    # iris_1	1	1	1	1	1	1	1	1	1	1
    # iris_2	1	2	2	2	2	1	2	2	2	2
    # iris_3	0	1	1	1	1	0	1	1	1	1
    # iris_4	2	3	3	3	3	2	3	3	3	3

    # Determine cluster homogeneity
    df_homogeneity = cluster_homogeneity(df_louvain)
    df_homogeneity
    # Iteration	0	1	2	3	4	5	6	7	8	9
    # Edge										
    # (iris_1, iris_0)	0	0	0	0	0	0	0	0	0	0
    # (iris_2, iris_0)	0	0	0	0	0	0	0	0	0	0
    # (iris_3, iris_0)	1	0	0	0	0	1	0	0	0	0
    # (iris_4, iris_0)	0	0	0	0	0	0	0	0	0	0
    # (iris_1, iris_2)	1	0	0	0	0	1	0	0	0	0
    # (iris_3, iris_1)	0	1	1	1	1	0	1	1	1	1
    # (iris_4, iris_1)	0	0	0	0	0	0	0	0	0	0
    # (iris_3, iris_2)	0	0	0	0	0	0	0	0	0	0
    # (iris_4, iris_2)	0	0	0	0	0	0	0	0	0	0
    # (iris_4, iris_3)	0	0	0	0	0	0	0	0	0	0

    df_homogeneity.mean(axis=1)[lambda x: x > 0.5]
    # Edge
    # (iris_3, iris_1)    0.8
    # dtype: float64
    """

    # Adapted from @code-different:
    # https://stackoverflow.com/questions/58566957/how-to-transform-a-dataframe-of-cluster-class-group-labels-into-a-pairwise-dataf


    # `x` is a table of (n=nodes, p=iterations)
    nodes = df.index
    iterations = df.columns
    x = df.values
    n,p = x.shape

    # `y` is an array of n tables, each having 1 row and p columns
    y = x[:, None]

    # Using numpy broadcasting, `z` contains the result of comparing each
    # table in `y` against `x`. So the shape of `z` is (n x n x p)
    z = x == y

    # Reshaping `z` by merging the first two dimensions
    data = z.reshape((z.shape[0] * z.shape[1], z.shape[2]))

    # Redundant pairs
    redundant_pairs = list(map(lambda node:frozenset([node]), nodes))

    # Create pairwise clustering matrix
    df_pairs = pd.DataFrame(
        data=data,
        index=pd.Index(list(map(frozenset, product(nodes,nodes))), name=edge_type),
        columns=pd.Index(iterations, name=iteration_type),
        dtype=int,
    ).drop(redundant_pairs, axis=0)


    return df_pairs[~df_pairs.index.duplicated(keep="first")]

# ==============================================================================
# Associations and graph constructors
# ==============================================================================
@check_packages(["umap"])
def umap_fuzzy_simplical_set_graph(
    dism,
    n_neighbors,
    into=pd.Series,
    name=None,
    node_type=None,
    edge_type="membership strength of the 1-simplex",
    random_state=0,
    set_op_mix_ratio=1.0,
    local_connectivity=1.0,
    angular=False,
    apply_set_operations=True,
    verbose=False,
    ):
    # Imports
    from umap.umap_ import fuzzy_simplicial_set, nearest_neighbors
    from scipy.sparse import tril
    
    # Checks
    assert isinstance(dism, (Symmetric, pd.Series, pd.DataFrame)), "`dism` must be a labeled object as either a pandas[pd.DataFrame, pd.Series] or ensemble_networkx[Symmetric]"
    
    # Convert dism to pd.DataFrame
    if not isinstance(dism, pd.DataFrame):
        dism = convert_network(dism, pd.DataFrame)
    nodes = dism.index
    
    # Compute nearest neighbors
    knn_indices, knn_dists, rp_forest = nearest_neighbors(
        X=dism.values, 
        n_neighbors=n_neighbors, 
        metric="precomputed", 
        metric_kwds=None, 
        angular=angular, 
        random_state=random_state,
    )

    # Fuzzy simplical set
    connectivities, sigmas, rhos = fuzzy_simplicial_set(
            X=dism.values,
            n_neighbors=n_neighbors,
            random_state=random_state, 
            metric="precomputed",
            knn_indices=knn_indices,
            knn_dists=knn_dists,
            angular=angular,
            set_op_mix_ratio=set_op_mix_ratio,
            local_connectivity=local_connectivity,
            apply_set_operations=apply_set_operations,
            verbose=verbose,
            return_dists=None,
        )
    
    # Get non-zero edge weights and construct graph as a pd.Series with frozenset edges
    index_sources, index_targets = tril(connectivities).nonzero()
    weights = np.asarray(connectivities[index_sources,index_targets]).ravel()
    data = pd.Series( 
            data = np.asarray(connectivities[index_sources,index_targets]).ravel(),
            index = map(frozenset, zip(nodes[index_sources], nodes[index_targets])),
            name=name,
    )

    if into == pd.Series:
        return data
    else:
        # Get symmetric object
        network = Symmetric(
            data=data, 
            association_type="network", 
            assert_symmetry=False, 
            remove_missing_values=True, 
            name=name, 
            node_type=node_type, 
            edge_type=edge_type,
            )
        
        if into == Symmetric:
            return network
        else:
            return convert_network(network, into)
        
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

# Matthews correlation coefficient
def pairwise_mcc(X:pd.DataFrame, check=True):
    """
    # Description
    Returns a correlation table containing Matthews correlation coefficients for a given matrix (np.array or pd.dataframe) of binary, categorical variables.
    This implementation was created as an alternative to Scikit-Learn's implementation as a measure of dependency reduction.

    # Scikit-Learn implementation of MCC:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.matthews_corrcoef.html
    
    # Parameters
        * X:
            - NumPy array or Pandas dataframe
    Output: 
        pd.DataFrame or np.array of pairwise MCC values
    """
    # Checks
    if check:
        n_dimensions = len(X.shape)
        assert n_dimensions in {2}, "`X` must be 2D"
        assert np.all(X == X.astype(bool)), "`X` must be either dtype boolean or integers[0,1]"

    # Convert input data to a NumPy array
    # index=None
    components=None
    if isinstance(X, pd.DataFrame):
        # index = X.index
        features = X.columns
        X = X.values
        
    X = X.astype(bool)
    
    # Shape of matrix
    n,m = X.shape

    # Calculate pairwise MCC values
    N11 = (X[:, None] & X[:, :, None]).sum(axis=0)
    N10 = (X[:, None] & ~X[:, :, None]).sum(axis=0)
    N01 = (~X[:, None] & X[:, :, None]).sum(axis=0)
    N00 = (~X[:, None] & ~X[:, :, None]).sum(axis=0)
    denominator = np.sqrt((N11 + N10) * (N11 + N01) * (N00 + N10) * (N00 + N01))
    denominator[denominator == 0] = 1  # Handle division by zero case

    output = (N11 * N00 - N10 * N01) / denominator

    # Fill the lower triangular part with the symmetric values
    output[np.tril_indices(m, k=-1)] = output.T[np.tril_indices(m, k=-1)]

    # Set diagonal elements to 1.0
    np.fill_diagonal(output, 1.0)

    # Return the result as a DataFrame if the input was a DataFrame
    if features is not None:
        output = pd.DataFrame(output, index=features, columns=features)
        
    return output

# =======================================================
# Classes
# =======================================================
# Symmetrical dataframes represented as augment pd.Series
class Symmetric(object):
    """
    An indexable symmetric matrix stored as the lower triangle for space.

    Usage:
    import soothsayer_utils as syu
    import ensemble_networkx as enx

    # Load data
    X, y, colors = syu.get_iris_data(["X", "y", "colors"])
    n, m = X.shape

    # Get association matrix (n,n)
    method = "pearson"
    df_sim = X.T.corr(method=method)
    ratio = 0.382
    number_of_edges = int((n**2 - n)/2)
    number_of_edges_negative = int(ratio*number_of_edges)

    # Make half of the edges negative to showcase edge coloring (not statistically meaningful at all)
    for a, b in zip(np.random.RandomState(0).randint(low=0, high=149, size=number_of_edges_negative), np.random.RandomState(1).randint(low=0, high=149, size=number_of_edges_negative)):
        if a != b:
            df_sim.values[a,b] = df_sim.values[b,a] = df_sim.values[a,b]*-1

    # Create a Symmetric object from the association matrix
    sym_iris = enx.Symmetric(data=df_sim, node_type="iris sample", edge_type=method, name="iris", association_type="network")
    # ====================================
    # Symmetric(Name:iris, dtype: float64)
    # ====================================
    #     * Number of nodes (iris sample): 150
    #     * Number of edges (correlation): 11175
    #     * Association: network
    #     * Memory: 174.609 KB
    #     --------------------------------
    #     | Weights
    #     --------------------------------
    #     (iris_1, iris_0)        0.995999
    #     (iris_0, iris_2)        0.999974
    #     (iris_3, iris_0)        0.998168
    #     (iris_0, iris_4)        0.999347
    #     (iris_0, iris_5)        0.999586
    #                               ...   
    #     (iris_148, iris_146)    0.988469
    #     (iris_149, iris_146)    0.986481
    #     (iris_147, iris_148)    0.995708
    #     (iris_149, iris_147)    0.994460
    #     (iris_149, iris_148)    0.999916
    =====
    devel
    =====
    2022-Feb-12
    * Completely rewrote Symmetric to clean it up. A little slower but much easier to prototype
    
    2022-Feb-08
    * Added support for iGraph
    * Added support for non-fully-connected graphs
    
    2020-June-23
    * Replace self._dense_to_condensed to dense_to_condensed
    * Dropped math operations
    * Added input for Symmetric or pd.Series with a frozenset index

    2018-August-16
    * Added __add__, __sub__, etc.
    * Removed conversion to dissimilarity for tree construction
    * Added .iteritems method
    

    Future:
    * Add support for non-fully-connected networks
    * Add suport for iGraph
    
    Dropped:
    Fix the diagonal arithmetic
    """
    def __init__(
        self, 
        data, 
        name=None, 
        node_type=None, 
        edge_type=None, 
        func_metric=None,  
        association_type=None, 
        
        # Symmetry
        assert_symmetry=True, 
        tol=None, 
        diagonal=None,

        # Missing values
        remove_missing_values=True,
        fill_missing_values_with=np.nan,
        nans_ok=True, 
        
        # Attributes
        **attrs,
        ):
        
        # Metadata
        self.metadata = dict()
        
        # Association type
        assert_acceptable_arguments(association_type, {"similarity", "dissimilarity", "statistical_test", "network", None})

        
        # Keywords
        
        kwargs = {"name":name, "node_type":node_type, "edge_type":edge_type, "func_metric":func_metric, "association_type":association_type}

        # From Symmetric object
        # if input_category == "Symmetric":
        if isinstance(data, type(self)):
            if not nans_ok:
                assert not np.any(data.weights.isnull()), "Cannot move forward with missing values"
            self.__dict__.update(data.__dict__)
            
            # Set keywords
            for k, v in kwargs.items():
                if v is not None:
                    setattr(self, k, v)
            
        # Convert network type
        else:
            self.weights = convert_network(
                data=data, 
                into=pd.Series, 
                remove_self_interactions=True, 
                assert_symmetry=assert_symmetry, 
                tol=tol, 
                remove_missing_values=remove_missing_values, 
                fill_missing_values_with=fill_missing_values_with,
                name="Weights",
            )
            self.edges = pd.Index(self.weights.index, name="Edges")
            self.nodes = pd.Index(sorted(frozenset.union(*self.weights.index)), name="Nodes")
            
            # Set keywords
            for k, v in kwargs.items():
                setattr(self, k, v)

        # If there's still no `edge_type` and `func_metric` is not empty, then use this the name of `func_metric`
        if (self.edge_type is None) and (self.func_metric is not None):
            self.edge_type = self.func_metric.__name__
        
        self.set_diagonal(diagonal)
        self.values = self.weights.values
        self.number_of_nodes = self.nodes.size
        self.number_of_edges = self.edges.size
        self.memory = self.weights.memory_usage()
        self.metadata.update(attrs)
        self.__synthesized__ = datetime.datetime.utcnow()
        
        
    # Setting the diagonal
    def set_diagonal(self, diagonal):
        if diagonal is None:
            self.diagonal = None
        else:
            if is_number(diagonal):
                diagonal = dict_build([(diagonal, self.nodes)])
            assert is_dict_like(diagonal), "`diagonal` must be dict-like"
            assert set(diagonal.keys()) >= set(self.nodes), "Not all `nodes` are in `diagonal`"
            self.diagonal =  pd.Series(diagonal, name="Diagonal")[self.nodes]
            
    # =======
    # Built-in
    # =======
    def __repr__(self):
        pad = 4
        header = format_header("Symmetric(Name:{}, dtype: {})".format(self.name, self.weights.dtype),line_character="=")
        n = len(header.split("\n")[0])
        expected_edges_if_square = 0.5*(self.number_of_nodes**2 - self.number_of_nodes)
        fields = [
            header,
            pad*" " + "* Number of nodes ({}): {}".format(self.node_type, self.number_of_nodes),
            pad*" " + "* Number of edges ({}): {}".format(self.edge_type, self.number_of_edges),
            pad*" " + "* Association: {}".format(self.association_type),
            pad*" " + "* is_square: {}".format(expected_edges_if_square == self.number_of_edges),
            pad*" " + "* Memory: {}".format(format_memory(self.memory)),

            *map(lambda line:pad*" " + line, format_header("| Weights", "-", n=n-pad).split("\n")),
            *map(lambda line: pad*" " + line, repr(self.weights).split("\n")[1:-1]),
            ]

        return "\n".join(fields)
    
    def __getitem__(self, key):
        """
        `key` can be a node or non-string iterable of edges
        """
    
        if is_nonstring_iterable(key):
            # assert len(key) >= 2, "`key` must have at least 2 identifiers. e.g. ('A','B')"
            key = frozenset(key)
            
            # Is it a diagonal value
            if len(key) == 1:
                assert self.diagonal is not None, "Please use `set_diagonal` before querying single node edge weights"
                return self.diagonal[list(key)[0]]
            else:
                # A list of nodes
                if len(key) > 2:
                    key = list(map(frozenset, combinations(key, r=2)))
            return self.weights[key]
        else:
            # Get all edges connected to a node
            if key in self.nodes:
                s = frozenset([key])
                mask = self.edges.map(lambda x: bool(s & x))
                return self.weights[mask]
            else:
                raise KeyError("{} not in node list".format(key))
        
    def __call__(self, key, func=np.sum):
        """
        This can be used for connectivity in the context of networks but can be confusing with the versatiliy of __getitem__
        """
        if hasattr(key, "__call__"):
            return self.weights.groupby(key).apply(func)
        else:
            return func(self[key])
        
    def __len__(self):
        return self.number_of_edges
    def __iter__(self):
        for v in self.weights:
            yield v
    def items(self):
        return self.weights.items()
    def iteritems(self):
        return self.weights.iteritems()
    def keys(self):
        return self.weights.keys()
    def apply(self, func):
        return func(self.weights)
    def mean(self):
        return self.weights.mean()
    def median(self):
        return self.weights.median()
    def min(self):
        return self.weights.min()
    def max(self):
        return self.weights.max()
    def idxmin(self):
        return self.weights.idxmin()
    def idxmax(self):
        return self.weights.idxmax()
    def sum(self):
        return self.weights.sum()
    def sem(self):
        return self.weights.sem()
    def var(self):
        return self.weights.var()
    def std(self):
        return self.weights.std()
    def describe(self, **kwargs):
        return self.weights.describe(**kwargs)
    def map(self, func):
        return self.weights.map(func)
    def entropy(self, base=2):
        assert np.all(self.weights > 0), "All weights must be greater than 0"
        return stats.entropy(self.weights, base=base)
    
    # ==========
    # Network Metrics
    # ==========
    def edge_connectivity(self, node_subgraph=None, edge_subgraph=None, remove_self_interactions=True):
        return self.to_pandas_series(node_subgraph=node_subgraph, edge_subgraph=edge_subgraph)
    
    def node_connectivity(self, node_subgraph=None, edge_subgraph=None, groups:pd.Series=None, remove_self_interactions=True):
        """
        Total node_connectivity is twice that of total edge_connectivity because of symmetry
        """
        data = self.to_pandas_dataframe(node_subgraph=node_subgraph, edge_subgraph=edge_subgraph)
        assert np.all(data.values >= 0), "To use network metrics such as connectivity, density, centralization, and heterogeneity. All weights must ≥ 0."
        return connectivity(data=data, groups=groups, remove_self_interactions=remove_self_interactions)
    
    def density(self, node_subgraph=None, edge_subgraph=None,  remove_self_interactions=True):
        data = self.to_pandas_dataframe(node_subgraph=node_subgraph, edge_subgraph=edge_subgraph)
        k = connectivity(data=data, remove_self_interactions=remove_self_interactions)
        return density(k)
    
    def centralization(self, node_subgraph=None, edge_subgraph=None,  remove_self_interactions=True):
        data = self.to_pandas_dataframe(node_subgraph=node_subgraph, edge_subgraph=edge_subgraph)
        k = connectivity(data=data, remove_self_interactions=remove_self_interactions)
        return centralization(k)
    
    def heterogeneity(self, node_subgraph=None, edge_subgraph=None,  remove_self_interactions=True):
        data = self.to_pandas_dataframe(node_subgraph=node_subgraph, edge_subgraph=edge_subgraph)
        k = connectivity(data=data, remove_self_interactions=remove_self_interactions)
        return heterogeneity(k)
    
    def network_summary_statistics(self, node_subgraph=None, edge_subgraph=None,remove_self_interactions=True):
        data = self.to_pandas_dataframe(node_subgraph=node_subgraph, edge_subgraph=edge_subgraph)
        k = connectivity(data=data, remove_self_interactions=remove_self_interactions)
        return pd.Series(OrderedDict([
            
            ("node_connectivity(∑)", k.sum()),
            ("node_connectivity(µ)", k.mean()),
            ("node_connectivity(σ)", k.std()),
            ("density", density(k)),
            ("centralization", centralization(k)),
            ("heterogeneity", heterogeneity(k)),
        ]), name=self.name)
    
    def topological_overlap_measure(self, into=pd.Series, node_subgraph=None, edge_subgraph=None, fill_diagonal=np.nan):
        return topological_overlap_measure(
            data=self.to_pandas_dataframe(node_subgraph=node_subgraph, edge_subgraph=edge_subgraph, fill_diagonal=np.nan), 
            fill_diagonal=fill_diagonal,
            node_type=self.node_type,
        )
    # ==========
    # Conversion
    # ==========
    # def to_dense(self, node_subgraph=None, fill_diagonal=None):
    #     if fill_diagonal is None:
    #         fill_diagonal = self.diagonal
    #     if node_subgraph is None:
    #         node_subgraph = self.nodes
    #     return condensed_to_dense(y=self.weights, fill_diagonal=fill_diagonal, index=node_subgraph)
    
    def to_pandas_dataframe(self, node_subgraph=None, edge_subgraph=None, fill_diagonal=None, vertical=False, **convert_network_kws):
        # df = self.to_dense(node_subgraph=node_subgraph, fill_diagonal=fill_diagonal)
        if fill_diagonal is None:
            fill_diagonal = self.diagonal
        if (node_subgraph is None) & (edge_subgraph is None):
            node_subgraph = self.nodes
        
        df = convert_network(
            data=self.weights, 
            into=pd.DataFrame,
            node_subgraph=node_subgraph,
            edge_subgraph=edge_subgraph,
            fill_diagonal=fill_diagonal,
            **convert_network_kws,
        )
            
        if not vertical:
            return df
        else:
            df = df.stack().to_frame().reset_index()
            df.columns = ["Node_A", "Node_B", "Weight"]
            df.index.name = "Edge_Index"
        return df

#     def to_condensed(self):
#         return self.weights
                                     
    def to_pandas_series(self, node_subgraph=None, edge_subgraph=None, **convert_network_kws):
        return convert_network(
            data=self.weights, 
            into=pd.Series,
            node_subgraph=node_subgraph,
            edge_subgraph=edge_subgraph,
            **convert_network_kws,
        )
#     @check_packages(["ete3", "skbio"])
#     def to_tree(self, method="average", into=None, node_prefix="y"):
#         assert self.association == "dissimilarity", "`association` must be 'dissimilarity' to construct tree"
#         if method in {"centroid", "median", "ward"}:
#             warnings.warn("Methods ‘centroid’, ‘median’, and ‘ward’ are correctly defined only if Euclidean pairwise metric is used.\nSciPy Documentation - https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html#scipy.cluster.hierarchy.linkage") 
#         if into is None:
#             into = ete3.Tree
#         if not hasattr(self,"Z"):
#             self.Z = linkage(self.weights.values, metric="precomputed", method=method)
#         if not hasattr(self,"newick"):
#             self.newick = linkage_to_newick(self.Z, self.nodes)
#         tree = into(newick=self.newick, name=self.name)
#         return name_tree_nodes(tree, node_prefix)

    def to_networkx(self, into=None, node_subgraph=None, edge_subgraph=None, **attrs):
        if into is None:
            into = nx.Graph
        metadata = {"name":self.name, "node_type":self.node_type, "edge_type":self.edge_type, "func_metric":self.func_metric, "association_type":self.association_type}
        metadata.update(attrs)
#         graph = into(name=self.name, **metadata)

#         for (node_A, node_B), weight in self.weights.iteritems():
#             graph.add_edge(node_A, node_B, weight=weight)
        graph = convert_network(
            data=self.weights, 
            into=into,
            node_subgraph=node_subgraph,
            edge_subgraph=edge_subgraph,
        )
        graph.graph.update(metadata)
        return graph
    
    def to_igraph(self, directed=False, node_subgraph=None, edge_subgraph=None, **attrs):
        metadata = {"name":self.name, "node_type":self.node_type, "edge_type":self.edge_type, "func_metric":self.func_metric, "association_type":self.association_type}
        metadata.update(attrs)
        
        # graph = ig.Graph(directed=directed)
        # graph.add_vertices(self.nodes)  # this adds adjacency.shape[0] vertices
        # graph.add_edges(self.edges.map(tuple))
        # graph.es['weight'] = self.weights.values
        
        graph = convert_network(
            data=self.weights, 
            into=ig.Graph,
            node_subgraph=node_subgraph,
            edge_subgraph=edge_subgraph,
        )
        if directed:
            graph.to_directed()
        
        for k, v in metadata.items():
            graph[k] = v
            
        return graph
    
    def to_file(self, path, **kwargs):
        write_object(obj=self, path=path, **kwargs)

    def copy(self):
        return copy.deepcopy(self)


# =============================
# Feature Engineering
# =============================
class CategoricalEngineeredFeature(object):
    """
    Combine features using multiple categories.
    
    # =========================================
    from soothsayer_utils import get_iris_data
    from scipy import stats
    import pandas as pd
    import ensemble_networkx as enx

    X, y = get_iris_data(["X","y"])
    # Usage
    CEF = enx.CategoricalEngineeredFeature(name="Iris", observation_type="sample")


    # Add categories
    category_1 = pd.Series(X.columns.map(lambda x:x.split("_")[0]), X.columns)
    CEF.add_category(
        name_category="leaf_type", 
        mapping=category_1,
    )
    # Optionally add scaling factors, statistical tests, and summary statistics
    # Compile all of the data
    CEF.compile(scaling_factors=X.sum(axis=0), stats_tests=[stats.normaltest])
    # Unpacking engineered groups: 100%|██████████| 1/1 [00:00<00:00, 2974.68it/s]
    # Organizing feature sets: 100%|██████████| 4/4 [00:00<00:00, 17403.75it/s]
    # Compiling synopsis [Basic Feature Info]: 100%|██████████| 2/2 [00:00<00:00, 32768.00it/s]
    # Compiling synopsis [Scaling Factor Info]: 100%|██████████| 2/2 [00:00<00:00, 238.84it/s]

    # View the engineered features
    CEF.synopsis_
    # 	initial_features	number_of_features	leaf_type(level:0)	scaling_factors	sum(scaling_factors)	mean(scaling_factors)	sem(scaling_factors)	std(scaling_factors)
    # leaf_type								
    # sepal	[sepal_width, sepal_length]	2	sepal	[458.6, 876.5]	1335.1	667.55	208.95	208.95
    # petal	[petal_length, petal_width]	2	petal	[563.7, 179.90000000000003]	743.6	371.8	191.9	191.9

    # Transform a dataset using the defined categories
    CEF.fit_transform(X, aggregate_fn=np.sum)
    # leaf_type	sepal	petal
    # sample		
    # iris_0	8.6	1.6
    # iris_1	7.9	1.6
    # iris_2	7.9	1.5
    # iris_3	7.7	1.7

    """
    def __init__(self,
                 initial_feature_type=None,
                 engineered_feature_type=None,
                 observation_type=None,
                 unit_type=None,
                 name=None,
                 description=None,
                 assert_mapping_intersection=False,
                ):
        self.initial_feature_type = initial_feature_type
        self.engineered_feature_type=engineered_feature_type
        self.observation_type = observation_type
        self.unit_type = unit_type
        self.name = name
        self.description = description
        self.assert_mapping_intersection = assert_mapping_intersection
        self.__data__ = dict()
        self.number_of_levels_ = 0
        self.memory_ = sys.getsizeof(self)
        self.compiled_ = False
        
    def add_category(self, name_category:Hashable, mapping:Union[Mapping, pd.Series], level:int="infer", assert_mapping_exclusiveness=False, assert_level_nonexistent=True):
        if level == "infer":
            level = len(self.__data__)
        assert name_category not in self.__data__, "Already added category: {}".format(name_category)
        assert isinstance(mapping, (Mapping, pd.Series)), "`mapping` must be dict-like"
        # Force iterables into set type
        def f(x):
            # How should this handle frozensets and tuples?
            if is_nonstring_iterable(x) and not isinstance(x, Hashable):
                if assert_mapping_exclusiveness:
                    raise AssertionError("name_category=`{}` must have one-to-one mapping exclusiveness.  If this is not desired, please set `assert_mapping_exclusiveness=False` when adding component via `add_category`".format(name_category))
                x = set(x)
            return x
        # Add categories
        if assert_level_nonexistent:
            assert level not in self.__data__, "`level={}` already existent".format(level)
        self.__data__[level] = {
            "name_category":name_category,
            "mapping":pd.Series(mapping).map(f)
        }
        

        return self
    
    # Compile all the categories
    def compile(
        self,
        scaling_factors:pd.Series=None, # e.g. Gene Lengths,
        stats_summary = [np.sum, np.mean, stats.sem, np.std],
        stats_tests = [],
        ):
        
        # Check features
        def check_initial_features():
            self.initial_features_union_ = set.union(*map(lambda dict_values: set(dict_values["mapping"].index), self.__data__.values()))
            self.initial_features_intersection_ = set.intersection(*map(lambda dict_values: set(dict_values["mapping"].index), self.__data__.values()))
            if self.assert_mapping_intersection:
                assert self.initial_features_union_ == self.initial_features_intersection_, \
                "All `mapping` must have same features mapped.  features_union = {}; features_intersection = {}".format(
                    len(self.initial_features_union_), 
                    len(self.initial_features_intersection_),
                )
            if scaling_factors is not None:
                assert isinstance(scaling_factors, (Mapping, pd.Series)), "`scaling_factors` must be dict-like"
                self.scaling_factors_ = pd.Series(scaling_factors)
                assert set(self.initial_features_intersection_) <= set(self.scaling_factors_.index), "`scaling_factors` does not have all required `initial_features_intersection_`.  In particular, the following number of features are missing:\n{}".format(len(self.initial_features_intersection_ - set(query_features)))
                
            else:
                self.scaling_factors_ = None
                
            
        # Organizing features and groups
        def organize_and_group_initial_features():
            # Organize the data w/ respect to feature
            feature_to_grouping = defaultdict(lambda: defaultdict(set))
            for level, data in pv(sorted(self.__data__.items(), key=lambda item:item[0]), description="Unpacking engineered groups"):
                name_category = data["name_category"]
                for id_feature, values in data["mapping"].items():
                    if isinstance(values, Hashable):
                        values = set([values])
                    for v in values:
                        feature_to_grouping[id_feature][level].add(v)

            # Organize the groups and create sets of features
            self.engineered_to_initial_features_ = defaultdict(set)
            self.initial_features_ = set(feature_to_grouping.keys())
            for id_feature, grouping in pv(feature_to_grouping.items(), description="Organizing feature sets"):
                grouping_iterables = map(lambda item: item[1], sorted(grouping.items(), key=lambda item: item[0]))
                for engineered_feature in product(*grouping_iterables):
                    if len(engineered_feature) == self.number_of_levels_:
                        self.engineered_to_initial_features_[engineered_feature].add(id_feature)
        
        # Compute synopsis
        def get_synopsis():
            name_to_level = dict(map(lambda item: (item[1]["name_category"], item[0]), self.__data__.items()))
            self.synopsis_ = defaultdict(dict)
            for engineered_feature, initial_features in pv(self.engineered_to_initial_features_.items(), description="Compiling synopsis [Basic Feature Info]"):
                self.synopsis_[engineered_feature]["initial_features"] = list(initial_features)
                self.synopsis_[engineered_feature]["number_of_features"] = len(initial_features)                
                for i, value in enumerate(engineered_feature):
                    level = self.levels_[i]
                    name_category = self.__data__[level]["name_category"]
                    self.synopsis_[engineered_feature]["{}(level:{})".format(name_category, level)] = value

            if self.scaling_factors_ is not None:
                for engineered_feature in pv(self.synopsis_.keys(), description="Compiling synopsis [Scaling Factor Info]"):
                    initial_features = self.synopsis_[engineered_feature]["initial_features"]
                    query_scaling_factors = self.scaling_factors_[initial_features]
                    self.synopsis_[engineered_feature]["scaling_factors"] = list(query_scaling_factors)
                    
                    for func in stats_summary:
                        with Suppress():
                            self.synopsis_[engineered_feature]["{}(scaling_factors)".format(func.__name__)] = func(query_scaling_factors)
                        
                    for func in stats_tests:
                        with Suppress():
                            try:
                                stat, p = func(query_scaling_factors)
                                self.synopsis_[engineered_feature]["{}|stat(scaling_factors)".format(func.__name__)] = stat
                                self.synopsis_[engineered_feature]["{}|p_value(scaling_factors)".format(func.__name__)] = p
                            except:
                                pass


            self.synopsis_ = pd.DataFrame(self.synopsis_).T
            if isinstance(self.synopsis_.index, pd.MultiIndex):
                self.synopsis_.index.names = map(lambda item: item[1]["name_category"], sorted(self.__data__.items(), key=lambda item:item[0]))
                        
        # Basic Info
        self.levels_ = list(self.__data__.keys())
        self.number_of_levels_ = len(self.__data__)
        if stats_summary is None:
            stats_summary = []
        if stats_tests is None:
            stats_tests = []
            
        # Run compilation
        print(format_header("CategoricalEngineeredFeature(Name:{})".format(self.name),line_character="="), file=sys.stderr)
        check_initial_features()
        organize_and_group_initial_features()
        get_synopsis()   
        self.stats_summary_ = stats_summary
        self.stats_tests_ = stats_tests
        self.memory_ = sys.getsizeof(self)
        self.compiled_ = True
        return self
    
    # Transform a dataset
    def fit_transform(
        self, 
        X:pd.DataFrame,         
        aggregate_fn=np.sum,
        ) -> pd.DataFrame:
        query_features = set(X.columns)
        assert query_features >= self.initial_features_, "X.columns does not have all required `initial_features_`.  In particular, the following number of features are missing:\n{}".format(len(self.initial_features_ - query_features))
        
        # Aggregate features
        results = dict()
        for engineered_feature, initial_features in pv(self.engineered_to_initial_features_.items(), description="Aggregating engineered features"):
            X_subset = X[initial_features]
            aggregate = X_subset.apply(aggregate_fn, axis=1)
            results[engineered_feature] = aggregate
        df_aggregate = pd.DataFrame(results)
        
        # Properly label MultiIndex
        if isinstance(df_aggregate.columns, pd.MultiIndex):
            df_aggregate.columns.names = map(lambda item: item[1]["name_category"], sorted(self.__data__.items(), key=lambda item:item[0]))
#         df_aggregate.columns.names = self.synopsis_.index.names
        df_aggregate.index.name = self.observation_type
        return df_aggregate
    
    # =======
    # Built-in
    # =======

    def __repr__(self):
        pad = 4
        n_preview = 5
        header = format_header("CategoricalEngineeredFeature(Name:{})".format(self.name),line_character="=")
        n = len(header.split("\n")[0])
        fields = [
            header,
            pad*" " + "* Number of levels: {}".format(self.number_of_levels_),
            pad*" " + "* Memory: {}".format(format_memory(self.memory_)),
            pad*" " + "* Compiled: {}".format(self.compiled_),
            ]
        # Types
        fields += [
            *map(lambda line:pad*" " + line, format_header("| Types", "-", n=n-pad).split("\n")),
            pad*" " + "* Initial feature type: {}".format(self.initial_feature_type),
            pad*" " + "* Engineered feature type: {}".format(self.engineered_feature_type),
            pad*" " + "* Observation feature type: {}".format(self.observation_type),
            pad*" " + "* Unit type: {}".format(self.unit_type),
        ]

        if self.compiled_:
            fields += [
                *map(lambda line:pad*" " + line, format_header("| Statistics", "-", n=n-pad).split("\n")),
                2*pad*" " + "Scaling Factors: {}".format(self.scaling_factors_ is not None),
                2*pad*" " + "Summary: {}".format(list(map(lambda fn: fn.__name__, self.stats_summary_))),
                2*pad*" " + "Tests: {}".format(list(map(lambda fn: fn.__name__, self.stats_tests_))),

            ]
            fields += [
                *map(lambda line:pad*" " + line, format_header("| Categories", "-", n=n-pad).split("\n")),
            ]
            for level, d in self.__data__.items():
                fields += [
                    pad*" " + "* Level {} - {}:".format(level, d["name_category"]),
                    2*pad*" " + "Number of initial features: {}".format(d["mapping"].index.nunique()),
                    2*pad*" " + "Number of categories: {}".format(len(flatten(d["mapping"].values, into=set))), 
                ]
            fields += [
                *map(lambda line:pad*" " + line, format_header("| Features", "-", n=n-pad).split("\n")),
            ]
                
            fields += [
                pad*" " + 2*" " + "Number of initial features (Intersection): {}".format(len(self.initial_features_intersection_)),
                pad*" " + 2*" " + "Number of initial features (Union): {}".format(len(self.initial_features_union_)),
                pad*" " + 2*" " + "Number of engineered features: {}".format(len(self.engineered_to_initial_features_)),                
            ]


        return "\n".join(fields)
    
    def __getitem__(self, key):
        """
        `key` can be a node or non-string iterable of edges
        """
        recognized = False
        if isinstance(key, int):
            try:
                recognized = True
                return self.__data__[key]
            except KeyError:
                raise KeyError("{} level not in self.__data__".format(key))
        if isinstance(key, tuple):
            assert self.compiled_, "Please compile before using self.__getitem__ method."
            try:
                recognized = True
                return self.engineered_to_initial_features_[key]
            except KeyError:
                raise KeyError("{} engineered feature not in self.engineered_to_initial_features_".format(key))
        if not recognized:
            raise KeyError("Could not interpret key: {}. Please use self.__getitem__ method for querying level data with an int or features with a tuple.".format(key))
        
    def __len__(self):
        return len(self.engineered_to_initial_features_)
    def __iter__(self):
        for v in self.engineered_to_initial_features_.items():
            yield v
    def items(self):
        return self.engineered_to_initial_features_.items()
    def iteritems(self):
        for v in self.engineered_to_initial_features_.items():
            yield v
            
    def to_file(self, path, **kwargs):
        write_object(obj=self, path=path, **kwargs)

    def copy(self):
        return copy.deepcopy(self)

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

        acceptable_metrics = {"rho", "phi", "biweight_midcorrelation", "spearman", "pearson", "kendall","mcc"}
        if isinstance(metric, str):
            assert_acceptable_arguments(metric, acceptable_metrics)
            metric_name = metric
            if metric == "rho":
                metric = pairwise_rho
            if metric == "phi":
                metric = pairwise_phi
            if metric == "biweight_midcorrelation":
                metric = pairwise_biweight_midcorrelation
            if metric == "mcc":
                metric = pairwise_mcc
            if metric in {"spearman", "pearson", "kendall"}:
                association = metric
                metric = lambda X: self._pandas_association(X=X, metric=association)
                # def metric(X, metric):
                #     return self._pandas_association(X=X, metric=association)

                
        assert hasattr(metric, "__call__"), "`metric` must be either one of the following: [{}], \
        a custom metric that returns an association (set `function_is_pairwise=False`), or a custom \
        metric that returns a 2D square/symmetric pd.DataFrame (set `function_is_pairwise=True`)".format(acceptable_metrics)
        # Transformations
        acceptable_transformations = {"abs"}
        if transformation:
            if isinstance(transformation, str):
                assert_acceptable_arguments(transformation, acceptable_transformations)
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
            association_type="network", 
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
        include_reference_for_samplespecific=True,
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
        y = y[X.index].astype(str)

        # Data
        self.classes_ = y.value_counts()

        if copy_X:
            self.X_ = X.copy()
            self.X_memory_ = X.memory_usage().sum()
            self.memory_ += self.X_memory_ 

        if copy_y:
            self.y_ = y.copy()

        # Checks for index
        if isinstance(X.index, pd.MultiIndex): # https://github.com/jolespin/ensemble_networkx/issues/2
            warnings.warn("SampleSpecificPerturbationNetwork cannot work with MultiIndex or tupled index.  Converting MultiIndex to flat index strings. (i.e., X.index = y.index = X.index.to_flat_index().map(str))")
            X.index = y.index = X.index.to_flat_index().map(str)
        X.index.name = y.index.name = None # https://github.com/jolespin/ensemble_networkx/issues/1

        if isinstance(X.columns, pd.MultiIndex): # https://github.com/jolespin/ensemble_networkx/issues/2
            warnings.warn("SampleSpecificPerturbationNetwork cannot work with MultiIndex or tupled index.  Converting MultiIndex to flat index strings (i.e., X.columns = X.columns.to_flat_index().map(str))")
            X.columns = X.columns.to_flat_index().map(str)
        X.columns.name = None # https://github.com/jolespin/ensemble_networkx/issues/1

        if include_reference_for_samplespecific:
            y_clone = y[y == reference]
            y_clone[y_clone == reference] = f"Reference({reference}[clone])"
            y_clone.index = y_clone.index.map(lambda x: f"{x}[clone]")
            y = pd.concat([y, y_clone])

            X_clone = X.loc[y[y == reference].index]
            X_clone.index = X_clone.index.map(lambda x: f"{x}[clone]")
            X = pd.concat([X, X_clone], axis=0)

            reference = f"Reference({reference}[clone])"

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
                        "Samples":list(index_samplespecific), 
                        "Iterations":list(range(n_iter)),
                        "Edges":list(edges), 
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
            association_type="network", 
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
        stats_tests_comparative = [stats.mannwhitneyu],
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
        index_treatment = y[lambda i: i == treatment].index #sorted(set(X.index) - set(index_reference))
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
            for func in pv(stats_comparative, description="Computing comparative statistics", unit="stat"):  
                for i in range(number_of_edges):
                    u = ensemble_reference.ensemble_.values[:,i]
                    v = ensemble_treatment.ensemble_.values[:,i]
                    try:
                        stat = func(u, v)
                    except ValueError:
                        stat = np.nan
                    self.stats_comparative_[i,k] = stat
                k += 1

        if stats_tests_comparative:# is not None:
            for func in pv(stats_tests_comparative, description="Computing comparative tests", unit="stat"):
                for i in range(number_of_edges):
                    u = ensemble_reference.ensemble_.values[:,i]
                    v = ensemble_treatment.ensemble_.values[:,i]
                    try:
                        stat, p = func(u, v)
                    except ValueError:
                        stat, p = np.nan, np.nan
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
        for func in pv(stats_differential, description="Computing differential", unit="stat"):
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
            association_type="network", 
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

